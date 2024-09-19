import math
import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import CrossEntropyLoss, MSELoss
from omegaconf import DictConfig
from pytorch_lightning.utilities import grad_norm

from functional.loss import get_per_layer_spike_probs
from models.alif import EFAdLIF, SEAdLIF
from models.li import LI
from models.lif import LIF
from models.rnn import LSTMCellWrapper


layer_map = {
    "lif": LIF,
    "se_adlif": SEAdLIF,
    "ef_adlif": EFAdLIF,
    'lstm': LSTMCellWrapper,
}


class MLPSNN(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        print(cfg)
        self.ignore_target_idx = -1
        self.two_layers = cfg.two_layers
        self.output_size = cfg.dataset.num_classes
        self.tracking_metric = cfg.tracking_metric
        self.tracking_mode = cfg.tracking_mode
        self.lr = cfg.lr 

        # For learning rate scheduling (used for oscillation task)
        self.factor = cfg.factor
        self.patience = cfg.patience

        self.auto_regression =  cfg.get('auto_regression', False)
        self.output_size = cfg.dataset.num_classes
        self.batch_size = cfg.dataset.batch_size

        # Define the model
        self.cell = layer_map[cfg.cell]
        self.l1 = self.cell(cfg)
        self.dropout = cfg.dropout
        if cfg.two_layers:
            cfg.input_size = cfg.n_neurons
            self.l2 = self.cell(cfg)
        cfg.input_size = cfg.n_neurons
        self.out_layer = LI(cfg)
        
        self.output_func = cfg.get('loss_agg', 'softmax')
        self.init_metrics_and_loss()
        self.save_hyperparameters()

    # @torch.compile
    def forward(
        self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        self.states = []
        s1 = self.l1.initial_state(inputs.shape[0], inputs.device)
        s1_list = [s1]
        
        s_out = self.out_layer.initial_state(inputs.shape[0], inputs.device)
        s_out_list = [s_out,]
        if self.two_layers:
            s2 = self.l2.initial_state(inputs.shape[0], inputs.device)
            s2_list = [s2,]
        out_sequence = []
        single_step_prediction_limit = int(math.ceil(inputs.shape[1] * 0.5))
        
        # Iterate over each time step in the data
        for t, x_t in enumerate(inputs.unbind(1)):
            # Auto-regression for oscillator task
            if self.auto_regression and t >= single_step_prediction_limit:
                x_t = out.detach()
            out, s1 = self.l1(x_t, s1)
            s1_list.append(s1)
            out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)
            if self.two_layers:
                out, s2 = self.l2(out, s2)
                out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)
                s2_list.append(s2)
            out, s_out = self.out_layer(out, s_out)
            s_out_list.append(s_out)
            # out[:,0] += 100
            out_sequence.append(out)
        # s_list a list of tuples [(u_0, z_0, w_0), (u_1, z_1, w_1), ..., (u_T, z_T, w_T)]
        # we tranform it a Tensor of tensor Tensor([Tensor(u_0, ..., u_T), Tensor(z_0, ..., z_T), Tensor(w_0, ..., w_T)])
        # of shape (S, B, T, N) S: number of states, B: number of batch, T: number of time-steps, N: number of neurons
        
        s1_list = torch.stack([torch.stack(x, dim=-2) for x in zip(*s1_list)], dim=0)
        self.states.append(s1_list)
        if self.two_layers:
            s2_list = torch.stack([torch.stack(x, dim=-2) for x in zip(*s2_list)], dim=0)
        s_out_list = torch.stack([torch.stack(x, dim=-2) for x in zip(*s_out_list)], dim=0)
        
        self.states.append(s_out_list)
        return torch.stack(out_sequence, dim=1)

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        self.l1.apply_parameter_constraints()
        if self.two_layers:
            self.l2.apply_parameter_constraints()
        self.out_layer.apply_parameter_constraints()

    def process_predictions_and_compute_losses(self, outputs, targets, block_idx):
        """
        Process the model output into prediction
        with respect to the temporal segmentation defined by the
        block_idx tensor.
        Then compute losses
        Args:
            outputs (torch.Tensor): full outputs
            targets (torch.Tensor): targets
            block_idx (torch.Tensor): tensor of index that determined which temporal segements of
            output time-step depends on which specific target,
            used by the scatter reduce operation.

        Returns:
            (): _description_
        """
        # compute softmax for every time-steps with respect to
        # the number of class
        if self.auto_regression:
            targets = targets[:, 1:]
            l2_loss = (outputs - targets) ** 2
            
            block_outputs = torch.zeros(
                size=(targets.shape[0], 2, outputs.shape[2]),
                dtype=outputs.dtype,
                device=outputs.device,
            )
            _block_idx = block_idx.unsqueeze(2).expand(size=(-1, -1, outputs.size(2)))
            block_output = torch.scatter_reduce(
                block_outputs,
                dim=1,
                index=_block_idx,
                src=l2_loss,
                reduce="mean",
                include_self=False,
            )
            block_idx = block_idx.unsqueeze(-1)
            block_output = block_output[:, 1]
            outputs_reduce = outputs
            loss = block_output.mean()
        else:
            if self.output_func == "softmax":
                outputs = torch.softmax(outputs, -1)
                reduction = "sum"
            else:
                reduction = "mean"
            # create a zero array of size (batch, number_of_targets, number_of_classes)
            # this will be used to defined the prediction for each targets for each classes
            block_outputs = torch.zeros(
                size=(targets.size(0), targets.size(1), outputs.size(2)),
                dtype=outputs.dtype,
                device=outputs.device,
            )
            block_idx = block_idx.unsqueeze(-1)

            block_output = torch.scatter_reduce(
                block_outputs,
                dim=1,
                index=block_idx.broadcast_to(outputs.shape),
                src=outputs,
                reduce=reduction,
                include_self=False,
            )


            outputs_reduce = block_output.reshape(-1, outputs.size(-1))
            targets_reduce = targets.flatten()

            block_mask = torch.where(targets_reduce != self.ignore_target_idx)

            loss = self.loss(outputs_reduce[block_mask].float(), targets_reduce[block_mask].long())
        return (outputs_reduce, loss, block_idx)

    def update_and_log_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        loss: float,
        metrics: torchmetrics.MetricCollection,
        prefix: str,
    ):
        """
        Method centralizing the metrics logging mecanisms.

        Args:
            outputs_reduce (torch.Tensor): output prediction
            targets_reduce (torch.Tensor): target
            loss (float): loss
            metrics (torchmetrics.MetricCollection): collection of torchmetrics metrics
            aux_metrics (dict): auxiliary metrics that do not
            fit the torchmetrics logic
            prefix (str): prefix defining the stage of model either
            "train_": training stage
            "val_": validation stage
            "test_": testing stage
            Those prefix prevent clash of names in the logger.

        """
        if self.auto_regression:
            single_step_prediction_limit = int(math.ceil(0.5*outputs.shape[1]))
            outputs = outputs[:, single_step_prediction_limit:].squeeze()
            targets = targets[:, single_step_prediction_limit+1:].squeeze()
            outputs = outputs.reshape(-1, outputs.shape[-1])
            targets = targets.reshape(-1, targets.shape[-1])
        else:
            targets = targets.flatten()

        metrics(outputs, targets)
        self.log_dict(
            metrics,
            prog_bar=True,
            on_epoch=True,
            on_step=True if prefix == "train_" else False,
        )
        self.log(
            f"{prefix}loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True if prefix == "train_" else False,
        )

    def training_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(
            inputs,
        )
        (
            outputs_reduce,
            loss,
            block_idx,
        ) = self.process_predictions_and_compute_losses(outputs, targets, block_idx)

        self.update_and_log_metrics(
            outputs_reduce,
            targets,
            loss,
            self.train_metric,
            prefix="train_",
        )

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(inputs)
        (
            outputs_reduce,
            loss,
            block_idx,
        ) = self.process_predictions_and_compute_losses(outputs, targets, block_idx)

        self.update_and_log_metrics(
            outputs_reduce,
            targets,
            loss,
            self.val_metric,
            prefix="val_",
        )
        # report statistics (weights and spiking distribution) and plot an example of
        # model behavior against a random input
        if batch_idx == 0:
            # determine a random example to visualized
            spike_probabilities = get_per_layer_spike_probs(
                self.states,
                block_idx,
            )
            rnd_batch_idx = torch.randint(0, inputs.shape[0], size=()).item()
            prev_layer_input = inputs[rnd_batch_idx]
            layers = [self.l1,]
            if self.two_layers:
                layers.append(self.l2)
            layers.append(self.out_layer)
            for layer, module in enumerate(layers):
                if hasattr(module, "layer_stats"):
                    module.layer_stats(
                        logger=self.logger,
                        epoch_step=self.current_epoch,
                        inputs=prev_layer_input,
                        states=self.states[layer][:, rnd_batch_idx],
                        targets=targets[rnd_batch_idx],
                        layer_idx=layer,
                        block_idx=block_idx[rnd_batch_idx],
                        spike_probabilities=spike_probabilities[layer]
                        if len(spike_probabilities) > layer
                        else None,
                        output_size=self.output_size,
                        auto_regression=self.auto_regression
                    )
                    if layer < len(layers) - 1:
                        prev_layer_input = self.states[layer][1, rnd_batch_idx]

        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(inputs)

        (
            outputs_reduce,
            loss,
            block_idx,
        ) = self.process_predictions_and_compute_losses(outputs, targets, block_idx)

        self.update_and_log_metrics(
            outputs_reduce,
            targets,
            loss,
            self.test_metric,
            prefix="test_",
        )

        return loss

    def init_metrics_and_loss(self):
        if self.auto_regression:
            metrics = torchmetrics.MetricCollection(
                {
                    "mse": torchmetrics.MeanSquaredError(),
                }
            )
            self.loss = MSELoss()
        else:
            metrics = torchmetrics.MetricCollection(
                {
                    "acc": torchmetrics.Accuracy(
                        task="multiclass",  # type: ignore
                        num_classes=self.output_size,
                        average="micro",
                        ignore_index=self.ignore_target_idx,
                    )
                }
            )
            self.loss = CrossEntropyLoss(ignore_index=self.ignore_target_idx)
        self.train_metric = metrics.clone(prefix="train_")
        self.val_metric = metrics.clone(prefix="val_")
        self.test_metric = metrics.clone(prefix="test_")

    def on_before_optimizer_step(self, optimizer) -> None:
        # log weights gradient norm
        self.log_dict(grad_norm(self, norm_type=2))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=self.tracking_mode,
            factor=self.factor,
            patience=self.patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": self.tracking_metric,
            },
        }
