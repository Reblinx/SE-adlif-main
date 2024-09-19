
from typing import Optional
from models.helpers import SLAYER, get_event_indices, save_distributions_to_aim, save_fig_to_aim
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor
from torch.nn.parameter import Parameter
import os

from module.tau_trainers import TauTrainer, get_tau_trainer_class
from omegaconf import DictConfig
import matplotlib.pyplot as plt

class LIF(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        cfg: DictConfig,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(**kwargs)
        self.in_features = cfg.input_size
        self.out_features = cfg.n_neurons
        self.dt = 1.0
        self.tau_u_range = cfg.tau_u_range
        self.train_tau_u_method = cfg.get('train_tau_u_method', 'interpolation')
        
        self.use_recurrent = cfg.get('use_recurrent', True)
        self.thr = cfg.get('thr', 1.0)
        
        self.alpha = cfg.get('alpha', 5.0)
        self.c = cfg.get('c', 0.4)

        self.weight = Parameter(
            torch.empty((self.out_features, self.in_features), **factory_kwargs)
        )
        self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        if self.use_recurrent:
            self.recurrent = Parameter(
                    torch.empty((self.out_features, self.out_features), **factory_kwargs)
                )
        else:
            self.register_buffer("recurrent", None)
        self.tau_u_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_u_method)(
            self.out_features,
            self.dt,
            self.tau_u_range[0],
            self.tau_u_range[1],
            **factory_kwargs,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.tau_u_trainer.reset_parameters()
        torch.nn.init.uniform_(
            self.weight,
            -1.0 * torch.sqrt(1 / torch.tensor(self.in_features)),
            torch.sqrt(1 / torch.tensor(self.in_features)),
        )
        torch.nn.init.zeros_(self.bias)
        if self.use_recurrent:
            torch.nn.init.orthogonal_(
                self.recurrent,
                gain=1.0,
            )

    def initial_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tensor:
        size = (batch_size, self.out_features)
        u = torch.zeros(
            size=size, device=device, dtype=torch.float, requires_grad=True
        )
        z = torch.zeros(size=size, device=device, dtype=torch.float, requires_grad=True)
        return u, z

    def forward(self, input_tensor: Tensor, states: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        u_tm1, z_tm1 = states
        decay_u = self.tau_u_trainer.get_decay()
        soma_current = F.linear(input_tensor, self.weight, self.bias)
        if self.use_recurrent:
            soma_rec_current = F.linear(z_tm1, self.recurrent, None)
            soma_current += soma_rec_current

        u_t = decay_u * u_tm1 + (1.0 - decay_u) * (soma_current)
        u_thr = u_t - self.thr
        # Forward Gradient Injection trick (credits to Sebastian Otte)
        z_t = torch.heaviside(u_thr, torch.as_tensor(0.0).type(u_thr.dtype)).detach() + (u_thr - u_thr.detach()) * SLAYER(u_thr, self.alpha, self.c).detach()
        u_t = u_t * (1 - z_t.detach())
        return z_t, (u_t, z_t)


    def apply_parameter_constraints(self):
        self.tau_u_trainer.apply_parameter_constraints()
    
    @staticmethod
    def plot_states(layer_idx, inputs, states):
        figure, axes = plt.subplots(nrows=3, ncols=1, sharex="all", figsize=(8, 11))
        is_events = torch.all(inputs == inputs.round())
        inputs = inputs.cpu().detach().numpy()
        states = states.cpu().detach().numpy()
        
        if is_events:
            axes[0].eventplot(get_event_indices(inputs.T), color='black', orientation='horizontal')
        else:
            axes[0].plot(inputs)
        axes[0].set_ylabel("input")
        axes[1].plot(states[0])
        axes[1].set_ylabel("v_t")
        axes[2].eventplot(
            get_event_indices(states[1].T), color="black", orientation="horizontal"
        )
        axes[2].set_ylabel("z_t/output")
        nb_spikes_str = str(states[1].sum())
        figure.suptitle(f"Layer {layer_idx}\n Nb spikes: {nb_spikes_str},")
        plt.close(figure)
        return figure

    @torch.jit.ignore
    def layer_stats(
        self,
        layer_idx: int,
        logger,
        epoch_step: int,
        spike_probabilities: torch.Tensor,
        inputs: torch.Tensor,
        states: torch.Tensor,
        **kwargs,
    ):
        """Generate statistisc from the layer weights and a plot of the layer dynamics for a random task example
        Args:
            layer_idx (int): index for the layer in the hierarchy
            logger (_type_): aim logger reference
            epoch_step (int): epoch
            spike_probability (torch.Tensor): spike probability for each neurons
            inputs (torch.Tensor): random example
            states (torch.Tensor): states associated to the computation of the random example
        """

        save_fig_to_aim(
            logger=logger,
            name=f"{layer_idx}_Activity",
            figure=LIF.plot_states(layer_idx, inputs, states),
            epoch_step=epoch_step,
        )

        distributions = [
            ("soma_tau", self.tau_u_trainer.get_tau().cpu().detach().numpy()),
            ("soma_weights", self.weight.cpu().detach().numpy()),
            ("spike_prob", spike_probabilities.cpu().detach().numpy()),
            ("bias", self.bias.cpu().detach().numpy()),
        ]

        if self.use_recurrent:
            distributions.append(
                ("recurrent_weights", self.recurrent.cpu().detach().numpy())
            )
        save_distributions_to_aim(
            logger=logger,
            distributions=distributions,
            name=f"{layer_idx}",
            epoch_step=epoch_step,
        )