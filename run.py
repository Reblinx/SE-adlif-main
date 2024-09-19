import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import logging
from models.pl_module import MLPSNN

from pytorch_lightning.strategies import SingleDeviceStrategy
import os
import matplotlib.pyplot as plt
import matplotlib
colors = matplotlib.colormaps.get_cmap('tab20').colors + matplotlib.colormaps.get_cmap('Set1').colors
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

# Main entry point. We use Hydra (https://hydra.cc) for configuration management. Note, that Hydra changes the working directory, such that each run gets a unique directory.

@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: DictConfig):        
    logging.getLogger().addHandler(logging.FileHandler("out.log"))
    logging.info(f"Experiment name: {cfg.exp_name}")
    pl.seed_everything(cfg.random_seed, workers=True)
    datamodule = hydra.utils.instantiate(cfg.dataset)
    
    model = MLPSNN(cfg)
    callbacks = []
    model_ckpt_tracker: ModelCheckpoint = ModelCheckpoint(
        monitor=cfg.get('tracking_metric', "val_acc_epoch"),
        mode=cfg.get('tracking_mode', 'max'),
        save_last=False,
        save_top_k=1,
        dirpath="ckpt"
    )
    lr_monitor = LearningRateMonitor(
        logging_interval='step'
    )
    callbacks = [model_ckpt_tracker, lr_monitor]
    try:
        from aim.pytorch_lightning import AimLogger
        # a repo is were the database of all experiments results
        # could be found.
        # hydra modify working directory, so we want to resolve cfg.log_dir (which could be a relative path)
        # with respect to the orginal working directory (se-adlif folder)
        cwd = hydra.utils.get_original_cwd()
        log_dir = cfg.logdir
        if not os.path.isabs(log_dir):
            log_dir = os.path.abspath(os.path.join(cwd, log_dir))
        logging.info(f"Aim logging, please run aim up --repo={log_dir} to visualize experiments")
        logger = AimLogger(
            repo=log_dir,
            experiment=cfg.exp_name,
            train_metric_prefix='train_',
            val_metric_prefix='val_',
        )
    except ImportError:
        logging.warning(f"Aim import failed logging will fallback  to csv\n CSV file at {os.path.abspath(os.path.join(os.getcwd(),'logs/mlp/snn'))} location")
        logger = pl.loggers.CSVLogger("logs", name="mlp_snn")
    trainer: pl.Trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        max_epochs=cfg.n_epochs,
        gradient_clip_val=1.5,
        enable_progress_bar=True,
        strategy=SingleDeviceStrategy(device=cfg.device),
        )
    trainer.fit(model, datamodule=datamodule)
    result = trainer.test(model, ckpt_path="best", datamodule=datamodule)
    logging.info(f"Final result: {result}")

    return trainer.checkpoint_callback.best_model_score.cpu().detach().numpy()


if __name__ == "__main__":
    main()
