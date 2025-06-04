import autoroot
import autorootcwd

import torch
torch.set_float32_matmul_precision('medium')

import hydra
import lightning as L
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies import Strategy
from omegaconf import DictConfig, OmegaConf
import wandb

from src.utils.hydra import instantiate_list, flat_dict_to_nested_dict

def train(cfg: DictConfig, is_sweep: bool = False):
    
    if is_sweep:
        run = wandb.init(
            project=cfg.loggers.wandb.project,
            job_type="sweep"
        )
        
        sweep_cfg = flat_dict_to_nested_dict(wandb.config)
        cfg = OmegaConf.merge(cfg, sweep_cfg)
        
    # Initialize the data module
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # Initialize the model
    system: LightningModule = hydra.utils.instantiate(cfg.systems)
    
    # Initialize the logger
    logger: list[Logger] = instantiate_list(cfg.loggers.values(), cls=Logger)

    # Initialize the callbacks
    callbacks: list[Callback] = instantiate_list(cfg.callbacks.values(), cls=Callback)

    # Set the logger and callbacks in the trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        logger=logger,
        callbacks=callbacks,
    )
    
    trainer.fit(
        model=system,
        datamodule=datamodule,
    )

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    
    if cfg.sweep is None:
        train(cfg, is_sweep=False)
        return
    
    train_ae = lambda: train(cfg=cfg, is_sweep=True)
    sweep_cfg = OmegaConf.to_container(cfg.sweep, resolve=True)
    sweep_id = wandb.sweep(
        sweep_cfg,
        project=cfg.loggers.wandb.project,
    )
    wandb.agent(
        sweep_id,
        function=train_ae,
        count=cfg.sweep.agent.count,
        project=cfg.loggers.wandb.project,
    )
        
    
if __name__ == "__main__":
    main()