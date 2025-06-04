import autoroot
import autorootcwd

import torch
torch.set_float32_matmul_precision('medium')

import hydra
import lightning as pl
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import wandb

from src.utils.hydra import instantiate_list, flat_dict_to_nested_dict

@rank_zero_only
def setup_wandb(cfg: DictConfig):
    wandb.init(
        project=cfg.loggers.wandb.project,
        job_type="sweep",
    )
    sweep_cfg = flat_dict_to_nested_dict(wandb.config)
    comb_cfg = OmegaConf.merge(cfg, sweep_cfg)
    return comb_cfg

def train(cfg: DictConfig, is_sweep: bool = False):
    
    logger: list[Logger] = instantiate_list(cfg.loggers.values(), cls=Logger)
    callbacks: list[Callback] = instantiate_list(cfg.callbacks.values(), cls=Callback)
    
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        logger=logger,
        callbacks=callbacks,
    )
    
    if is_sweep:
        comb_cfg = setup_wandb(cfg)
        cfg = trainer.strategy.broadcast(comb_cfg, src=0)
        
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    system: LightningModule = hydra.utils.instantiate(cfg.systems)
    
    trainer.fit(
        model=system,
        datamodule=datamodule,
    )

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    
    if cfg.sweep is None:
        train(cfg, is_sweep=False)
        return
    
    @rank_zero_only
    def run_agent():
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
        
        return True
    
    is_rank_zero = run_agent()
    if is_rank_zero:
        return
    
    train(cfg, is_sweep=True)
    
        
    
if __name__ == "__main__":
    main()