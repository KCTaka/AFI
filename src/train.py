import autoroot
import autorootcwd

import torch
torch.set_float32_matmul_precision('medium')

# wandb sweep --project "AE test 256x256" configs/sweep/wandb_ae.yaml
# wandb agent 'Anime-Frame-Interpolation/AE test 256x256/xgan3zyz'

import hydra
import lightning as pl
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
import wandb

from src.utils.hydra import instantiate_list, flat_dict_to_nested_dict

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig):
            
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    system: LightningModule = hydra.utils.instantiate(cfg.systems)
    
    logger: list[Logger] = instantiate_list(cfg.loggers.values(), cls=Logger)
    callbacks: list[Callback] = instantiate_list(cfg.callbacks.values(), cls=Callback)
    
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(
        model=system,
        datamodule=datamodule,
    )
    
if __name__ == "__main__":
    train()