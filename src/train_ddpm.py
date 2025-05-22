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

from src.utils.hydra import instantiate_list
from src.systems.autoencoder import AutoEncoder

@hydra.main(version_base=None, config_path="../configs", config_name="train_ddpm")
def train(cfg: DictConfig):
    
    
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    
    ## Load the autoencoder weights
    model_ae = hydra.utils.instantiate(cfg.models.autoencoder)
    model_d = hydra.utils.instantiate(cfg.models.discriminator)
    lpips = hydra.utils.instantiate(cfg.models.perceptual)
    system_ae = AutoEncoder.load_from_checkpoint(cfg.autoencoder_weights, strict=False, model_ae=model_ae, model_d=model_d, lpips=lpips)
    model_ae = system_ae.model_ae
    
    system_ddpm: LightningModule = hydra.utils.instantiate(cfg.ddpm, model_ae=model_ae)

    logger: list[Logger] = instantiate_list(cfg.loggers.values(), cls=Logger)
    callbacks: list[Callback] = instantiate_list(cfg.callbacks.values(), cls=Callback)

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        logger=logger,
        callbacks=callbacks,
    )

    
    trainer.fit(
        model=system_ddpm,
        datamodule=datamodule,
    )

if __name__ == "__main__":
    train()