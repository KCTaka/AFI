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

from src.utils.hydra import instantiate_list
from src.systems.autoencoder import AutoEncoder

def train(cfg: DictConfig, model_ae: AutoEncoder = None):
    if model_ae is None:
        raise ValueError("model_ae must be provided as an argument to train function.")
    
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    
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

@hydra.main(version_base=None, config_path="../configs", config_name="hparam_tune")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    
    ## Load the autoencoder weights
    model_ae = hydra.utils.instantiate(cfg.models.autoencoder)
    model_d = hydra.utils.instantiate(cfg.models.discriminator)
    lpips = hydra.utils.instantiate(cfg.models.perceptual)
    system_ae = AutoEncoder.load_from_checkpoint(cfg.autoencoder_weights, strict=False, model_ae=model_ae, model_d=model_d, lpips=lpips)
    model_ae = system_ae.model_ae.eval()
    
    train_ddpm = lambda cfg: train(cfg, model_ae=model_ae)
    
    sweep_id = wandb.sweep(
        cfg,
        project=cfg.loggers.wandb.project,
    )
    wandb.agent(
        sweep_id,
        function=train_ddpm,
        count=50,
        project=cfg.loggers.wandb.project,
    )


if __name__ == "__main__":
    main()