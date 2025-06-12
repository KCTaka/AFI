import autoroot
import autorootcwd
import os

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
from src.utils.formats import get_relative_model_ckpt_path
from src.systems.autoencoder import AutoEncoder
from src.systems.ddpm import DDPM

L.seed_everything(0, workers=True)

def load_pretrained_ae(run, artifact_path: str, models_cfg: DictConfig):
    model_ae = hydra.utils.instantiate(models_cfg.autoencoder)
    model_d = hydra.utils.instantiate(models_cfg.discriminator)
    lpips = hydra.utils.instantiate(models_cfg.perceptual)
    
    artifact = run.use_artifact(artifact_path, type="model")
    artifact_dir = artifact.download()
    print(f"Artifact downloaded to: {artifact_dir}")
    artifact_dir = get_relative_model_ckpt_path(artifact_dir)
    print(f"Relative path to artifact: {artifact_dir}")
    system_ae = AutoEncoder.load_from_checkpoint(artifact_dir, strict=False, model_ae=model_ae, model_d=model_d, lpips=lpips)
    model_ae = system_ae.model_ae.eval()
    
    return model_ae

def load_checkpoint(run, artifact_path: str, cfg: DictConfig):
    model_ae = hydra.utils.instantiate(cfg.models.autoencoder)
    model_diffusion = hydra.utils.instantiate(cfg.models.diffusion)
    noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)
    
    artifact = run.use_artifact(artifact_path, type="model")
    artifact_dir = artifact.download()
    print(f"Artifact downloaded to: {artifact_dir}")
    artifact_dir = get_relative_model_ckpt_path(artifact_dir)
    print(f"Relative path to artifact: {artifact_dir}")
    system_ddpm = DDPM.load_from_checkpoint(artifact_dir, strict=False, model_ae=model_ae, model_diffusion=model_diffusion, noise_scheduler=noise_scheduler)
    
    return system_ddpm

@hydra.main(version_base=None, config_path="../configs", config_name="train_ddpm")
def train(cfg: DictConfig):
    run = wandb.init(
        project=cfg.loggers.wandb.project,
    )
    
    if cfg.checkpoint is not None:
        system_ddpm: LightningModule = load_checkpoint(
            run=run,
            artifact_path=cfg.checkpoint,
            cfg=cfg
        )
        
    else:
        if cfg.ae_checkpoint is None:
            raise ValueError("No autoencoder checkpoint provided. Please provide a valid checkpoint path in the configuration. You need a trained autoencoder to train the DDPM model.")
        model_ae = load_pretrained_ae(
            run=run,
            artifact_path=cfg.ae_checkpoint,
            models_cfg=cfg.models
        )
        
        system_ddpm: LightningModule = hydra.utils.instantiate(cfg.ddpm, model_ae=model_ae)


    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    
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

# This method for some reason only works with one GPU. Tried other ways, but its best to run wandb sweep from the command line. 
# def main(cfg: DictConfig):
    
#     if cfg.sweep is None:
#         train(main_cfg=cfg)
#         return
    
#     train_ddpm = lambda: train(main_cfg=cfg)

#     sweep_cfg = OmegaConf.to_container(cfg.sweep, resolve=True)

#     sweep_id = wandb.sweep(
#         sweep_cfg,
#         project=cfg.loggers.wandb.project,
#     )
    
#     wandb.agent(
#         sweep_id,
#         function=train_ddpm,
#         count=cfg.sweep_agent_count,
#         project=cfg.loggers.wandb.project,
#     )


if __name__ == "__main__":
    train()