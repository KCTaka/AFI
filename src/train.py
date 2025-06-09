import autoroot
import autorootcwd

import torch
torch.set_float32_matmul_precision('medium')

import hydra
import lightning as pl
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
import wandb

from src.systems.autoencoder import AutoEncoder

from src.utils.hydra import instantiate_list, flat_dict_to_nested_dict
from src.utils.formats import get_relative_model_ckpt_path


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
    
    return system_ae

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig):
    
    if cfg.checkpoint is not None:
        run = wandb.init(
            project=cfg.loggers.wandb.project,
        )
        system: LightningModule = load_pretrained_ae(
            run=run,
            artifact_path=cfg.checkpoint,
            models_cfg=cfg.models,
        )
    else:
        system: LightningModule = hydra.utils.instantiate(cfg.systems)        
    
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    
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