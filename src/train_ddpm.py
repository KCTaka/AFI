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

from src.utils.hydra import instantiate_list
from src.systems.autoencoder import AutoEncoder

import wandb


def get_relative_model_ckpt_path(absolute_artifact_dir_path: str, model_filename: str = "model.ckpt") -> str:
    """
    Converts an absolute artifact directory path to a path relative to the
    project root, and appends a model filename.
    The output will not start with "./" for subdirectories but will retain "../"
    if the path is outside the current project root.

    Relies on os.getcwd() returning the project root (e.g., /workspace/AFI),
    typically ensured by autorootcwd.

    Args:
        absolute_artifact_dir_path: The full path to the artifact directory,
                                     e.g., "/workspace/AFI/artifacts/model-cyfz4n65:v59"
        model_filename: The filename to append, defaults to "model.ckpt".

    Returns:
        A relative path to the model file.
        Examples:
        - "artifacts/model-cyfz4n65:v59/model.ckpt"
        - "model.ckpt" (if absolute_artifact_dir_path is the project root)
        - "../another_project/model_dir:v1/model.ckpt" (if outside project root)
    """
    project_root = os.getcwd()  # e.g., /workspace/AFI thanks to autorootcwd

    # Get the relative path for the input directory
    # os.path.relpath is robust and handles various cases correctly.
    relative_dir_path = os.path.relpath(absolute_artifact_dir_path, project_root)

    # If the artifact directory is the project root itself,
    # os.path.relpath returns "."
    if relative_dir_path == ".":
        # In this case, the model file is directly in the project root.
        return model_filename
    else:
        # For all other cases (subdirectories, or paths like "../foo"),
        # join the relative directory path with the model filename.
        # os.path.join correctly handles path separators for the OS.
        return os.path.join(relative_dir_path, model_filename)

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

@hydra.main(version_base=None, config_path="../configs", config_name="train_ddpm")
def train(cfg: DictConfig):
    
    run = wandb.init(
        project=cfg.loggers.wandb.project,
    )
    
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    
    model_ae = load_pretrained_ae(
        run=run,
        artifact_path="Anime-Frame-Interpolation/Anime Auto Encoder/model-cyfz4n65:v59",
        models_cfg=cfg.models
    )
    
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