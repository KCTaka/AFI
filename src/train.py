import autoroot
import autorootcwd

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig):
    # Initialize the data module
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # Initialize the model
    system: LightningModule = hydra.utils.instantiate(cfg.systems)
    
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)
    
    trainer.fit(
        model=system,
        datamodule=datamodule,
    )
    
if __name__ == "__main__":
    train()