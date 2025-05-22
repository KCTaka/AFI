import autoroot
import autorootcwd

import hydra
import lightning as L
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies import Strategy
from omegaconf import DictConfig

from src.utils.hydra import instantiate_list

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig):
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
    
if __name__ == "__main__":
    train()