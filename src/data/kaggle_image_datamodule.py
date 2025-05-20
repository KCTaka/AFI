import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from lightning.pytorch import LightningDataModule

import kagglehub

from typing import Optional, Tuple, List

def scaler(x: torch.Tensor) -> torch.Tensor:
    """
    Assume the input tensor is in the range [0, 1].
    """
    return x * 2 - 1

class KaggleImageDataModule(LightningDataModule):
    """
    A custom PyTorch Lightning DataModule for loading and processing images from a Kaggle dataset.
    """
    def __init__(self, 
                 kaggle_dataset_path,
                 channels=3,
                 image_size=(64, 64),
                 train_val_test_split={"train": 0.8, "val": 0.1, "test": 0.1},
                 batch_size=64,
                 num_workers=4,
                 pin_memory=True):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_path: Optional[str] = None  # Initialize dataset_path
        
        # For badasstechie/celebahq-resized-256x256
        # mean: (tensor([ 0.0352, -0.1657, -0.2721]) + 1)*.5, 
        # std: tensor([0.5969, 0.5393, 0.5294])*.5

        self.transforms = transforms.Compose([
            transforms.Resize(self.hparams.image_size),
            transforms.ToTensor(),
            transforms.Lambda(scaler),
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.batch_size_per_device = self.hparams.batch_size
        
    def prepare_data(self):
        self.dataset_path = kagglehub.dataset_download(self.hparams.kaggle_dataset_path)
        # self.dataset_path should point to the directory containing subdirectories for classes,
        # or the direct image files if ImageFolder is not used directly in setup.
        # For ImageFolder, it expects a structure like:
        # dataset_path/class_a/image1.jpg
        # dataset_path/class_b/image2.jpg
        # If your Kaggle dataset has a different structure, self.dataset_path might need adjustment
        # or a custom Dataset class would be more appropriate.
        
    def setup(self, stage: Optional[str] = None):
        # Adjust batch size for DDP
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                # Consider logging a warning or adjusting batch size more gracefully
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # Assuming self.dataset_path is the root directory for ImageFolder
            # If your dataset is not structured for ImageFolder, you'll need a custom Dataset
            if self.dataset_path is None:
                self.dataset_path = kagglehub.dataset_download(self.hparams.kaggle_dataset_path)
                
            try:
                self.dataset = datasets.ImageFolder(root=self.dataset_path, transform=self.transforms)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Dataset not found at path: {self.dataset_path}. "
                    "Ensure prepare_data() has run and the path is correct, "
                    "and that it's structured for torchvision.datasets.ImageFolder."
                )
            dataset_split = self.hparams.train_val_test_split
            current_splits = (dataset_split["train"], dataset_split["val"], dataset_split["test"])
            self.data_train, self.data_val, self.data_test = random_split(dataset=self.dataset, lengths=current_splits)
    
    def train_dataloader(self):
        if not self.data_train:
            raise ValueError("Train data not available. Call setup() first.")
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
    
    def val_dataloader(self):
        if not self.data_val:
            raise ValueError("Validation data not available. Call setup() first.")
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def test_dataloader(self):
        if not self.data_test:
            raise ValueError("Test data not available. Call setup() first.")
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        
if __name__ == "__main__":
    # Example usage
    kaggle_dataset_path = "badasstechie/celebahq-resized-256x256"  # Replace with your actual dataset path
    datamodule = KaggleImageDataModule(
        kaggle_dataset_path=kaggle_dataset_path,
        channels=3,
        image_size=(128, 128),
        train_val_test_split={"train": 0.8, "val": 0.1, "test": 0.1},  # 80% train, 10% val, 10% test
        batch_size=10,
        num_workers=1,
        pin_memory=False,
    )
    datamodule.prepare_data()
    datamodule.setup()
    
    test_dataloader = datamodule.test_dataloader()
    batch = next(iter(test_dataloader))[0]
    print(f"Batch size: {batch.size()}")
    print(f"Batch min: {batch.min()}")
    print(f"Batch max: {batch.max()}")

