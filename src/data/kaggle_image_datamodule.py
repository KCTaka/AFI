import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from pytorch_lightning import LightningDataModule

import kagglehub

from typing import Optional, Tuple, List


class KaggleImageDataModule(LightningDataModule):
    """
    A custom PyTorch Lightning DataModule for loading and processing images from a Kaggle dataset.
    """
    def __init__(self, 
                 kaggle_dataset_path,
                 channels=3,
                 image_size=(64, 64),
                 train_val_test_split=(55_000, 5_000, 10_000),
                 batch_size=64,
                 num_workers=4,
                 pin_memory=True):
        super().__init__()
        self.save_hyperparameters()

        self.transforms = transforms.Compose([
            transforms.Resize(self.hparams.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # If channels=1, consider adding transforms.Grayscale(num_output_channels=1) before ToTensor
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
            try:
                full_dataset = datasets.ImageFolder(root=self.dataset_path, transform=self.transforms)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Dataset not found at path: {self.dataset_path}. "
                    "Ensure prepare_data() has run and the path is correct, "
                    "and that it's structured for torchvision.datasets.ImageFolder."
                )
            
            total_data = len(full_dataset)
            if sum(self.hparams.train_val_test_split) > total_data:
                # Adjust split if requested size is larger than available data
                print(f"Warning: Requested split {self.hparams.train_val_test_split} (total {sum(self.hparams.train_val_test_split)}) "
                      f"is larger than dataset size {total_data}. Using all available data for splits proportionally.")
                
                train_len, val_len, test_len = self.hparams.train_val_test_split
                total_requested = sum(self.hparams.train_val_test_split)

                # Proportional split based on available data
                train_len = int((train_len / total_requested) * total_data)
                val_len = int((val_len / total_requested) * total_data)
                # Assign remaining to test to ensure sum matches total_data
                test_len = total_data - train_len - val_len 
                
                current_splits = (train_len, val_len, test_len)
            else:
                current_splits = self.hparams.train_val_test_split
                # If there's leftover data not covered by the split, it will be ignored by random_split
                # Ensure sum of splits does not exceed total_data if you want to use all data
                if sum(current_splits) < total_data:
                    print(f"Warning: Sum of splits {sum(current_splits)} is less than total dataset size {total_data}. "
                          f"{total_data - sum(current_splits)} samples will be unused.")
                elif sum(current_splits) > total_data: # Should be caught by above, but as safeguard
                     raise ValueError(f"Sum of splits {sum(current_splits)} exceeds dataset size {total_data} after potential adjustment.")


            self.data_train, self.data_val, self.data_test = random_split(
                dataset=full_dataset,
                lengths=list(current_splits), # random_split expects a list or tuple of ints
                generator=torch.Generator().manual_seed(42) # for reproducibility
            )
    
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