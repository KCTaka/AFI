import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from pytorch_lightning import LightningDataModule

from typing import Optional
    
class RandomImageDataset(Dataset):
    """
    A custom PyTorch Dataset that generates random image tensors. Mainly for testing purposes.
    """
    def __init__(self, num_samples=100, channels=3, image_size=(64, 64), num_classes=10):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of image channels (e.g., 3 for RGB).
            height (int): Height of the images.
            width (int): Width of the images.
            num_classes (int): Number of classes for dummy labels.
        """
        super().__init__()
        self.num_samples = num_samples
        self.channels = channels
        self.image_size = image_size    
        self.height, self.width = image_size  
        self.num_classes = num_classes

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generates a single random image and a random label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (random_image, random_label)
                   - random_image (torch.Tensor): A tensor of shape (channels, height, width)
                                                  with random values between 0 and 1.
                   - random_label (torch.Tensor): A random integer label.
        """
        # Generate a random image (values between 0 and 1)
        random_image = torch.rand(self.channels, self.height, self.width)
        # Generate a random label
        random_label = torch.randint(0, self.num_classes, (1,)).squeeze() # Use squeeze to make it a 0-dim tensor
        return random_image, random_label
    
    
class RandomImageDataModule(LightningDataModule):
    def __init__(self, 
                 channels=3,
                 image_size=(64, 64),
                 num_classes=10,
                 train_val_test_split=(55_000, 5_000, 10_000),
                 batch_size=64,
                 num_workers=4,
                 pin_memory=False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['num_classes'])
        
        self.num_classes = num_classes
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.batch_size_per_device = batch_size
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = RandomImageDataset(
                num_samples=sum(self.hparams.train_val_test_split),
                channels=self.hparams.channels,
                image_size=self.hparams.image_size,
                num_classes=self.num_classes,
            )
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
            )
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
        
    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        
    
        


if __name__ == "__main__":
    _ = RandomImageDataModule(
        channels=3,
        image_size=(64, 64),
        num_classes=10,
        train_val_test_split=(55_000, 5_000, 10_000),
        batch_size=64,
        num_workers=4,
        pin_memory=False,
    )
