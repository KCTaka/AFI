import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

import autoroot
import autorootcwd

def test_ae_system():
    from src.systems.autoencoder import AutoEncoder
    from src.models.vae import VAE
    from src.models.discriminator import Discriminator
    from src.models.lpips import LPIPS
    from src.data.random_datamodule import RandomImageDataModule
    
    model_ae = VAE(
        latent_dim=128,
        beta = 1,
    )
    
    model_d = Discriminator(
        in_channels=3,
    )
    
    model_lpips = LPIPS()
    
    model = AutoEncoder(
        model_ae=model_ae,
        model_d=model_d,
        lpips=model_lpips,
        loss_weights={"reconst": 1.0, "internal": 0.5, "perceptual": 0.5, "adversarial": 0.5},
        lr_g=1e-4,
        lr_d=1e-4,
        betas_g=(0.9, 0.999),
        betas_d=(0.9, 0.999),
    )
    
    NUM_SAMPLES = 2   # Number of dummy samples to generate
    BATCH_SIZE = 5     # Batch size for the DataLoader
    IMG_CHANNELS = 3   # Number of image channels
    IMG_HEIGHT = 128    # Image height
    IMG_WIDTH = 128     # Image width
    NUM_CLASSES = 5    # Number of dummy classes

    # Create an instance of the random dataset
    
        # 4. Use Trainer with fast_dev_run
    # The `fast_dev_run=True` flag will run a single batch of training and validation
    # to quickly check for errors in your code.
    # It's very useful for debugging.
    # For a slightly more comprehensive check (e.g., a few batches or an epoch):
    # trainer = pl.Trainer(max_epochs=1, limit_train_batches=5, limit_val_batches=0, accelerator="auto")
    # `accelerator="auto"` will try to use GPU if available, otherwise CPU.
    # `devices=1` specifies using one device (CPU or GPU).

    print("Starting PyTorch Lightning fast_dev_run...")
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="auto", # Automatically selects GPU if available, else CPU
        devices=1           # Use 1 device (CPU or GPU)
    )
    
    

    train_dataloader = RandomImageDataModule(
        channels=IMG_CHANNELS,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        num_classes=NUM_CLASSES,
        train_val_test_split=(NUM_SAMPLES, NUM_SAMPLES, NUM_SAMPLES),
        batch_size=BATCH_SIZE,
        num_workers=0,  # Set to 0 for simplicity in this test
        pin_memory=False,  # Set to False for simplicity in this test
    )
    

    # Start the test run
    try:
        trainer.fit(model, train_dataloaders=train_dataloader)
        print("\nFast dev run completed successfully!")
        print("This means your training loop, data loading, and model forward pass are likely working.")
    except Exception as e:
        print(f"\nAn error occurred during the fast dev run: {e}")
        print("Please check the traceback for more details.")

    # # You can also test a specific number of batches:
    # print("\nStarting PyTorch Lightning training for a few batches...")
    # trainer_few_batches = pl.Trainer(
    #     max_epochs=1,
    #     limit_train_batches=3, # Run only 3 training batches
    #     accelerator="auto",
    #     devices=1,
    #     enable_checkpointing=False, # Disable checkpointing for quick tests
    #     logger=False # Disable logging for quick tests
    # )
    # try:
    #     trainer_few_batches.fit(model, train_dataloaders=train_dataloader)
    #     print("\nTraining for a few batches completed successfully!")
    # except Exception as e:
    #     print(f"\nAn error occurred during the few batches run: {e}")
    
    # print("AutoEncoder test passed!")
    
def test_ae_system_with_kaggle_dataset():
    from src.systems.autoencoder import AutoEncoder
    from src.models.vae import VAE
    from src.models.discriminator import Discriminator
    from src.models.lpips import LPIPS
    from src.data.kaggle_image_datamodule import KaggleImageDataModule
    
    model_ae = VAE(
        latent_dim=128,
        beta = 1,
    )
    
    model_d = Discriminator(
        in_channels=3,
    )
    
    model_lpips = LPIPS()
    
    model = AutoEncoder(
        model_ae=model_ae,
        model_d=model_d,
        lpips=model_lpips,
        loss_weights={"reconst": 1.0, "internal": 0.5, "perceptual": 0.5, "adversarial": 0.5},
        lr_g=1e-4,
        lr_d=1e-4,
        betas_g=(0.9, 0.999),
        betas_d=(0.9, 0.999),
    )
    
    # Assuming you have a Kaggle dataset path and other parameters defined
    kaggle_dataset_path = "badasstechie/celebahq-resized-256x256"  # Replace with your actual dataset path
    
    train_dataloader = KaggleImageDataModule(
        kaggle_dataset_path=kaggle_dataset_path,
        channels=3,
        image_size=(128, 128),
        train_val_test_split=(0.70, 0.15, 0.15),  # 70% train, 15% val, 15% test
        batch_size=3,
        num_workers=4,
        pin_memory=True,
    )
    
    print("Starting PyTorch Lightning fast_dev_run with Kaggle dataset...")
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="auto",
        devices=1
    )
    
    try:
        trainer.fit(model, train_dataloaders=train_dataloader)
        print("\nFast dev run with Kaggle dataset completed successfully!")
    except Exception as e:
        print(f"\nAn error occurred during the fast dev run with Kaggle dataset: {e}")
        
def test_ae_system_train_with_kaggle_dataset():
    from src.systems.autoencoder import AutoEncoder
    from src.models.vae import VAE
    from src.models.discriminator import Discriminator
    from src.models.lpips import LPIPS
    from src.data.kaggle_image_datamodule import KaggleImageDataModule
    
    model_ae = VAE(
        latent_dim=128,
        beta = 1,
    )
    
    model_d = Discriminator(
        in_channels=3,
    )
    
    model_lpips = LPIPS()
    
    model = AutoEncoder(
        model_ae=model_ae,
        model_d=model_d,
        lpips=model_lpips,
        loss_weights={"reconst": 1.0, "internal": 0.5, "perceptual": 0.5, "adversarial": 0.5},
        lr_g=1e-4,
        lr_d=1e-4,
        betas_g=(0.9, 0.999),
        betas_d=(0.9, 0.999),
    )
    
    # Assuming you have a Kaggle dataset path and other parameters defined
    kaggle_dataset_path = "badasstechie/celebahq-resized-256x256"  # Replace with your actual dataset path
    
    train_dataloader = KaggleImageDataModule(
        kaggle_dataset_path=kaggle_dataset_path,
        channels=3,
        image_size=(128, 128),
        train_val_test_split=(0.70, 0.15, 0.15),  # 70% train, 15% val, 15% test
        batch_size=3,
        num_workers=4,
        pin_memory=True,
    )

if __name__ == '__main__':
    test_ae_system_with_kaggle_dataset()
