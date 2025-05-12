import autoroot
import autorootcwd

from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning import Trainer
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
    loss_weights={"reconst": 1.0, "internal": 1.0, "perceptual": 1.0, "adversarial": 1.0},
    d_loss_weight=1.0,
    lr_g=1e-4,
    lr_d=1e-4,
    betas_g=(0.9, 0.999),
    betas_d=(0.9, 0.999),
)

# Assuming you have a Kaggle dataset path and other parameters defined
kaggle_dataset_path = "badasstechie/celebahq-resized-256x256"  # Replace with your actual dataset path

kaggle_datamodule = KaggleImageDataModule(
    kaggle_dataset_path=kaggle_dataset_path,
    channels=3,
    image_size=(128, 128),
    train_val_test_split=(0.70, 0.15, 0.15),  # 70% train, 15% val, 15% test
    batch_size=64,
    num_workers=4,
    pin_memory=True,
)

wandb_logger = WandbLogger(
    project="Anime Auto Encoder",
    name="kaggle_dataset_vae_test_run",
    log_model="all",
)

trainer = Trainer(
    max_epochs=50,
    accelerator="auto",
    devices=1,
    logger=wandb_logger,
    log_every_n_steps=1,
)

trainer.fit(
    model,
    datamodule=kaggle_datamodule,
)