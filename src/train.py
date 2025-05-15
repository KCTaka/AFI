import autoroot
import autorootcwd

from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint # Added import
from src.systems.autoencoder import AutoEncoder
from src.models.vae import VAE
from src.models.vqvae import VQVAE
from src.models.discriminator import Discriminator
from src.models.lpips import LPIPS
from src.data.kaggle_image_datamodule import KaggleImageDataModule

model_ae = VQVAE(
    embedding_dim=128,
    num_embeddings=512,
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
    lr_g=2e-5,
    lr_d=2e-5,
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
    batch_size=50*3,
    num_workers=4,
    pin_memory=True,
)

# Checkpoint callback for saving the last 5 models
checkpoint_callback_last = ModelCheckpoint(
    monitor='epoch', # Monitor epochs
    mode='max',      # Keep the ones with max epoch number
    save_top_k=5,
    filename='model-epoch{epoch:02d}-last',
    dirpath='checkpoints/last_models/'
)

# Checkpoint callback for saving the 3 best models based on validation loss
checkpoint_callback_best_val_loss = ModelCheckpoint(
    monitor="Validation/Loss-weighted-average",
    mode="min",
    save_top_k=3,
    filename='model-epoch{epoch:02d}-val_loss{Validation/Loss-weighted-average:.2f}',
    dirpath='checkpoints/best_val_loss_models/'
)

wandb_logger = WandbLogger(
    project="Anime Auto Encoder",
    name="kaggle_dataset_vqvae_test_run",
)

trainer = Trainer(
    max_epochs=50,
    accelerator="auto",
    devices="auto",
    logger=wandb_logger,
    strategy=DDPStrategy(find_unused_parameters=True),
    callbacks=[checkpoint_callback_last, checkpoint_callback_best_val_loss] # Added callbacks
)

trainer.fit(
    model,
    datamodule=kaggle_datamodule,
)