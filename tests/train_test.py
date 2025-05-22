import autoroot
import autorootcwd

def train_test():
    import torch
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch import Trainer
    from lightning.pytorch.strategies import DDPStrategy
    from lightning.pytorch.callbacks import ModelCheckpoint # Added import
    from src.systems.autoencoder import AutoEncoder
    from src.models.vae import VAE
    from src.models.vqvae import VQVAE
    from src.models.discriminator import PatchGAN
    from src.models.lpips import LPIPS
    from src.data.kaggle_image_datamodule import KaggleImageDataModule
    
    model_ae = VQVAE(
        embedding_dim=4,
        num_embeddings=8192,
        beta=0.25,
        im_channels=3,
        down_channels=[64, 128, 256, 256],
        mid_channels=[256, 256],
        downsamples=[True, True, True],
        down_attn=[False, False, False],
        num_heads=4,
        num_down_layers=2,
        num_mid_layers=2,
        num_up_layers=2,
    )

    model_d = PatchGAN(
        in_channels=3,
        conv_channels=[64, 128, 256],
    )

    model_lpips = LPIPS()

    model = AutoEncoder(
        model_ae=model_ae,
        model_d=model_d,
        lpips=model_lpips,
        loss_weights={"reconst": 10.0, "internal": 1.0, "perceptual": 0.5, "adversarial": 0.5},
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
        train_val_test_split={"train": 0.8, "val": 0.1, "test": 0.1},  # 80% train, 10% val, 10% test
        batch_size=45*3,
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
        log_model="all",
    )

    trainer = Trainer(
        max_epochs=50,
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback_last, checkpoint_callback_best_val_loss], # Added callbacks
        log_every_n_steps=5,
    )

    trainer.fit(
        model,
        datamodule=kaggle_datamodule,
    )