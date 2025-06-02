import autoroot
import autorootcwd

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import lightning.pytorch as pl

import wandb

from src.utils.visualizers import convert_to_target_visible_channels


class DDPM(pl.LightningModule):
    def __init__(self, model_ae, model_diffusion, noise_scheduler, lr=1e-4, betas=(0.9, 0.999)):
        super(DDPM, self).__init__()
        self.save_hyperparameters(ignore=["model_ae", "model_diffusion", "noise_scheduler"])
        self.model_ae = model_ae.eval()
        
        for param in self.model_ae.parameters():
            param.requires_grad = False
        
        self.model_diffusion = model_diffusion
        self.noise_scheduler = noise_scheduler
        
        self.criterion = nn.MSELoss()

    # --- Logging utilities (copied/adapted from autoencoder.py) ---
    def _log_metric_values(self, stage, metric_type, metric_values, additional_context="", 
                           on_step=True, on_epoch=False, prog_bar=False, logger=True):
        for key, value in metric_values.items():
            names_list = [metric_type, key]
            if additional_context:
                names_list = [metric_type, additional_context, key]
            metric_name = "-".join(names_list)
            self.log(f"{stage}/{metric_name}", value, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger, sync_dist=True)

    def _log_comparison_images(self, stage, originals, latents, noised_latents, denoised_latents, step, n_samples=8):
        """
        Log comparison images: original, latent, noised latent, denoised latent.
        """
        # All tensors: (B, C, H, W)
        compare_images = [
            originals[:n_samples],
            latents[:n_samples],
            noised_latents[:n_samples],
            denoised_latents[:n_samples]
        ]
        comparison_images = torch.cat(compare_images)
        grid = make_grid(comparison_images, nrow=n_samples, padding=2)
        grid = (grid + 1) / 2
        grid = torch.clamp(grid, 0, 1)
        grid = to_pil_image(grid)  # Convert to PIL Image for better visualization in wandb
        self.logger.experiment.log({
            f"{stage}/comparison-images": wandb.Image(grid, caption=f"{stage} comparison images at epoch {self.current_epoch}")
        }, step=step)

    def _log_progressive_denoising(self, stage, x_latent, n_samples=4, step=None):
        """
        Log the progressive denoising process for a batch of latent images.
        """
        device = x_latent.device
        num_timesteps = self.noise_scheduler.num_timesteps
        B, C, H, W = x_latent.shape
        n_samples = min(B, n_samples)
        xt = torch.randn((n_samples, C, H, W), device=device)
        images = []
        
        self.model_diffusion.eval()
        with torch.no_grad():
            for i in reversed(range(num_timesteps)):
                t_tensor = torch.full((n_samples,), i, device=device, dtype=torch.long)
                noise_pred = self.model_diffusion(xt, t_tensor)
                xt, x0_pred = self.noise_scheduler.sample_prev_timestep(xt, noise_pred, t_tensor)
                # Only decode and save the final image to save time
                if i % (num_timesteps // 10) == 0:
                    decoded = self.model_ae.decode(xt)
                    images.append(decoded)
                # Optionally, you can append intermediate xt for visualization
                # else:
                    images.append(convert_to_target_visible_channels(xt, target_channels=3, resize=(decoded.shape[2], decoded.shape[3])))
        self.model_diffusion.train()

        # images: list of (n_samples, 3, H, W)
        if images:
            images = torch.cat(images, dim=0)  # (n_samples, 3, H, W)
            grid = make_grid(images, nrow=n_samples*2, padding=2)
            grid = (grid + 1) / 2
            grid = torch.clamp(grid, 0, 1)
            self.logger.experiment.log({
                f"{stage}/progressive-denoising": wandb.Image(grid, caption=f"{stage} progressive denoising at epoch {self.current_epoch}")
            }, step=step if step is not None else self.global_step)

    def forward(self, x, t):
        """
        Forward pass through the model.
        """
        return self.model_diffusion(x, t)
    
    def get_images(self, batch):
        """
        Get images from the batch.
        """
        return batch[0]
    
    def training_step(self, batch, batch_idx):
        x = self.get_images(batch)
        with torch.no_grad():
            latent = self.model_ae.encode(x)
        t = torch.randint(0, self.noise_scheduler.num_timesteps, (latent.size(0),), device=latent.device).long()
        noise = torch.randn_like(latent)
        x_t = self.noise_scheduler.add_noise(latent, noise, t).detach()
        noise_pred = self.model_diffusion(x_t, t)
        loss = self.criterion(noise_pred, noise)
        # Log train loss
        self._log_metric_values("Train", "Loss", {"ddpm": loss}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.get_images(batch)
        self.model_diffusion.eval()  # Ensure model is in eval mode
        with torch.no_grad():
            latent = self.model_ae.encode(x)
            t = torch.randint(0, self.noise_scheduler.num_timesteps, (latent.size(0),), device=latent.device).long()
            noise = torch.randn_like(latent)
            x_t = self.noise_scheduler.add_noise(latent, noise, t)
            noise_pred = self.model_diffusion(x_t, t)
            loss = self.criterion(noise_pred, noise)
        self.model_diffusion.train()
        # Log val loss
        self._log_metric_values("Validation", "Loss", {"ddpm": loss}, on_step=False, on_epoch=True, prog_bar=True)
        # No need to store last batch
        return loss

    def on_validation_epoch_end(self):
        # Log comparison images and progressive denoising
        # Use next(iter()) on the validation dataloader
        val_loader = self.trainer.datamodule.val_dataloader()
        val_batch = next(iter(val_loader))
        x = self.get_images(val_batch).to(self.device)
        self.model_diffusion.eval()  # Ensure model is in eval mode
        with torch.no_grad():
            latent = self.model_ae.encode(x)
            t = torch.randint(0, self.noise_scheduler.num_timesteps, (latent.size(0),), device=latent.device).long()
            noise = torch.randn_like(latent)
            x_t = self.noise_scheduler.add_noise(latent, noise, t)
            noise_pred = self.model_diffusion(x_t, t)
            denoised_latent, x0_pred = self.noise_scheduler.sample_prev_timestep(x_t, noise_pred, t)
        n_samples = min(x.shape[0], 8)
        # Visualize latent, noised_latent, denoised_latent as images
        latent_vis = convert_to_target_visible_channels(latent, target_channels=3, resize=(x.shape[2], x.shape[3]))
        noised_latent_vis = convert_to_target_visible_channels(x_t, target_channels=3, resize=(x.shape[2], x.shape[3]))
        denoised_latent_vis = convert_to_target_visible_channels(denoised_latent, target_channels=3, resize=(x.shape[2], x.shape[3]))
        self._log_comparison_images(
            "Validation",
            originals=x,
            latents=latent_vis,
            noised_latents=noised_latent_vis,
            denoised_latents=denoised_latent_vis,
            step=self.global_step,
            n_samples=n_samples
        )
        # Progressive denoising visualization
        self._log_progressive_denoising("Validation", latent, n_samples=4, step=self.global_step)
        self.model_diffusion.train()
    
    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.model_diffusion.parameters(), lr=self.hparams.lr, betas=self.hparams.betas)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, threshold=1e-4)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "Validation/Loss-ddpm",
                "frequency": 1,
            }
        }



