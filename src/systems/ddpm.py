import autoroot
import autorootcwd

import torch
import torch.nn as nn
from torchvision.utils import make_grid
import lightning.pytorch as pl

import wandb

from src.utils.visualizers import convert_to_target_visible_channels


class DDPM(pl.LightningModule):
    def __init__(self, model_ae, model, noise_scheduler, lr=1e-4, betas=(0.9, 0.999)):
        super(DDPM, self).__init__()
        self.save_hyperparameters(ignore=["model_ae", "model", "noise_scheduler"])
        self.model_ae = model_ae
        self.model = model
        self.noise_scheduler = noise_scheduler

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
        self.logger.experiment.log({
            f"{stage}/comparison-images": wandb.Image(grid, caption=f"{stage} comparison images at epoch {self.current_epoch}")
        }, step=step)

    def _log_progressive_denoising(self, stage, x_latent, n_samples=4, step=None):
        """
        Log the progressive denoising process for a batch of latent images.
        """
        device = x_latent.device
        num_timesteps = self.noise_scheduler.num_timesteps
        # Start from pure noise
        B, C, H, W = x_latent.shape
        n_samples = min(B, n_samples)
        x = torch.randn(n_samples, C, H, W, device=device)
        images = []
        t_range = torch.arange(num_timesteps-1, -1, -1, device=device)
        xt = x
        for t in t_range:
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            noise_pred = self.model(xt, t_batch)
            xt = self.noise_scheduler.sample_prev_timestep(xt, noise_pred, t_batch)
            # Decode and visualize at intervals (e.g., every 1/8th of the process)
            if t % max(1, num_timesteps // 8) == 0 or t == 0:
                decoded = self.model_ae.decode(xt)
                decoded_vis = convert_to_target_visible_channels(decoded, target_channels=3)
                images.append(decoded_vis)
        # images: list of (n_samples, 3, H, W)
        # Stack as (num_steps, n_samples, 3, H, W) -> (num_steps * n_samples, 3, H, W)
        images = torch.stack(images, dim=0).reshape(-1, 3, H, W)
        grid = make_grid(images, nrow=n_samples, padding=2)
        grid = (grid + 1) / 2
        grid = torch.clamp(grid, 0, 1)
        self.logger.experiment.log({
            f"{stage}/progressive-denoising": wandb.Image(grid, caption=f"{stage} progressive denoising at epoch {self.current_epoch}")
        }, step=step if step is not None else self.global_step)

    def forward(self, x, t):
        """
        Forward pass through the model.
        """
        return self.model(x, t)
    
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
        x_t = self.noise_scheduler.add_noise(latent, noise, t)
        noise_pred = self.model(x_t, t)
        loss = self.noise_scheduler.loss_fn(noise_pred, noise)
        # Log train loss
        self._log_metric_values("Train", "Loss", {"ddpm": loss}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.get_images(batch)
        with torch.no_grad():
            latent = self.model_ae.encode(x)
            t = torch.randint(0, self.noise_scheduler.num_timesteps, (latent.size(0),), device=latent.device).long()
            noise = torch.randn_like(latent)
            x_t = self.noise_scheduler.add_noise(latent, noise, t)
            noise_pred = self.model(x_t, t)
            loss = self.noise_scheduler.loss_fn(noise_pred, noise)
            # Denoise the noised latent (one step)
            denoised_latent = self.noise_scheduler.sample_prev_timestep(x_t, noise_pred, t)
        # Log val loss
        self._log_metric_values("Validation", "Loss", {"ddpm": loss}, on_step=False, on_epoch=True, prog_bar=True)
        # Save for on_validation_epoch_end
        self._last_val_batch = {
            "x": x.detach().cpu(),
            "latent": latent.detach().cpu(),
            "noised_latent": x_t.detach().cpu(),
            "denoised_latent": denoised_latent.detach().cpu()
        }
        return loss

    def on_validation_epoch_end(self):
        # Log comparison images and progressive denoising
        if not hasattr(self, "_last_val_batch"):
            return
        x = self._last_val_batch["x"].to(self.device)
        latent = self._last_val_batch["latent"].to(self.device)
        noised_latent = self._last_val_batch["noised_latent"].to(self.device)
        denoised_latent = self._last_val_batch["denoised_latent"].to(self.device)
        n_samples = min(x.shape[0], 8)
        # Visualize latent, noised_latent, denoised_latent as images
        latent_vis = convert_to_target_visible_channels(latent, target_channels=3, resize=(x.shape[2], x.shape[3]))
        noised_latent_vis = convert_to_target_visible_channels(noised_latent, target_channels=3, resize=(x.shape[2], x.shape[3]))
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

