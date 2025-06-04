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

class AutoEncoder(pl.LightningModule):
    def __init__(self, model_ae, 
                 model_d,
                 lpips,
                 loss_weights = {"reconst": 1.0, "internal": 1.0, "perceptual": 1.0, "adversarial": 1.0},
                 d_loss_weight = 1.0,
                 lr_g = 2e-4,
                 lr_d = 2e-4,
                 betas_g = (0.9, 0.999),
                 betas_d = (0.9, 0.999),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model_ae', 'model_d', 'lpips']) # Ignore modules for hparams logging if preferred
        self.automatic_optimization = False

        self.model_ae = model_ae
        self.model_d = model_d
        
        self.criterion_reconst = nn.MSELoss()
        self.criterion_perceptual = lpips
        self.criterion_discriminator = nn.BCEWithLogitsLoss()
        
    def _forward(self, x):
        x_reconst, latent, loss_internal = self.model_ae(x)
        return x_reconst, latent, loss_internal    
    
    def forward(self, x):
        return self._forward(x)[0]
        
    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.model_ae.parameters(), lr=self.hparams.lr_g, betas=self.hparams.betas_g)
        optimizer_d = torch.optim.Adam(self.model_d.parameters(), lr=self.hparams.lr_d, betas=self.hparams.betas_d)
        
        scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=3)
        return (
            {
                "optimizer": optimizer_g,
                "lr_scheduler": {
                    "scheduler": scheduler_g,
                    "monitor": "Validation/percep_recon_loss",
                    "frequency": 1,
                }
            },
            {
                "optimizer": optimizer_d,
            }
        )
    
    def get_images(self, batch):
        return batch[0]
    
    def get_discriminator_accuracy(self, preds, labels, threshold=0.5):
        binary_preds = preds > threshold
        accuracy = (binary_preds == labels).float().mean()
        return accuracy
    
    def _compute_losses(self, x, x_reconst, loss_internal):
        loss_reconst = self.criterion_reconst(x_reconst, x)
        loss_perceptual = self.criterion_perceptual(torch.clamp(x_reconst, -1, 1), x)
        
        #### Adversarial loss ####
        pred_d = self.model_d(x_reconst).view(-1)
        label_real = torch.ones_like(pred_d)
        loss_adversarial = self.criterion_discriminator(pred_d, label_real)
        
        loss_unweighted = {
            "reconst": loss_reconst,
            "internal": loss_internal,
            "perceptual": loss_perceptual,
            "adversarial": loss_adversarial
        }
        
        loss_weighted = {
            "reconst": loss_reconst * self.hparams.loss_weights["reconst"],
            "internal": loss_internal * self.hparams.loss_weights["internal"],
            "perceptual": loss_perceptual * self.hparams.loss_weights["perceptual"],
            "adversarial": loss_adversarial * self.hparams.loss_weights["adversarial"]
        }
        
        return loss_unweighted, loss_weighted
    
    def _log_metric_values(self, stage, metric_type, metric_values, additional_context="", 
                           on_step=True, on_epoch=False, prog_bar=False, logger=True):

        for key, value in metric_values.items():
            names_list = [metric_type, key]
            if additional_context:
                names_list = [metric_type, additional_context, key]
            metric_name = "-".join(names_list)  
            self.log(f"{stage}/{metric_name}", value, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger, sync_dist=True)

    def _log_comparison_images(self, stage, x, x_reconst, step, latent_images=None, n_samples=8):
        """
        Log images to wandb.
        Args:
            stage (str): The stage of the training process (train, val, test).
            x (torch.Tensor): The original images.
            x_reconst (torch.Tensor): The reconstructed images.
            epoch (int): The current epoch number.
            n_samples (int): The number of samples to log.
        """
        compare_images = [x[:n_samples], x_reconst[:n_samples]]

        if latent_images is not None:
            compare_images = [x[:n_samples], latent_images[:n_samples], x_reconst[:n_samples]]

        comparison_images = torch.cat(compare_images)
        grid = make_grid(comparison_images, nrow=n_samples, padding=2)

        # denorm with mean and std
        grid = (grid + 1) / 2
        grid = torch.clamp(grid, 0, 1)
        grid = to_pil_image(grid)
        self.logger.experiment.log({
            f"{stage}/comparison-images": wandb.Image(grid, caption=f"{stage} comparison images at epoch {self.current_epoch}")
        }, step=step)
        
    def training_step(self, batch, batch_idx):
        x = self.get_images(batch)
        optimizer_g, optimizer_d = self.optimizers()
        x_reconst, latent, loss_internal = self._forward(x)
        
        self.toggle_optimizer(optimizer_g)
        loss_unweighted, loss_weighted = self._compute_losses(x, x_reconst, loss_internal)
        self.manual_backward(sum(loss_weighted.values()))
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        
        self.toggle_optimizer(optimizer_d)
        pred_real = self.model_d(x).view(-1)
        pred_fake = self.model_d(x_reconst.detach()).view(-1)
        loss_d_real = self.criterion_discriminator(pred_real, torch.ones_like(pred_real))
        loss_d_fake = self.criterion_discriminator(pred_fake, torch.zeros_like(pred_fake))
        loss_d = (loss_d_real + loss_d_fake) / 2 * self.hparams.d_loss_weight
        self.manual_backward(loss_d)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        
        ### Calculate patch discriminator accuracy
        true_positive = self.get_discriminator_accuracy(pred_real, torch.ones_like(pred_real))
        true_negative = self.get_discriminator_accuracy(pred_fake, torch.zeros_like(pred_fake))
        accuracy = (true_positive + true_negative) / 2
        
        losses_d = {
            "real": loss_d_real,
            "fake": loss_d_fake,
            "total": loss_d
        }
        
        accuracies = {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "accuracy": accuracy
        }
        
        loss_weighted["sum"] = sum(loss_weighted.values())
        loss_unweighted["sum"] = sum(loss_unweighted.values())
        
        self._log_metric_values("Train", "Loss", loss_weighted, additional_context="weighted", on_step=True, on_epoch=True)
        self._log_metric_values("Train", "Loss", loss_unweighted, on_step=True, on_epoch=True)
        
        self._log_metric_values("Train", "Discriminator-Loss", losses_d, on_step=True, on_epoch=True)
        self._log_metric_values("Train", "Accuracy", accuracies, on_step=True, on_epoch=True)
        
    def validation_step(self, batch, batch_idx):
        x = self.get_images(batch)
        x_reconst, latent, loss_internal = self._forward(x)
        loss_unweighted, loss_weighted = self._compute_losses(x, x_reconst, loss_internal)
        
        loss_weighted["sum"] = sum(loss_weighted.values())
        loss_unweighted["sum"] = sum(loss_unweighted.values())
        
        self._log_metric_values("Validation", "Loss", loss_weighted, additional_context="weighted", on_step=False, on_epoch=True)
        self._log_metric_values("Validation", "Loss", loss_unweighted, on_step=False, on_epoch=True)
        
        percep_recon_loss = loss_weighted["perceptual"] + loss_weighted["reconst"]
        self.log("Validation/percep_recon_loss", percep_recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def on_validation_epoch_end(self):
        val_loader = self.trainer.datamodule.val_dataloader()
        
        
        val_batch = next(iter(val_loader))
        x = self.get_images(val_batch).to(self.device)
        n_samples = min(x.shape[0], 8)
        x = x[:n_samples]
        
        self.eval()
        with torch.no_grad():
            x_reconst, latent, _ = self._forward(x)
        self.train()
        
        # if self.trainer.state.stage == "sanity_check":
        #     return
        
        latent_images = convert_to_target_visible_channels(latent, target_channels=3, resize=(x.shape[2], x.shape[3]))
        self._log_comparison_images("Validation", x, x_reconst, self.global_step, 
                                    latent_images=latent_images, n_samples=n_samples)
        
    def test_step(self, batch, batch_idx):
        x = self.get_images(batch)
        x_reconst, latent, loss_internal = self._forward(x)
        loss_unweighted, loss_weighted = self._compute_losses(x, x_reconst, loss_internal)
        
        loss_weighted["sum"] = sum(loss_weighted.values())
        loss_unweighted["sum"] = sum(loss_unweighted.values())
        
        self._log_metric_values("Test", "Loss", loss_weighted, additional_context="weighted", on_step=False, on_epoch=True)
        self._log_metric_values("Test", "Loss", loss_unweighted, on_step=False, on_epoch=True)
        
    def on_test_epoch_end(self):
        test_loader = self.trainer.datamodule.test_dataloader()
        test_batch = next(iter(test_loader))
        x = self.get_images(test_batch).to(self.device)
        n_samples = min(x.shape[0], 8)
        x = x[:n_samples]
        
        self.eval()
        with torch.no_grad():
            x_reconst, latent, _ = self._forward(x)
        self.train()
        
        # if self.trainer.state.stage == "sanity_check":
        #     return
        
        latent_images = convert_to_target_visible_channels(latent, target_channels=3, resize=(x.shape[2], x.shape[3]))
        self._log_comparison_images("Test", x, x_reconst, self.global_step, 
                                    latent_images=latent_images, n_samples=n_samples)
