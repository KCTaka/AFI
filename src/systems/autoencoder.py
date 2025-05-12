import torch
import torch.nn as nn
import pytorch_lightning as pl

import wandb

class AutoEncoder(pl.LightningModule):
    def __init__(self, model_ae, 
                 model_d,
                 lpips,
                 loss_weights = {"reconst": 1.0, "internal": 1.0, "perceptual": 1.0, "adversarial": 1.0},
                 lr_g = 2e-4,
                 lr_d = 2e-4,
                 betas_g = (0.9, 0.999),
                 betas_d = (0.9, 0.999),
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.criterion_reconst = nn.MSELoss()
        self.criterion_perceptual = lpips
        self.criterion_discriminator = nn.BCEWithLogitsLoss()
        
    def _forward(self, x):
        x_reconst, latent, loss_internal = self.hparams.model_ae(x)
        return x_reconst, latent, loss_internal    
    
    def forward(self, x):
        return self._forward(x)[0]
        
    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.hparams.model_ae.parameters(), lr=self.hparams.lr_g, betas=self.hparams.betas_g)
        optimizer_d = torch.optim.Adam(self.hparams.model_d.parameters(), lr=self.hparams.lr_d, betas=self.hparams.betas_d)
        
        return [optimizer_g, optimizer_d], []
    
    def get_images(self, batch):
        return batch[0]
    
    def get_discriminator_accuracy(self, preds, labels, threshold=0.5):
        binary_preds = preds > threshold
        accuracy = (binary_preds == labels).float().mean()
        return accuracy
    
    def _compute_losses(self, x, x_reconst, loss_internal):
        loss_reconst = self.criterion_reconst(x_reconst, x)
        loss_perceptual = self.criterion_perceptual(x_reconst, x)
        
        #### Adversarial loss ####
        pred_d = self.hparams.model_d(x_reconst).view(-1)
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
    
    def _log_metric_values(self, stage, model_type, metric_name, metric_values, sub_text="", on_step=True, on_epoch=False, 
                   prog_bar=False, 
                   logger=True):
        
        for key, value in metric_values.items():
            self.log(f"{stage}/{model_type}/{metric_name}/{key}_{sub_text}", value, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger)
            
    def _log_images(self, stage, images):
        """
        Log images to wandb.
        Args:
            stage (str): The stage of the training process (e.g., "train", "val", "test").
            model_type (str): The type of model (e.g., "autoencoder", "discriminator").
            image_type (str): The type of image (e.g., "input", "reconstructed").
            images (torch.Tensor): The images to log.
        """
        
        # Convert images to numpy and log them
        images = images.detach().cpu().numpy()
        images = (images + 1) / 2
        images = images.clip(0, 1)
        images = images.transpose(0, 2, 3, 1)  # Change to HWC format for logging
        self.logger.experiment.log({
            f"{stage}/autoencoder/images": wandb.Image(images, caption=f"{stage} images"),
            "global_step": self.global_step
        })
        
    def training_step(self, batch, batch_idx):
        x = self.get_images(batch)
        optimizer_g, optimizer_d = self.optimizers()
        x_reconst, latent, loss_internal = self._forward(x)
        
        self.toggle_optimizer(optimizer_g)
        loss_unweighted, loss_weighted = self._compute_losses(x, x_reconst, loss_internal)
        loss_weighted_avg = sum(loss_weighted.values()) / len(loss_weighted)
        self.manual_backward(loss_weighted_avg)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        
        self.toggle_optimizer(optimizer_d)
        pred_real = self.hparams.model_d(x).view(-1)
        pred_fake = self.hparams.model_d(x_reconst.detach()).view(-1)
        loss_d_real = self.criterion_discriminator(pred_real, torch.ones_like(pred_real))
        loss_d_fake = self.criterion_discriminator(pred_fake, torch.zeros_like(pred_fake))
        loss_d = (loss_d_real + loss_d_fake) / 2
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
        
        self._log_metric_values("train", "autoencoder", "loss", loss_unweighted, on_step=True, on_epoch=False, prog_bar=False)
        self._log_metric_values("train", "autoencoder", "loss", loss_weighted, sub_text="weighted", on_step=True, on_epoch=True, prog_bar=True)
        
        self._log_metric_values("train", "discriminator", "loss", losses_d, on_step=True, on_epoch=True, prog_bar=True)
        self._log_metric_values("train", "discriminator", "accuracy", accuracies, on_step=True, on_epoch=True, prog_bar=True)
        
        self._log_images("train-dataset", x[:16])
        self._log_images("train-reconstructed", x_reconst[:16])
        
    def validation_step(self, batch, batch_idx):
        x = self.get_images(batch)
        x_reconst, latent, loss_internal = self._forward(x)
        loss_unweighted, loss_weighted = self._compute_losses(x, x_reconst, loss_internal)
        
        self._log_metric_values("val", "autoencoder", "loss", loss_unweighted, on_step=False, on_epoch=True, prog_bar=True)
        self._log_metric_values("val", "autoencoder", "loss", loss_weighted, sub_text="weighted", on_step=False, on_epoch=True, prog_bar=True)
        
        self._log_images("val-dataset", x[:16])
        self._log_images("val-reconstructed", x_reconst[:16])
        
    def test_step(self, batch, batch_idx):
        x = self.get_images(batch)
        x_reconst, latent, loss_internal = self._forward(x)
        loss_unweighted, loss_weighted = self._compute_losses(x, x_reconst, loss_internal)
        
        self._log_metric_values("test", "autoencoder", "loss", loss_unweighted, on_step=False, on_epoch=True, prog_bar=True)
        self._log_metric_values("test", "autoencoder", "loss", loss_weighted, sub_text="weighted", on_step=False, on_epoch=True, prog_bar=True)
        
        self._log_images("test-dataset", x[:16])
        self._log_images("test-reconstructed", x_reconst[:16])