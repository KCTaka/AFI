import os
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torchvision.utils import make_grid

from models.autoencoder.utils import format_colored, format_input

class VAETrainer:
    def __init__(self, model, discriminator, lpips,
                 dataset_train, 
                 dataset_test=None, 
                 dataset_val=None,
                  batch_size=32, 
                  learning_rate=2e-5, 
                  epochs=10, 
                  save_every=5,
                  num_workers=4,
                  d_start_step=50,
                  run_name="vae_experiment",
                  local_dir_base='./runs/',):
        
        self.model = model
        self.discriminator = discriminator
        self.lpips = lpips
        self.dataset_train = dataset_train 
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.save_every = save_every
        self.run_name = run_name
        
        self.d_start_step = d_start_step
        self.start_epoch = 0
        
        
        self.local_dir = os.path.join(local_dir_base, run_name)
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)
            tqdm.write(format_colored(f"Created directory: {self.local_dir}", color='blue'))
        
        # Determine device from model parameters
        self.device = next(model.parameters()).device
        
        # DataLoaders
        self.train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if dataset_val else None
        self.test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if dataset_test else None
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
        
        # Loss functions
        self.l2_loss = nn.MSELoss()
        self.d_criterion = nn.BCEWithLogitsLoss() # New: Correct for PatchGAN logits output
        self.lpips_loss = lpips
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.local_dir)

        
    def save_checkpoint(self, epoch):
        # Ensure checkpoints dir exists and under that the run_name dir exists
        checkpoint_dir = os.path.join("checkpoints", self.run_name)
        
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            tqdm.write(format_colored(f"Created checkpoint directory: {checkpoint_dir}", color='blue'))
            
        checkpoint_name = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),            
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
        }
        
        torch.save(checkpoint, checkpoint_name)
        tqdm.write(format_colored(f"Checkpoint saved at {checkpoint_name}", color='blue'))
        
    def _find_checkpoint(self, load_epoch="latest"):
        # if checkpoint_epoch is None, find the latest checkpoint (largest epoch number)
        checkpoint_dir = os.path.join("checkpoints", self.run_name)
        
        if load_epoch != "latest":
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{load_epoch}.pth")
            if os.path.exists(checkpoint_path):
                return checkpoint_path
            else:
                return None
            
        if not os.path.exists(checkpoint_dir):
            return None
        
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if not checkpoint_files:
            return None
        
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
        return os.path.join(checkpoint_dir, latest_checkpoint)
        
        
    def load_checkpoint(self, load_epoch='latest'):
        # if checkpoint_epoch is None, find the latest checkpoint (largest epoch number)
        checkpoint_path = self._find_checkpoint(load_epoch)
        if checkpoint_path is None:
            tqdm.write(format_colored("No checkpoint found. Starting from scratch.", color='red'))
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] 
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        tqdm.write(format_colored(f"Checkpoint loaded from {checkpoint_path}", color='blue'))
        
    def _collect_sample(self, data):    
        return data[0]
    
    def _denormalize_images(self, x):
        mean_tensor = torch.tensor([0.5] * x.shape[1], device=self.device).view(1, x.shape[1], 1, 1)
        std_tensor = torch.tensor([0.5] * x.shape[1], device=self.device).view(1, x.shape[1], 1, 1)
        return x.clone() * std_tensor + mean_tensor
            
    def _compare_images(self):
         # Take a random sample from the dataset, compare it with the reconstructed image
        sample_x_display = self._collect_sample(next(iter(self.train_dataloader)))
        sample_x_display = sample_x_display.to(self.device) # Use self.device
        with torch.no_grad():
            reconst_display = self.model(sample_x_display)[0]
        
        # Denormalize images for display
        denorm_sample_x_display = self._denormalize_images(sample_x_display)
        denorm_reconst_display = self._denormalize_images(reconst_display)
        
        num_samples_display = min(8, sample_x_display.size(0))
        
        # Create a comparison image grid
        comparison_orig = denorm_sample_x_display[:num_samples_display].cpu().clamp(0, 1)
        comparison_reconst = denorm_reconst_display[:num_samples_display].cpu().clamp(0, 1)
        
        if comparison_orig.shape[1] == 1: # Grayscale to RGB for consistent grid display if needed
            comparison_orig = comparison_orig.repeat(1, 3, 1, 1)
            comparison_reconst = comparison_reconst.repeat(1, 3, 1, 1)
        
        return comparison_orig, comparison_reconst
    
    def _log_comparison_images(self, orig, reconst, epoch):
        # Log the comparison images to TensorBoard
        num_samples = min(8, orig.size(0))
        img_grid = torch.cat([orig, reconst], dim=0)
        grid_tensor = make_grid(img_grid, nrow=num_samples, normalize=True)
        self.writer.add_image('Reconstructed Images Comparison', grid_tensor, epoch)
        tqdm.write(format_colored("Comparison images logged to TensorBoard.", color='blue'))
        
    def _fit(self, x, global_step):
        x = x.to(self.device)
        batch_num = x.size(0)
        x_reconst, z, internal_loss = self.model(x)
        recon_loss = self.l2_loss(x_reconst, x)
        perceptual_loss = self.lpips(x_reconst, x)
        
        if global_step > self.d_start_step:
            # Compute adversarial loss only after the first epoch
            pred_g = self.discriminator(x_reconst)
            label_real_g = torch.ones_like(pred_g).to(self.device)
            g_adv_loss = self.d_criterion(pred_g, label_real_g)
        
        loss = recon_loss + internal_loss + perceptual_loss + g_adv_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return x_reconst, recon_loss.item(), internal_loss.item(), perceptual_loss.item(), g_adv_loss.item()
    
    def train(self, load_epoch='latest'):
        # Log model graph to TensorBoard
        try:
            data_iter = iter(self.train_dataloader)
            sample_images = self._collect_sample(next(data_iter))
            sample_images = sample_images.to(self.device)
            self.writer.add_graph(self.model, sample_images)
            tqdm.write(format_colored("Model graph added to TensorBoard.", color='blue'))
        except Exception as e:
            tqdm.write(format_colored(f"Could not add model graph to TensorBoard: {e}", color='yellow'))
            
        if load_epoch is not None:
            self.load_checkpoint(load_epoch)
            
        for epoch in tqdm(range(self.start_epoch, self.epochs), desc="Training Progress ", unit="epoch", initial=self.start_epoch, total=self.epochs):
            self.model.train()
            total_loss = 0
            
            num_batches = len(self.train_dataloader)
            total_losses = {'Reconstruct Loss': 0, 'Internal Loss': 0, 'Perceptual Loss': 0, 'Adversarial Loss': 0}
            for batch_idx, data in tqdm(enumerate(self.train_dataloader), desc="Batch Progress ", unit="batch", total=num_batches, leave=False):
                global_step = epoch * len(self.train_dataloader) + batch_idx       
                x = self._collect_sample(data)
                x_reconst, recon_loss, internal_loss, perceptual_loss, g_adv_loss = self._fit(x, global_step)
                         
                self.writer.add_scalars('VAE/Train Losses', {
                    'Reconstruction Loss': recon_loss,
                    'Internal Loss': internal_loss,
                    'Perceptual Loss': perceptual_loss,
                    'Adversarial Loss': g_adv_loss,
                }, global_step)
                
                total_losses['Reconstruct Loss'] += recon_loss
                total_losses['Internal Loss'] += internal_loss
                total_losses['Perceptual Loss'] += perceptual_loss
                total_losses['Adversarial Loss'] += g_adv_loss
                
                if batch_idx % 10 == 0:
                    loss = recon_loss + internal_loss + perceptual_loss + g_adv_loss
                    tqdm.write(format_colored(f"Epoch [{epoch}/{self.epochs}] |\tStep [{batch_idx}/{len(self.train_dataloader)}] |\t\
                        Total Loss: {loss:.4f} |", color='green'))

                ########## Update Discriminator ##########
                fake_images = x_reconst.detach()
                real_images = x.detach()
                
                pred_fake = self.discriminator(fake_images)
                pred_real = self.discriminator(real_images)
                
                label_fake = torch.zeros_like(pred_fake).to(self.device)
                label_real = torch.ones_like(pred_real).to(self.device)
                
                loss_fake = self.d_criterion(pred_fake, label_fake)
                loss_real = self.d_criterion(pred_real, label_real)
                d_loss = (loss_fake + loss_real) / 2
                
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()
                
                self.writer.add_scalar('Discriminator/Fake Loss', loss_fake, global_step)
                self.writer.add_scalar('Discriminator/Real Loss', loss_real, global_step)
                self.writer.add_scalar('Discriminator/Total Loss', d_loss, global_step)
                ###########################################
                
                # Log embeddings for the epoch
                feature_vector = z.cpu().detach().reshape(batch_num, -1)
                label_image = x.cpu().detach()
                self.writer.add_embedding(feature_vector, label_img=label_image, global_step=global_step)
                
                self.writer.flush()
            
            for key, value in total_losses.items():
                self.writer.add_scalars(f'VAE/{key}', {'Train': value / len(self.train_dataloader)}, epoch)
            self.writer.add_scalars('VAE/Total Loss', {'Train': (sum(total_losses.values()) / len(self.train_dataloader))}, epoch)
            

            avg_loss = total_loss / len(self.train_dataloader)
            tqdm.write(format_colored(f">>> Epoch [{epoch}] Average Loss: {avg_loss:.4f}", color='blue'))
            
            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch)
                
            orig, reconst = self._compare_images()
            self._log_comparison_images(orig, reconst, epoch)
            self.writer.flush()
            
            self.model.eval()
            total_losses = {'Reconstruct Loss': 0, 'Internal Loss': 0, 'Perceptual Loss': 0, 'Adversarial Loss': 0}
            for batch_idx, data in tqdm(enumerate(self.val_dataloader), desc="Validation Progress ", unit="batch", total=len(self.val_dataloader), leave=False):
                x = self._collect_sample(data)
                batch_num = x.size(0)
                with torch.no_grad():
                    _, recon_loss, internal_loss, perceptual_loss, g_adv_val_loss = self._fit(x, global_step)
                
                total_losses['Reconstruct Loss'] += recon_loss
                total_losses['Internal Loss'] += internal_loss
                total_losses['Perceptual Loss'] += perceptual_loss
                total_losses['Adversarial Loss'] += g_adv_val_loss
                
                if batch_idx % 10 == 0:
                    loss = recon_loss + internal_loss + perceptual_loss + g_adv_val_loss
                    tqdm.write(format_colored(f"Validation - Epoch [{epoch}/{self.epochs}] | Batch [{batch_idx}/{len(self.val_dataloader)}] | Total Loss: {loss:.4f}", color='blue'))
            
            for key, value in total_losses.items():
                self.writer.add_scalars(f'VAE/{key}', {'Validation': value / len(self.val_dataloader)}, epoch)
            self.writer.add_scalars('VAE/Total Loss', {'Validation': (sum(total_losses.values()) / len(self.val_dataloader))}, epoch)
                
if __name__ == '__main__':
    # Define dataset and transformations
    from models.autoencoder.vae import VAE
    from models.autoencoder.vqvae import VQVAE
    from models.autoencoder.discriminator import Discriminator
    from models.lpips import LPIPS
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("badasstechie/celebahq-resized-256x256")  
    
    tqdm.write(format_colored(f"Dataset downloaded to {path}", color='blue'))
    
        
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = datasets.ImageFolder(root=path, transform=transform)
    dataset_split = [0.70, 0.15, 0.15] # Train, Validation, Test split
    dataset_train, dataset_val, dataset_test = random_split(dataset, dataset_split)

    print(f"Loaded dataset with {len(dataset)} images.")
    
    # Initialize model
    # model = VAE(latent_dim=128).to(device) # Assuming VAE is modified to take input_channels
    model = VQVAE(embedding_dim=1024, num_embeddings=512, beta=0.25).to(device) # Assuming VQVAE is modified to take input_channels
    
    # Loss functions
    discriminator = Discriminator(in_channels=3).to(device) # Assuming Discriminator is modified to take input_channels
    lpips = LPIPS().to(device) # Assuming LPIPS is modified to take input_channels
    
    # Initialize trainer
    trainer = VAETrainer(model, discriminator, lpips,
                         dataset_train=dataset_train,
                        dataset_val=dataset_val,
                        dataset_test=dataset_test,
                         batch_size=64, 
                         learning_rate=2e-5,
                         epochs=50, 
                         save_every=1,
                         run_name="vqvae_experiment_3",
                         )
    
    # Train the model
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.writer.close()
        





