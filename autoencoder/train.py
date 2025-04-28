import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from models import VQVAE, VAE
import multiprocessing
import torch_directml
import scipy  # Required for downloading imagenet dataset

# Select GPU device if available, otherwise use CPU
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch_directml.is_available():
        return torch_directml.device()
    else:
        return torch.device('cpu')

def main():
    device = get_device()
    
    # Hyperparameters
    image_size = 128
    batch_size = 32
    embedding_dim = 128
    num_embeddings = 512
    learning_rate = 2e-4
    epochs = 32
    save_every = 5

    # Data transforms for Food101 dataset with normalization
    # Food101 normalization values (approximately ImageNet values)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Load ImageNet dataset
    data_dir = './data'
    dataset = datasets.Food101(root=data_dir, transform=transform, download=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=4, pin_memory=True)

    # Model
    model = VQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)

            x_reconst, q_loss = model(x)
            recon_loss = nn.functional.mse_loss(x_reconst, x)
            loss = recon_loss + q_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f">>> Epoch [{epoch}] Average Loss: {avg_loss:.4f}")

        # Save reconstructed images
        if epoch % save_every == 0:
            model.eval()
            with torch.no_grad():
                sample_x, _ = next(iter(dataloader))
                sample_x = sample_x.to(device)
                reconst, _ = model(sample_x)
                
                # Denormalize images
                for t, m, s in zip(sample_x.transpose(1, 0), mean, std):
                    t.mul_(s).add_(m)
                for t, m, s in zip(reconst.transpose(1, 0), mean, std):
                    t.mul_(s).add_(m)
                
                save_image(torch.cat([sample_x[:8], reconst[:8]], dim=0), 
                           f"food101_recon_epoch{epoch}.png", nrow=8, normalize=True)

            torch.save(model.state_dict(), f"food101_vqvae_epoch{epoch}.pt")
            model.train()  # Switch back to training mode

if __name__ == '__main__':
    # This is needed for Windows to support multiprocessing with DataLoader
    torch.multiprocessing.freeze_support()
    main()