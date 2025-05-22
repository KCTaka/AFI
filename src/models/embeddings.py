import autoroot
import autorootcwd

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from src.utils.formats import format_input

# TimeEmbedding based on https://nn.labml.ai/diffusion/ddpm/unet.html
class TimeEmbedding(nn.Module):
    def __init__(self, out_dim: int, time_dim: int, max_time: int = 10000):
        """Time embedding for a given number of dimensions and maximum time. Ensure that the number of dimensions is divisible by 4.
        The embedding is computed using a linear layer followed by a SiLU activation and another linear layer.

        Args:
            time_dim (int): The number of dimensions for the embedding.
            max_time (int, optional): The maximum time value for the embedding. Defaults to 1000.
        """
        super(TimeEmbedding, self).__init__()
        self.time_dim = time_dim
        self.max_time = max_time

        self.mlp = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        
    def get_time_embedding(self, t: torch.Tensor, embedding_dim: int, max_period: int = 10000):
        half = embedding_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / (half - 1) )
        # [batch, half]
        args = t[:, None] * freqs[None, :]  # [batch, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
            
        return emb  # [batch, embedding_dim]
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        time = format_input(time, expected_input_dim=1).view(-1)
        emb = self.get_time_embedding(time, self.time_dim, self.max_time)
        return self.mlp(emb)
    
if __name__ == "__main__":
    import torch_directml
    device = torch_directml.device()
    
    # Example usage
    time_dim = 512
    out_dim = 128
    time_embedding = TimeEmbedding(out_dim=out_dim, time_dim=time_dim).to(device)
    time_steps = torch.randint(0, 10, (4,1)).to(device)
    # print(time_steps)
    # print(time_embedding(time_steps))


    # Time test
    import time

    # Test all the time embeddings
    time_record = []
    for i in range(1000):
        start = time.time()
        time_embedding(time_steps)
        end = time.time()
        time_record.append(end - start)
        
    print(f"Average time total: {sum(time_record) / len(time_record)}")
    
    # Test the time embedding generation
    time_record = []
    for i in range(1000):
        start = time.time()
        time_embedding.get_time_embedding(time_steps, time_dim)
        end = time.time()
        time_record.append(end - start)
    
    print(f"Average time get_time_embedding: {sum(time_record) / len(time_record)}")
    
    # Test the time embedding generation
    time_record = []
    for i in range(1000):
        emb = time_embedding.get_time_embedding(time_steps.view(-1), time_dim)
        start = time.time()
        time_embedding.mlp(emb)
        end = time.time()
        time_record.append(end - start)

    print(f"Average time mlp: {sum(time_record) / len(time_record)}")
