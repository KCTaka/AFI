import torch
import torch.nn.functional as F
import math

def get_time_embedding(t: torch.Tensor, embedding_dim: int, max_period: int = 10000):
    half = embedding_dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / (half - 1) )
    # [batch, half]
    args = t[:, None] * freqs[None, :]  # [batch, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
        
    return emb  # [batch, embedding_dim]