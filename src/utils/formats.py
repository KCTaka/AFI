

def format_input(x):
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if len(x.shape) != 4:
        raise ValueError(f"Input tensor must be 4D - B, C, H, W. Got {x.shape}")
    return x