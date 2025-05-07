import torch

def format_input(x):
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if len(x.shape) != 4:
        raise ValueError(f"Input tensor must be 4D - B, C, H, W. Got {x.shape}")
    return x

def print_colored(text, color='white'):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    return f"{colors.get(color, colors['white'])}{text}{colors['reset']}"