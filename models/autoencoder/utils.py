import torch

def format_input(x):
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if len(x.shape) != 4:
        raise ValueError(f"Input tensor must be 4D - B, C, H, W. Got {x.shape}")
    return x

def format_colored(text, color='white'):
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
    if color not in colors:
        return text 
    return f"{colors[color]}{text}{colors['reset']}"