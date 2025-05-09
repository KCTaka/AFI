from tqdm import tqdm

TEXT_COLORS = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'magenta': '\033[95m',
    'cyan': '\033[96m',
    'white': '\033[97m',
    'reset': '\033[0m'
}

def print_colored(text, color='white'):
    if color not in TEXT_COLORS:
        print(text)
    else:
        print(f"{TEXT_COLORS[color]}{text}{TEXT_COLORS['reset']}")
        
def tqdm_write_colored(text, color='white'):
    if color not in TEXT_COLORS:
        tqdm.write(text)
    else:
        tqdm.write(f"{TEXT_COLORS[color]}{text}{TEXT_COLORS['reset']}")