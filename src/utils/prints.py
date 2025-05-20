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
def format_text_with_color(text, color='white'):
    """
    Format text with ANSI color codes.
    
    Args:
        text (str): The text to format.
        color (str): The color to use. Options are 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.
    
    Returns:
        str: The formatted text with ANSI color codes.
    """
    if color not in TEXT_COLORS:
        return text
    return f"{TEXT_COLORS[color]}{text}{TEXT_COLORS['reset']}"

def print_colored(text, color='white'):
    print(format_text_with_color(text, color))

def tqdm_write_colored(text, color='white'):
    tqdm.write(format_text_with_color(text, color))