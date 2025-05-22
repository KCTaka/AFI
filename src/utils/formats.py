def format_input(x, expected_input_dim = 3, batched=True):
    """Formats the input tensor to the expected dimensions.

    Args:
        x (torch.Tensor): The input tensor to format.
        expected_input_dim (int, optional): The expected number of dimensions (excluding batch). Defaults to 3.
        batched (bool, optional): Whether the input is batched. Defaults to True.
    """
    
    if batched:
        if x.dim() == expected_input_dim + 1:
            return x
        elif x.dim() == expected_input_dim:
            return x.unsqueeze(0)
        else:
            raise ValueError(f"Expected input tensor to have {expected_input_dim} or {expected_input_dim + 1} dimensions, but got {x.dim()}")
        
    else:
        if x.dim() == expected_input_dim:
            return x
        elif x.dim() == expected_input_dim + 1:
            x_squeezed = x.squeeze(0)
            if x_squeezed.dim() == expected_input_dim:
                return x_squeezed
            else:
                raise ValueError(f"Likely got batched input, but expected {expected_input_dim} dimensions, got {x.dim()}")
            
    raise ValueError(f"Expected input tensor to have {expected_input_dim} or {expected_input_dim + 1} dimensions, but got {x.dim()}")


if __name__ == "__main__":
    import torch
    # Test the function
    x = torch.randn(4, 3, 64, 64)  # Example input tensor
    formatted_x = format_input(x, expected_input_dim=3, batched=True)
    print(f"Formatted tensor shape: {formatted_x.shape}")
    # Test with a non-batched tensor
    x_non_batched = torch.randn(1, 3, 64, 64)  # Example non-batched input tensor
    formatted_x_non_batched = format_input(x_non_batched, expected_input_dim=3, batched=False)
    print(f"Formatted non-batched tensor shape: {formatted_x_non_batched.shape}")