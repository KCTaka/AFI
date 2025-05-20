import torch

def convert_to_target_visible_channels(latent_image: torch.Tensor, target_channels: int = 3, resize=None, mode='nearest') -> torch.Tensor:
    """
    Converts a batch of images to a target number of channels, suitable for viewing (typically 3).

    The behavior is as follows:
    - If C == target_channels: Returns the image tensor as is.
    - If C < target_channels:
        - If C == 1 (e.g., grayscale): Repeats the single channel `target_channels` times.
        - If C > 1 and C < target_channels (e.g., C=2, target_channels=3): Pads with zero channels to reach `target_channels`.
        - If C == 0: Raises a ValueError.
    - If C > target_channels: Applies Principal Component Analysis (PCA) to reduce the
      channel dimensionality from C to `target_channels`. PCA is computed on the
      channel data across all pixels in the batch. It attempts to use `torch.pca_lowrank`
      for efficiency and falls back to an SVD-based implementation if needed.

    Args:
        image_tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
                                     It's assumed to be on the desired device (e.g., GPU for efficiency).
        target_channels (int): The desired number of output channels. Defaults to 3.

    Returns:
        torch.Tensor: Output tensor of shape (B, target_channels, H, W) on the same device as input.

    Raises:
        ValueError: If the input tensor has 0 channels or if target_channels is not positive.
        RuntimeError: If PCA computation fails via both primary and fallback methods.
    """
    if not isinstance(latent_image, torch.Tensor):
        raise TypeError("Input image_tensor must be a PyTorch tensor.")
    if latent_image.ndim != 4:
        raise ValueError(f"Input image_tensor must be 4-dimensional (B, C, H, W), but got {latent_image.ndim} dimensions.")

    B, C, H, W = latent_image.shape
    device = latent_image.device
    dtype = latent_image.dtype

    if not isinstance(target_channels, int) or target_channels <= 0:
        raise ValueError("target_channels must be a positive integer.")

    if C == target_channels:
        # print(f"Input images already have {target_channels} channels. No conversion needed.")
        latent_image = torch.nn.functional.interpolate(latent_image, size=resize, mode=mode, align_corners=False) if resize is not None else latent_image
        return latent_image
    elif C == 0:
        raise ValueError("Input tensor has 0 channels.")
    elif C < target_channels:
        # print(f"Input images have {C} channels, target is {target_channels}.")
        if C == 1:
            # print(f"Repeating single channel to create {target_channels} channels.")
            latent_image = torch.nn.functional.interpolate(latent_image, size=resize, mode=mode, align_corners=False) if resize is not None else latent_image
            return latent_image.repeat(1, target_channels, 1, 1)
        else: # C > 1 and C < target_channels (e.g., C=2, target_channels=3 when C_original=2)
            # print(f"Padding with {target_channels - C} zero channels.")
            num_channels_to_add = target_channels - C
            padding_channels = torch.zeros(B, num_channels_to_add, H, W, device=device, dtype=dtype)
            latent_image = torch.cat((latent_image, padding_channels), dim=1)
            latent_image = torch.nn.functional.interpolate(latent_image, size=resize, mode=mode, align_corners=False) if resize is not None else latent_image
            return latent_image
    else: # C > target_channels
        # print(f"Input images have {C} channels. Applying PCA to reduce to {target_channels} channels.")

        # Reshape for PCA: (B, C, H, W) -> (B*H*W, C_original)
        # Each pixel's channel vector is treated as a sample point in a C_original-dimensional space.
        data_permuted = latent_image.permute(0, 2, 3, 1)  # Shape: (B, H, W, C_original)
        # data_reshaped has shape (N_total_pixels, C_original)
        data_reshaped = data_permuted.reshape(-1, C)

        # Perform PCA
        try:
            # `torch.pca_lowrank` is generally efficient. `center=True` handles mean subtraction internally.
            # `q` is the number of principal components to keep (i.e., target_channels).
            # Input A: (n_samples, n_features) -> (N_total_pixels, C_original)
            # Returns U (n_samples, q), S (q,), V (n_features, q)
            # Transformed data (scores) = U @ diag(S)
            U_pca, S_pca, _V_components = torch.pca_lowrank(data_reshaped, q=target_channels, center=True)
            
            transformed_data = U_pca @ torch.diag(S_pca) # Shape: (N_total_pixels, target_channels)
            # print("PCA computed using torch.pca_lowrank.")

        except Exception as e_pca:
            # print(f"torch.pca_lowrank failed ('{str(e_pca)}'). Falling back to SVD method for PCA.")
            
            # Manual SVD-based PCA:
            # 1. Center the data (X_centered = X - mean(X))
            mean = torch.mean(data_reshaped, dim=0, keepdim=True)
            data_centered = data_reshaped - mean  # Shape: (N_total_pixels, C_original)

            # 2. Perform SVD: X_centered = U_svd @ diag(S_svd) @ Vh_svd
            #    U_svd: (N_total_pixels, k_svd)
            #    S_svd: (k_svd,)
            #    Vh_svd: (k_svd, C_original), where k_svd = min(N_total_pixels, C_original)
            #    The columns of V (where V = Vh_svd.adjoint()) are the principal components.
            try:
                _U_svd, _S_svd, Vh_svd = torch.linalg.svd(data_centered, full_matrices=False)
            except Exception as e_svd:
                # Chain the exceptions for better debugging info
                raise RuntimeError(f"SVD computation failed during PCA fallback after pca_lowrank failed with '{str(e_pca)}'. SVD error: '{str(e_svd)}'") from e_svd

            # Principal components (eigenvectors of X_centered.T @ X_centered)
            # V_components_svd has shape (C_original, target_channels)
            V_components_svd = Vh_svd.adjoint()[:, :target_channels]
            
            # 3. Project centered data onto these principal components:
            #    Transformed_data = X_centered @ V_components_svd
            transformed_data = torch.matmul(data_centered, V_components_svd)
            # print("PCA computed using SVD fallback.")

        # Reshape transformed data back to image format (B, target_channels, H, W)
        output_images_permuted = transformed_data.reshape(B, H, W, target_channels) # (B, H, W, target_channels)
        output_images = output_images_permuted.permute(0, 3, 1, 2) # (B, target_channels, H, W)
        
        # Resize if needed
        if resize is not None:
            # Assuming resize is a tuple (new_height, new_width)
            output_images = torch.nn.functional.interpolate(output_images, size=resize, mode=mode, align_corners=False)

        return output_images
    
    
if __name__ == "__main__":
        # Example usage
        # --- Test cases ---
    # Assuming operations are on CPU for this example, add .cuda() for GPU
    B, H, W = 2, 32, 32 # Small batch for testing

    # Case 1: C > target_channels (e.g., 128 -> 3)
    latent_images_128 = torch.randn(B, 128, H, W)
    visible_128_to_3 = convert_to_target_visible_channels(latent_images_128, target_channels=3, resize=(128, 128))
    print(f"Original shape: {latent_images_128.shape}, Converted shape: {visible_128_to_3.shape}")
    # Expected: Original shape: torch.Size([2, 128, 32, 32]), Converted shape: torch.Size([2, 3, 32, 32])

    # Case 2: C == target_channels (e.g., 3 -> 3)
    latent_images_3 = torch.randn(B, 3, H, W)
    visible_3_to_3 = convert_to_target_visible_channels(latent_images_3, target_channels=3)
    print(f"Original shape: {latent_images_3.shape}, Converted shape: {visible_3_to_3.shape}")
    # Expected: Original shape: torch.Size([2, 3, 32, 32]), Converted shape: torch.Size([2, 3, 32, 32])

    # Case 3: C == 1 (e.g., 1 -> 3, grayscale to RGB-like)
    latent_images_1 = torch.randn(B, 1, H, W)
    visible_1_to_3 = convert_to_target_visible_channels(latent_images_1, target_channels=3)
    print(f"Original shape: {latent_images_1.shape}, Converted shape: {visible_1_to_3.shape}")
    # Check if channels are repeated:
    # print(torch.allclose(visible_1_to_3[:, 0, :, :], visible_1_to_3[:, 1, :, :]))
    # print(torch.allclose(visible_1_to_3[:, 0, :, :], visible_1_to_3[:, 2, :, :]))
    # Expected: Original shape: torch.Size([2, 1, 32, 32]), Converted shape: torch.Size([2, 3, 32, 32])
    # True, True

    # Case 4: 1 < C < target_channels (e.g., 2 -> 3)
    latent_images_2 = torch.randn(B, 2, H, W)
    visible_2_to_3 = convert_to_target_visible_channels(latent_images_2, target_channels=3, resize=(128, 128))
    print(f"Original shape: {latent_images_2.shape}, Converted shape: {visible_2_to_3.shape}")
    # Check if last channel is zeros:
    # print(torch.all(visible_2_to_3[:, 2, :, :] == 0))
    # Expected: Original shape: torch.Size([2, 2, 32, 32]), Converted shape: torch.Size([2, 3, 32, 32])
    # True

    # Case 5: C > target_channels (e.g., 5 -> 2, custom target)
    latent_images_5 = torch.randn(B, 5, H, W)
    visible_5_to_2 = convert_to_target_visible_channels(latent_images_5, target_channels=2)
    print(f"Original shape: {latent_images_5.shape}, Converted shape: {visible_5_to_2.shape}")
    # Expected: Original shape: torch.Size([2, 5, 32, 32]), Converted shape: torch.Size([2, 2, 32, 32])

    # Case 6: C == 0 (Error)
    try:
        latent_images_0 = torch.randn(B, 0, H, W)
        convert_to_target_visible_channels(latent_images_0)
    except ValueError as e:
        print(f"Handled C=0 error: {e}")
    # Expected: Handled C=0 error: Input tensor has 0 channels.