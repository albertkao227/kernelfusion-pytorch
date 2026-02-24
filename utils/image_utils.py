"""
Image utilities for 16-bit grayscale SEM images.

Provides loading, saving, and patch extraction functions
adapted for the full 16-bit dynamic range (0–65535).
"""

import cv2
import torch
import numpy as np


def load_sem_16bit(image_path, device='cpu', percentile_norm=False):
    """
    Load a 16-bit grayscale image and normalize to [-1, 1] for diffusion.
    
    Args:
        image_path: Path to the 16-bit grayscale image (PNG or TIFF)
        device: Target device
        percentile_norm: If True, use 1st-99th percentile normalization
                        (recommended for SEM images with sparse histograms)
    
    Returns:
        tensor: Normalized image tensor [1, 1, H, W] in [-1, 1]
    """
    img_16bit = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_16bit is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    
    # Handle multi-channel images (convert to grayscale)
    if len(img_16bit.shape) == 3:
        img_16bit = cv2.cvtColor(img_16bit, cv2.COLOR_BGR2GRAY)
    
    img_float = img_16bit.astype(np.float32)
    
    if percentile_norm:
        # Percentile-based normalization for SEM images
        p_low = np.percentile(img_float, 1)
        p_high = np.percentile(img_float, 99)
        if p_high - p_low < 1.0:
            p_high = p_low + 1.0  # Avoid division by zero
        img_normalized = 2.0 * (img_float - p_low) / (p_high - p_low) - 1.0
    else:
        # Standard normalization: 0-65535 → [-1, 1]
        img_normalized = (img_float / 32767.5) - 1.0
    
    tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


def save_sem_16bit(tensor, save_path):
    """
    Convert a [-1, 1] tensor back to a 16-bit grayscale image and save.
    
    Args:
        tensor: Image tensor in [-1, 1] (any shape, will be squeezed)
        save_path: Output path (.tif or .png recommended for 16-bit)
    """
    tensor = tensor.detach().cpu().squeeze()
    img_normalized = (tensor.numpy() + 1.0) * 32767.5
    img_16bit = np.clip(img_normalized, 0, 65535).astype(np.uint16)
    cv2.imwrite(save_path, img_16bit)


def save_kernel(kernel_tensor, save_path):
    """
    Save the estimated kernel as a normalized 8-bit image for visualization.
    
    Args:
        kernel_tensor: Kernel tensor [1, 1, K, K] or [K, K]
        save_path: Output path (.png)
    """
    kernel = kernel_tensor.detach().cpu().squeeze().numpy()
    # Normalize to 0-255 for visualization
    kernel_vis = kernel - kernel.min()
    if kernel_vis.max() > 0:
        kernel_vis = (kernel_vis / kernel_vis.max() * 255).astype(np.uint8)
    else:
        kernel_vis = np.zeros_like(kernel, dtype=np.uint8)
    # Upscale for visibility
    scale = max(1, 256 // kernel_vis.shape[0])
    kernel_vis = cv2.resize(kernel_vis, None, fx=scale, fy=scale, 
                            interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(save_path, kernel_vis)


def extract_random_patches(tensor, patch_size=64, num_patches=16):
    """
    Randomly crops patches from a tensor (used in Phase 2 for score distillation).
    
    Args:
        tensor: Image tensor [B, C, H, W]
        patch_size: Size of patches to crop
        num_patches: Number of patches to extract
    
    Returns:
        patches: Stacked patches [num_patches, C, patch_size, patch_size]
    """
    H, W = tensor.shape[2], tensor.shape[3]
    patches = []
    for _ in range(num_patches):
        y = np.random.randint(0, max(1, H - patch_size))
        x = np.random.randint(0, max(1, W - patch_size))
        patches.append(tensor[:, :, y:y+patch_size, x:x+patch_size])
    return torch.cat(patches, dim=0)