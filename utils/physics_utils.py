"""
Physics-based utilities for SR kernel handling.

With the SIREN INR approach, explicit kernel constraints (clamp, normalize)
are handled intrinsically by the SIREN architecture (leaky sigmoid output +
sum-to-1 normalization). This module provides helper utilities for
kernel operations.
"""

import torch
import torch.nn.functional as F


def apply_kernel_and_downsample(hr_image, kernel, scale_factor):
    """
    Apply the SR degradation model: LR = (HR * kernel) ↓s
    
    This is the core physics equation (Eq. 1 in the paper):
    I_LR = (I_HR * k_s) ↓s
    
    Args:
        hr_image: HR image [B, 1, H, W]
        kernel: SR kernel [1, 1, K, K] (should be normalized)
        scale_factor: Subsampling factor s
    
    Returns:
        lr_estimate: Estimated LR [B, 1, H/s, W/s]
    """
    k = kernel.shape[-1]
    pad = k // 2
    blurred = F.conv2d(hr_image, kernel, padding=pad)
    lr_estimate = blurred[:, :, ::scale_factor, ::scale_factor]
    return lr_estimate


def kernel_symmetry_loss(kernel):
    """
    Optional regularization: encourage kernel symmetry.
    Useful for SEM images where the degradation is often symmetric.
    
    Args:
        kernel: Kernel tensor [1, 1, K, K]
    
    Returns:
        symmetry_loss: Scalar loss
    """
    k = kernel.squeeze()
    # Horizontal symmetry
    h_sym = F.mse_loss(k, k.flip(0))
    # Vertical symmetry
    v_sym = F.mse_loss(k, k.flip(1))
    return (h_sym + v_sym) / 2.0