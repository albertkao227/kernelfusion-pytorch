"""
Phase 2: Joint HR image and kernel estimation via reverse diffusion.

Implements Algorithm 1 from KernelFusion (arXiv:2503.21907), Section 4.2.

At each reverse diffusion timestep t:
  1. DIP U-Net refines the current estimate → produces x_t
  2. Frozen PD model denoises x_t → predicts x̂_0
  3. DIP U-Net refines x̂_0 → produces refined x̂_0
  4. SIREN INR generates the SR kernel
  5. Downscale x̂_0 with kernel → compare to LR (consistency loss)
  6. Gradient steps on DIP U-Net + SIREN INR parameters
  7. DDPM reverse step to get x_{t-1}
"""

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.image_utils import load_sem_16bit
from models.dip_unet import DIPUNet
from models.siren import SIRENKernel
from core.diffusion import (
    DDPMSchedule,
    velocity_to_x0,
    ddpm_reverse_step,
    noise_to_timestep,
)


def downscale_with_kernel(hr_image, kernel, scale_factor):
    """
    Apply the SR degradation model: LR = (HR * kernel) ↓s
    
    Args:
        hr_image: HR image [1, 1, H, W]
        kernel: SR kernel [1, 1, K, K]
        scale_factor: Downsampling factor
    
    Returns:
        lr_estimate: Estimated LR image [1, 1, H/s, W/s]
    """
    k = kernel.shape[-1]
    pad = k // 2
    
    # Convolve with kernel then subsample
    blurred = F.conv2d(hr_image, kernel, padding=pad)
    lr_estimate = blurred[:, :, ::scale_factor, ::scale_factor]
    
    return lr_estimate


def optimize_hr_and_kernel(
    lr_image_path,
    trained_pd_model,
    schedule,
    scale_factor=2,
    T_nd=200,
    n_iter_start=100,
    n_iter=20,
    lr=1e-4,
    kernel_size=13,
    device='cuda',
):
    """
    Phase 2: Joint optimization of HR image and SR kernel via reverse diffusion.
    
    Uses the frozen PD model as a patch-level prior while simultaneously
    training a DIP U-Net (for global structure) and SIREN INR (for kernel).
    
    Args:
        lr_image_path: Path to the 16-bit LR SEM image
        trained_pd_model: Frozen PD model from Phase 1
        schedule: DDPMSchedule from Phase 1 (same noise schedule)
        scale_factor: Upscaling factor (default: 2)
        T_nd: Number of noise/denoise timesteps (paper default: 200)
        n_iter_start: Optimization iterations at first timestep (paper: 100)
        n_iter: Optimization iterations per subsequent timestep (paper: 20)
        lr: Learning rate for DIP U-Net and SIREN INR (paper: 1e-4)
        kernel_size: Size of the SR kernel (paper: flexible via SIREN)
        device: Device to run on
    
    Returns:
        hr_image: Final HR image tensor [1, 1, H*s, W*s]
        kernel: Estimated SR kernel [1, 1, K, K]
    """
    # 1. Load LR image
    lr_tensor = load_sem_16bit(lr_image_path, device=device)  # [1, 1, H, W]
    _, _, lr_h, lr_w = lr_tensor.shape
    hr_h, hr_w = lr_h * scale_factor, lr_w * scale_factor
    
    # 2. Initialize: bicubic upscale of LR image
    bicubic_up = F.interpolate(
        lr_tensor, size=(hr_h, hr_w), mode='bicubic', align_corners=False
    )
    
    # 3. Noise the bicubic guess to timestep T_nd
    x_t = noise_to_timestep(bicubic_up, T_nd, schedule)
    
    # 4. Initialize DIP U-Net and SIREN INR (both trained from scratch)
    dip_unet = DIPUNet(in_channels=1, base_channels=32).to(device)
    siren_kernel = SIRENKernel(
        hidden_features=256, num_layers=5, omega=5.0, kernel_size=kernel_size
    ).to(device)
    
    # Freeze PD model
    trained_pd_model.eval()
    for param in trained_pd_model.parameters():
        param.requires_grad_(False)
    
    # Joint optimizer for DIP U-Net and SIREN INR
    optimizer = torch.optim.Adam(
        list(dip_unet.parameters()) + list(siren_kernel.parameters()),
        lr=lr,
    )
    
    # Cosine annealing across all timesteps
    total_opt_steps = n_iter_start + n_iter * (T_nd - 1)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_opt_steps, eta_min=lr * 0.5)
    
    # Previous x_0 estimate (initialized as bicubic upscale)
    x_0_prev = bicubic_up.clone()
    
    print(f"Phase 2: Reverse diffusion from t={T_nd} to t=0")
    print(f"  HR size: {hr_h}×{hr_w}, Kernel size: {kernel_size}×{kernel_size}")
    print(f"  n_iter_start={n_iter_start}, n_iter={n_iter}")
    
    # 5. Reverse diffusion loop: t = T_nd, T_nd-1, ..., 1, 0
    for t in range(T_nd, -1, -1):
        current_n_iter = n_iter_start if t == T_nd else n_iter
        
        for opt_step in range(current_n_iter):
            optimizer.zero_grad()
            
            # (a) Apply DIP U-Net to previous x_0 and construct x_t
            x_0_refined_prev = dip_unet(x_0_prev)
            
            # Reconstruct x_t from refined x_0 using the forward diffusion relation
            if t > 0:
                sqrt_ab = schedule.sqrt_alpha_bar[t]
                sqrt_1_ab = schedule.sqrt_one_minus_alpha_bar[t]
                # x_t should be consistent with the noised version of refined x_0
                noise_component = (x_t - sqrt_ab * x_0_refined_prev) / (sqrt_1_ab + 1e-8)
            
            # (b) Denoise x_t with frozen PD model → predict x̂_0
            with torch.no_grad():
                t_tensor = torch.tensor([t], device=device)
                v_pred = trained_pd_model(x_t, t_tensor)
                x_0_pd = velocity_to_x0(v_pred, x_t, t_tensor, schedule)
            
            # (c) Apply DIP U-Net to PD prediction for global coherence
            x_0_hat = dip_unet(x_0_pd)
            
            # (d) Generate kernel from SIREN INR
            kernel = siren_kernel()
            
            # (e) LR consistency loss: downscaled HR should match LR
            lr_estimate = downscale_with_kernel(x_0_hat, kernel, scale_factor)
            
            # Handle size mismatch (due to kernel padding)
            if lr_estimate.shape != lr_tensor.shape:
                lr_estimate = F.interpolate(
                    lr_estimate, size=(lr_h, lr_w), mode='bilinear', align_corners=False
                )
            
            loss = F.mse_loss(lr_estimate, lr_tensor)
            
            # (f) Backprop through DIP U-Net and SIREN INR
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        # Update x_0_prev for next iteration
        x_0_prev = x_0_hat.detach()
        
        # (g) DDPM reverse step to get x_{t-1}
        if t > 0:
            x_t = ddpm_reverse_step(x_t, x_0_hat.detach(), t, schedule)
        
        # Logging
        if t % 20 == 0 or t == T_nd:
            print(f"  t={t:>4d} | LR Consistency Loss: {loss.item():.6f}")
    
    # Final output
    hr_image = x_0_hat.detach()
    final_kernel = siren_kernel().detach()
    
    print("Phase 2 optimization complete.")
    return hr_image, final_kernel