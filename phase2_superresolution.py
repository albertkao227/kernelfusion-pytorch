import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from diffusers import UNet2DModel
from models.dip_unet import DIPUNet
from models.siren import SIRENKernel
import math
import matplotlib.pyplot as plt
from tqdm import tqdm  
import numpy as np # ADDED: Required for safe 16-bit image loading

def get_ddpm_schedule(num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
    betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def apply_degradation(x, kernel, target_size):
    """Blurs the image with the SIREN kernel and downsamples it to an exact size."""
    padding = kernel.shape[-1] // 2
    
    # Apply depthwise convolution (blurring)
    blurred = F.conv2d(x, kernel, padding=padding, groups=x.shape[1])
    
    # Downsample using exact target dimensions
    downsampled = F.interpolate(blurred, size=target_size, mode='bicubic', antialias=False)
    return downsampled


def run_phase_2(
    lr_image_path, 
    hr_output_image_path,
    kernel_plot_path,
    pd_checkpoint, 
    scale_factor=2, 
    num_timesteps=1000, # FIX: Changed default to 1000 to match training
    opt_steps_per_t=3, 
    device='cuda',
):
    print(f"Starting Phase 2: SR and Kernel Estimation for {lr_image_path}")
    
    # ---------------------------------------------------------
    # 1. LOAD CHECKPOINT FIRST TO DETECT CHANNELS
    # ---------------------------------------------------------
    print("--> Inspecting pre-trained Patch-Diffusion checkpoint...")
    checkpoint = torch.load(pd_checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    ckpt_channels = state_dict['conv_in.weight'].shape[1]
    print(f"--> Checkpoint was trained on {ckpt_channels}-channel images.")

    # ---------------------------------------------------------
    # 2. LOAD LR IMAGE (FIXED: Safely handling 16-bit TIFFs)
    # ---------------------------------------------------------
    print("--> Loading input image using NumPy to preserve bit-depth...")
    img_pil = Image.open(lr_image_path)
    img_np = np.array(img_pil)
    
    # Align channels with checkpoint requirements
    if ckpt_channels == 1:
        if img_np.ndim == 3:
            img_np = np.mean(img_np, axis=-1)
        # Add channel dimension to make it [H, W, 1]
        img_np = np.expand_dims(img_np, axis=-1)
    else:
        # If it needs 3 channels but is grayscale
        if img_np.ndim == 2:
            img_np = np.stack([img_np]*3, axis=-1)
        elif img_np.shape[-1] == 4: # Drop Alpha channel if RGBA
            img_np = img_np[:, :, :3]
            
    # Normalize based on exact bit depth
    if img_pil.mode.startswith('I') or img_np.dtype == np.uint16 or img_np.dtype == np.int32:
        img_np = img_np.astype(np.float32) / 65535.0
    else:
        img_np = img_np.astype(np.float32) / 255.0
        
    # Convert [H, W, C] to [1, C, H, W]
    lr_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    lr_tensor = lr_tensor * 2.0 - 1.0  # Scale to [-1, 1]
    
    _, C, lr_H, lr_W = lr_tensor.shape
    
    # Force HR dimensions to be multiples of 16 for the U-Net skip connections
    hr_H = int(round((lr_H * scale_factor) / 16.0) * 16)
    hr_W = int(round((lr_W * scale_factor) / 16.0) * 16)
    hr_H, hr_W = max(16, hr_H), max(16, hr_W)
    
    print(f"--> Input LR size: {lr_H}x{lr_W}")
    print(f"--> Target HR size: {hr_H}x{hr_W} (Adjusted to multiple of 16)")

    # ---------------------------------------------------------
    # 3. INITIALIZE MODELS WITH DETECTED CHANNELS
    # ---------------------------------------------------------
    pd_model = UNet2DModel(
        sample_size=64, 
        in_channels=C,  
        out_channels=C, 
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)
    
    pd_model.load_state_dict(state_dict)
    pd_model.eval()
    for param in pd_model.parameters():
        param.requires_grad = False
        
    print("--> Initializing DIP-UNet and SIREN Kernel...")
    dip_net = DIPUNet(in_channels=C, out_channels=C).to(device)
    siren_net = SIRENKernel(kernel_size=13, channels=C).to(device)
    
    optimizer = torch.optim.Adam([
        {'params': dip_net.parameters(), 'lr': 1e-4},
        {'params': siren_net.parameters(), 'lr': 1e-4}
    ])
    
    criterion = torch.nn.L1Loss()
    
    # Setup DDPM Math
    betas, alphas, alphas_cumprod = get_ddpm_schedule(num_timesteps, device=device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    # ---------------------------------------------------------
    # 4. REVERSE DIFFUSION PROCESS
    # ---------------------------------------------------------
    x_t = torch.randn(1, C, hr_H, hr_W, device=device)
    fixed_dip_input = torch.randn(1, C, hr_H, hr_W, device=device)
    
    print("\nBeginning reverse diffusion process...")
    pbar = tqdm(reversed(range(num_timesteps)), total=num_timesteps, desc="Sampling & Optimizing")
    
    for t_idx in pbar:
        t_tensor = torch.tensor([t_idx], device=device).long()
        
        # --- A. Get Patch-Diffusion prediction (x0_hat) ---
        with torch.no_grad():
            v_pred = pd_model(x_t, t_tensor).sample
            x0_hat = sqrt_alphas_cumprod[t_idx] * x_t - sqrt_one_minus_alphas_cumprod[t_idx] * v_pred
            x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
            
        # --- B. Optimize DIP & SIREN using Data Consistency ---
        for opt_step in range(opt_steps_per_t):
            optimizer.zero_grad()
            
            x0_dip = dip_net(fixed_dip_input)
            loss_structure = F.mse_loss(x0_dip, x0_hat)
            
            k_siren = siren_net()
            
            lr_hat = apply_degradation(x0_dip, k_siren, target_size=(lr_H, lr_W))
            loss_data = criterion(lr_hat, lr_tensor)

            total_loss = loss_data + 0.1 * loss_structure
            total_loss.backward()
            optimizer.step()

        # --- C. Posterior Sampling (Step backwards to t-1) ---
        with torch.no_grad():
            x0_refined = dip_net(fixed_dip_input)
            
            if t_idx > 0:
                alpha_t = alphas[t_idx]
                alpha_bar_t = alphas_cumprod[t_idx]
                alpha_bar_t_prev = alphas_cumprod[t_idx - 1]
                beta_t = betas[t_idx]
                
                mu = (math.sqrt(alpha_bar_t_prev) * beta_t / (1.0 - alpha_bar_t)) * x0_refined + \
                     (math.sqrt(alpha_t) * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)) * x_t
                     
                variance = beta_t * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)
                noise = torch.randn_like(x_t)
                x_t = mu + math.sqrt(variance) * noise
            else:
                x_t = x0_refined 
                
        pbar.set_postfix({
            "Data Loss": f"{loss_data.item():.4f}", 
            "Struct Loss": f"{loss_structure.item():.4f}"
        })

    # ---------------------------------------------------------
    # 5. SAVE OUTPUTS
    # ---------------------------------------------------------
    print("\nProcess complete. Saving outputs...")
    
    # FIX: Safe squeeze to preserve single-channel dimension
    x_out = (x_t.squeeze(0).cpu() + 1.0) / 2.0
    x_out = torch.clamp(x_out, 0.0, 1.0)
    out_img = transforms.ToPILImage()(x_out)
    
    out_img.save(hr_output_image_path)
    print(f"--> Saved High-Resolution Image to: {hr_output_image_path}")
    
    kernel_np = siren_net.get_kernel_numpy()
    
    if kernel_np.ndim == 3:
        kernel_2d = kernel_np[0]
    else:
        kernel_2d = kernel_np
        
    plt.figure(figsize=(6, 5)) 
    plt.imshow(kernel_2d, cmap='viridis')
    plt.colorbar(label="Kernel Weights")
    plt.title("SIREN-Estimated Degradation Kernel")
    plt.axis('off')
    
    
    plt.savefig(kernel_plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"--> Saved Kernel Plot to: {kernel_plot_path}")
    
    return out_img, kernel_np


if __name__ == "__main__":
    run_phase_2(
        lr_image_path="data/images/lena_gray16.tiff",
        hr_output_image_path="phase2_hr_output_lena_gray.png",
        kernel_plot_path="phase2_estimated_kernel_lena_gray.png",
        pd_checkpoint="pd_checkpoints_diffuser/pd_model_step_200000.pth", 
        scale_factor=2
    )