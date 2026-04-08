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

def get_ddpm_schedule(num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
    betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod

def apply_degradation(x, kernel, scale_factor):
    """Blurs the image with the SIREN kernel and downsamples it."""
    padding = kernel.shape[-1] // 2
    
    # Apply depthwise convolution (blurring)
    blurred = F.conv2d(x, kernel, padding=padding, groups=x.shape[1])
    
    # Downsample
    # Note: Antialiasing=False mimics standard SR degradation models
    downsampled = F.interpolate(blurred, scale_factor=1.0/scale_factor, mode='bicubic', antialias=False)
    return downsampled

def run_phase_2(
    lr_image_path, 
    pd_checkpoint, 
    scale_factor=2, 
    num_timesteps=1000, 
    opt_steps_per_t=3, 
    device='cuda'
):
    print(f"Starting Phase 2: SR and Kernel Estimation for {lr_image_path}")
    
    # 1. Load LR image
    lr_img_pil = Image.open(lr_image_path).convert('RGB')
    lr_tensor = transforms.ToTensor()(lr_img_pil).unsqueeze(0).to(device)
    lr_tensor = lr_tensor * 2.0 - 1.0  # Scale to [-1, 1]
    
    _, C, lr_H, lr_W = lr_tensor.shape
    hr_H, hr_W = lr_H * scale_factor, lr_W * scale_factor
    
    print(f"--> Input LR size: {lr_H}x{lr_W}")
    print(f"--> Target HR size: {hr_H}x{hr_W}")
    
    # 2. Load Frozen PD Model (Phase 1)
    print("--> Loading pre-trained Patch-Diffusion model...")
    pd_model = UNet2DModel(
        sample_size=64, 
        in_channels=3, out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)
    
    checkpoint = torch.load(pd_checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    pd_model.load_state_dict(state_dict)
    pd_model.eval()
    for param in pd_model.parameters():
        param.requires_grad = False
        
    # 3. Initialize DIP and SIREN (Trainable)
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
    
    # 4. Initialize x_T (Pure Noise)
    x_t = torch.randn(1, C, hr_H, hr_W, device=device)
    fixed_dip_input = torch.randn(1, C, hr_H, hr_W, device=device)
    
    print("\nBeginning reverse diffusion process...")
    # Wrap the loop in tqdm for a dynamic progress bar
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
            lr_hat = apply_degradation(x0_dip, k_siren, scale_factor)
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
                
        # Update progress bar messaging with current losses
        pbar.set_postfix({
            "Data Loss": f"{loss_data.item():.4f}", 
            "Struct Loss": f"{loss_structure.item():.4f}"
        })

    # 5. Save Final HR Output
    print("\nProcess complete. Saving outputs...")
    x_out = (x_t.squeeze().cpu() + 1.0) / 2.0
    x_out = torch.clamp(x_out, 0.0, 1.0)
    out_img = transforms.ToPILImage()(x_out)
    out_img.save("phase2_hr_output.png")
    print("--> Saved High-Resolution Image to: phase2_hr_output.png")
    
    # 6. Plot and Save Estimated Kernel
    kernel_np = siren_net.get_kernel_numpy()
    
    # Handle dimension squeezing (if kernel_np is [C, K, K], take the first channel)
    if kernel_np.ndim == 3:
        kernel_2d = kernel_np[0]
    else:
        kernel_2d = kernel_np
        
    plt.figure(figsize=(6, 5))
    plt.imshow(kernel_2d, cmap='viridis')
    plt.colorbar(label="Kernel Weights")
    plt.title("SIREN-Estimated Degradation Kernel")
    plt.axis('off')
    
    kernel_plot_path = "phase2_estimated_kernel.png"
    plt.savefig(kernel_plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"--> Saved Kernel Plot to: {kernel_plot_path}")
    
    return out_img, kernel_np

if __name__ == "__main__":
    run_phase_2(
        lr_image_path="data/images/lena_color.tiff", 
        pd_checkpoint="pd_checkpoints_diffuser/pd_model_step_180000.pth", 
        scale_factor=2
    )