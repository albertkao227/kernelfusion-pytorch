import os
import torch
import math
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from diffusers import UNet2DModel

# -------------------------------------------------------------
# 1. Reverse Diffusion Math (DDPM for v-prediction)
# -------------------------------------------------------------
def get_ddpm_schedule(num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    return betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

@torch.no_grad()
def p_sample(model, x, t_index, betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device):
    """Executes one step of the reverse diffusion process using v-prediction."""
    t = torch.tensor([t_index] * x.shape[0], device=device).long()
    
    sqrt_alpha_bar_t = sqrt_alphas_cumprod[t_index]
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alphas_cumprod[t_index]

    # Predict velocity using diffusers UNet
    v_pred = model(x, t).sample

    # 1. Reconstruct x_0 from v_pred
    # Math: v = sqrt(alpha_bar)*noise - sqrt(1-alpha_bar)*x_0  
    # Therefore: x_0 = sqrt(alpha_bar)*x_t - sqrt(1-alpha_bar)*v
    pred_x0 = sqrt_alpha_bar_t * x - sqrt_one_minus_alpha_bar_t * v_pred
    
    # Clip to [-1, 1] to prevent artifacting
    pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

    if t_index == 0:
        return pred_x0

    # 2. Compute mean (mu) and variance for the previous step (Standard DDPM posterior)
    alpha_t = alphas[t_index]
    alpha_bar_t = alphas_cumprod[t_index]
    alpha_bar_t_prev = alphas_cumprod[t_index - 1]
    beta_t = betas[t_index]
    
    # DDPM Mean formulation
    mu = (math.sqrt(alpha_bar_t_prev) * beta_t / (1.0 - alpha_bar_t)) * pred_x0 + \
         (math.sqrt(alpha_t) * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)) * x

    # DDPM Variance
    posterior_variance = beta_t * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)
    sigma = math.sqrt(posterior_variance)
    
    # Add noise back in
    noise = torch.randn_like(x)
    return mu + sigma * noise

@torch.no_grad()
def generate_patches(model, num_patches=16, patch_size=64, num_timesteps=1000, device='cuda'):
    """Generates patches from pure noise."""
    model.eval()
    x = torch.randn(num_patches, 3, patch_size, patch_size, device=device)
    
    betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = get_ddpm_schedule(num_timesteps)
    
    # Move schedule to device
    betas = betas.to(device)
    alphas = alphas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)

    print("Generating patches...")
    for t_index in reversed(range(num_timesteps)):
        x = p_sample(model, x, t_index, betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
        if t_index % 100 == 0:
            print(f"Sampling step: {t_index}/{num_timesteps}")
            
    # Un-normalize from [-1, 1] back to [0, 1] for visualization
    x = (x + 1.0) / 2.0 
    return x

# -------------------------------------------------------------
# 2. Extract Real Patches for Comparison
# -------------------------------------------------------------
def get_real_patches(image_path, num_patches=16, patch_size=64):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    w, h = image.size
    
    patches = []
    for _ in range(num_patches):
        x = random.randint(0, max(0, w - patch_size))
        y = random.randint(0, max(0, h - patch_size))
        patch = image.crop((x, y, x + patch_size, y + patch_size))
        patches.append(transform(patch))
        
    return torch.stack(patches)

# -------------------------------------------------------------
# 3. Main Evaluation Execution
# -------------------------------------------------------------
def evaluate_checkpoint(checkpoint_path, original_image_path, output_image_path="eval_result.png", device='cuda'):
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Initialize the EXACT SAME UNet2DModel used in your training script
    model = UNet2DModel(
        sample_size=64,           
        in_channels=3,            
        out_channels=3,           
        layers_per_block=2,       
        block_out_channels=(64, 128, 256, 512), 
        down_block_types=(
            "DownBlock2D",        
            "DownBlock2D",
            "AttnDownBlock2D",    
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",      
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)
    
    # Load the saved state dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle the custom dictionary format from your script
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model at step {checkpoint.get('step', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # Generate 16 new patches from pure noise
    generated_patches = generate_patches(model, num_patches=16, device=device)
    
    # Extract 16 real patches from the original image
    real_patches = get_real_patches(original_image_path, num_patches=16)
    
    # Create an image grid
    gen_grid = make_grid(generated_patches.cpu(), nrow=4, padding=2)
    real_grid = make_grid(real_patches.cpu(), nrow=4, padding=2)
    
    # Plot and save
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Matplotlib expects channel-last format: [H, W, C]
    axes[0].imshow(real_grid.permute(1, 2, 0).numpy())
    axes[0].set_title("Real Image Patches")
    axes[0].axis('off')
    
    axes[1].imshow(gen_grid.permute(1, 2, 0).numpy())
    axes[1].set_title("Generated Patches (From Checkpoint)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_image_path)
    print(f"Saved evaluation comparison to: {output_image_path}")

if __name__ == "__main__":
    # Point this to one of your generated checkpoints
    checkpoint_file = "pd_checkpoints_diffuser/pd_model_step_180000.pth" 
    original_image = "data/images/lena_color.tiff"
    
    # Ensure the file exists before evaluating
    if os.path.exists(checkpoint_file):
        evaluate_checkpoint(
            checkpoint_path=checkpoint_file, 
            original_image_path=original_image,
            output_image_path="evaluation_diffuser_200k.png"
        )
    else:
        print(f"Checkpoint not found at {checkpoint_file}. Please check the path.")