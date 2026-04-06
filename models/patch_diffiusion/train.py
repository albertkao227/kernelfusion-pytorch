import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import math
import random
from diffusers import UNet2DModel 

# -------------------------------------------------------------
# 1. Single Image Patch Dataset
# -------------------------------------------------------------
class SingleImagePatchDataset(Dataset):
    def __init__(self, image_path, patch_size=64, length=600000):
        super().__init__()
        self.image = Image.open(image_path).convert('RGB')
        self.patch_size = patch_size
        self.length = length
        self.transform = transforms.ToTensor()
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        w, h = self.image.size
        
        if w < self.patch_size or h < self.patch_size:
            self.image = self.image.resize((max(w, self.patch_size), max(h, self.patch_size)))
            w, h = self.image.size
            
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        patch = self.image.crop((x, y, x + self.patch_size, y + self.patch_size))
        
        if random.random() > 0.5:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            patch = patch.transpose(Image.FLIP_TOP_BOTTOM)
            
        tensor_patch = self.transform(patch) * 2.0 - 1.0 
        return tensor_patch

# -------------------------------------------------------------
# 2. Diffusion Noise Schedule
# -------------------------------------------------------------
def get_linear_schedule(num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

# -------------------------------------------------------------
# 3. Simple UNet Architecture
# -------------------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PatchUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, base_channels * 4),
            nn.GELU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )
        
        # Downsampling
        self.inc = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down1 = nn.Conv2d(base_channels, base_channels*2, 4, stride=2, padding=1)
        self.down2 = nn.Conv2d(base_channels*2, base_channels*4, 4, stride=2, padding=1)
        
        # Time Embeddings
        self.time_emb1 = nn.Linear(base_channels*4, base_channels*2)
        self.time_emb2 = nn.Linear(base_channels*4, base_channels*4)
        
        # Upsampling 1
        self.up1 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1)
        # After concatenating with skip connection (128 + 128 = 256), we convolve back to 128
        self.upconv1 = nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1)
        
        # Upsampling 2
        self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1)
        # After concatenating with skip connection (64 + 64 = 128), we convolve back to 64
        self.upconv2 = nn.Conv2d(base_channels*2, base_channels, 3, padding=1)
        
        self.outc = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        # 1. Downward Pass
        x1 = F.relu(self.inc(x)) # Shape: [B, 64, H, W]
        
        x2 = F.relu(self.down1(x1))
        x2 = x2 + self.time_emb1(t_emb)[:, :, None, None] # Shape: [B, 128, H/2, W/2]
        
        x3 = F.relu(self.down2(x2))
        x3 = x3 + self.time_emb2(t_emb)[:, :, None, None] # Shape: [B, 256, H/4, W/4]
        
        # 2. Upward Pass with Skip Connections
        x = F.relu(self.up1(x3)) # Outputs 128 channels
        x = torch.cat([x, x2], dim=1) # Skip connection: 128 + 128 = 256 channels
        x = F.relu(self.upconv1(x)) # Reduces back to 128 channels
        
        x = F.relu(self.up2(x)) # Outputs 64 channels
        x = torch.cat([x, x1], dim=1) # Skip connection: 64 + 64 = 128 channels
        x = F.relu(self.upconv2(x)) # Reduces back to 64 channels
        
        return self.outc(x)


    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        x1 = F.relu(self.inc(x))
        x2 = F.relu(self.down1(x1))
        
        x2 = x2 + self.time_emb1(t_emb)[:, :, None, None]
        
        x3 = F.relu(self.down2(x2))
        x3 = x3 + self.time_emb2(t_emb)[:, :, None, None]
        
        x = F.relu(self.up1(x3))
        x = F.relu(self.up2(x))
        
        return self.outc(x)

# -------------------------------------------------------------
# 4. Phase 1 Training Loop with Checkpointing
# -------------------------------------------------------------
def train_phase_1(image_path, batch_size=16, total_steps=600000, save_interval=50000, save_dir="checkpoints", device='cuda'):
    print(f"Starting Phase 1: Patch-Diffusion Training on {image_path}")
    os.makedirs(save_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # NEW: Production-Ready U-Net with Attention & ResBlocks
    # ---------------------------------------------------------
    model = UNet2DModel(
        sample_size=64,           # Target image resolution
        in_channels=3,            # RGB input
        out_channels=3,           # RGB output (velocity prediction)
        layers_per_block=2,       # Number of ResNet blocks per layer
        block_out_channels=(64, 128, 256, 512), # Feature map channels
        down_block_types=(
            "DownBlock2D",        # Standard ResNet downsampling
            "DownBlock2D",
            "AttnDownBlock2D",    # Self-Attention block!
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",      # Self-Attention block!
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
    
    dataset = SingleImagePatchDataset(image_path, patch_size=64, length=total_steps * batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    num_timesteps = 1000
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = get_linear_schedule(num_timesteps)
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
    
    model.train()
    step = 0
    
    for batch in dataloader:
        if step >= total_steps:
            break
            
        x_0 = batch.to(device)
        batch_size_cur = x_0.shape[0]
        
        t = torch.randint(0, num_timesteps, (batch_size_cur,), device=device).long()
        noise = torch.randn_like(x_0)
        
        sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
        v_target = sqrt_alpha_t * noise - sqrt_one_minus_alpha_t * x_0
        
        # ---------------------------------------------------------
        # NEW: Diffusers UNet returns an object, we want .sample
        # ---------------------------------------------------------
        v_pred = model(x_t, t).sample 
        
        loss = F.mse_loss(v_pred, v_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 5000 == 0:
            print(f"Step: {step}/{total_steps} | v-prediction Loss: {loss.item():.4f}")
            
        if step > 0 and step % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"pd_model_step_{step}.pth")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            
        step += 1
        
    return model


if __name__ == "__main__":
    # Example usage: Save a checkpoint every 50,000 steps up to 600,000
    train_phase_1("data/images/lena_color.tiff", batch_size=16, steps=200000, save_interval=20000, save_dir="pd_checkpoints_diffuser")
    pass