"""
DDPM noise schedule, forward/reverse diffusion, and velocity prediction utilities.

Implements the standard Denoising Diffusion Probabilistic Model (Ho et al., 2020)
with velocity prediction parameterization (Salimans & Ho, 2022) as used in the
KernelFusion paper (arXiv:2503.21907).
"""

import torch
import torch.nn.functional as F
import math


class DDPMSchedule:
    """
    Standard DDPM linear beta schedule with precomputed coefficients.
    
    The paper uses T=1000 timesteps with a linear schedule.
    """
    
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.T = T
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        
        # Precompute useful quantities
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)  # ᾱ_t
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)     # √ᾱ_t
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)  # √(1−ᾱ_t)
        self.sqrt_alphas = torch.sqrt(self.alphas)            # √α_t
        
        # For reverse step
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
    
    def to(self, device):
        """Move all tensors to the specified device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        self.sqrt_alphas = self.sqrt_alphas.to(device)
        self.alpha_bar_prev = self.alpha_bar_prev.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self


def forward_diffusion(x_0, t, schedule):
    """
    Forward diffusion process: q(x_t | x_0).
    
    x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε
    
    Args:
        x_0: Clean image tensor [B, C, H, W]
        t: Timestep indices [B] (integer, 0-indexed)
        schedule: DDPMSchedule instance
    
    Returns:
        x_t: Noised image
        epsilon: The noise that was added
    """
    epsilon = torch.randn_like(x_0)
    
    sqrt_alpha_bar_t = schedule.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
    
    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * epsilon
    
    return x_t, epsilon


def compute_velocity_target(x_0, epsilon, t, schedule):
    """
    Compute the velocity prediction target (Salimans & Ho, 2022).
    
    v_t = √ᾱ_t · ε − √(1−ᾱ_t) · x_0
    
    Args:
        x_0: Clean image tensor [B, C, H, W]
        epsilon: Noise tensor [B, C, H, W]
        t: Timestep indices [B]
        schedule: DDPMSchedule instance
    
    Returns:
        v_t: Velocity target
    """
    sqrt_alpha_bar_t = schedule.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
    
    v_t = sqrt_alpha_bar_t * epsilon - sqrt_one_minus_alpha_bar_t * x_0
    return v_t


def velocity_to_x0(v_t, x_t, t, schedule):
    """
    Recover x_0 from the predicted velocity and x_t.
    
    x̂_0 = √ᾱ_t · x_t − √(1−ᾱ_t) · v_t
    
    Args:
        v_t: Predicted velocity [B, C, H, W]
        x_t: Noised image [B, C, H, W]
        t: Timestep indices [B]
        schedule: DDPMSchedule instance
    
    Returns:
        x_0_pred: Predicted clean image
    """
    sqrt_alpha_bar_t = schedule.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
    
    x_0_pred = sqrt_alpha_bar_t * x_t - sqrt_one_minus_alpha_bar_t * v_t
    return x_0_pred


def velocity_to_epsilon(v_t, x_t, t, schedule):
    """
    Recover ε from the predicted velocity and x_t.
    
    ε̂ = √(1−ᾱ_t) · x_t + √ᾱ_t · v_t
    
    Args:
        v_t: Predicted velocity [B, C, H, W]
        x_t: Noised image [B, C, H, W]
        t: Timestep indices [B]
        schedule: DDPMSchedule instance
    
    Returns:
        epsilon_pred: Predicted noise
    """
    sqrt_alpha_bar_t = schedule.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
    
    epsilon_pred = sqrt_one_minus_alpha_bar_t * x_t + sqrt_alpha_bar_t * v_t
    return epsilon_pred


def ddpm_reverse_step(x_t, x_0_pred, t, schedule):
    """
    DDPM reverse sampling step: p(x_{t-1} | x_t).
    
    Uses the predicted x_0 to compute x_{t-1}.
    
    Args:
        x_t: Current noised image [B, C, H, W]
        x_0_pred: Predicted clean image [B, C, H, W]
        t: Current timestep (integer scalar, not batched)
        schedule: DDPMSchedule instance
    
    Returns:
        x_t_minus_1: Denoised one step
    """
    alpha_bar_t = schedule.alpha_bar[t]
    alpha_bar_prev = schedule.alpha_bar_prev[t]
    beta_t = schedule.betas[t]
    
    # Posterior mean
    coef1 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
    coef2 = (1.0 - alpha_bar_prev) * torch.sqrt(schedule.alphas[t]) / (1.0 - alpha_bar_t)
    
    mean = coef1 * x_0_pred + coef2 * x_t
    
    if t > 0:
        noise = torch.randn_like(x_t)
        variance = schedule.posterior_variance[t]
        x_t_minus_1 = mean + torch.sqrt(variance) * noise
    else:
        x_t_minus_1 = mean
    
    return x_t_minus_1


def noise_to_timestep(x_clean, t_start, schedule):
    """
    Noise a clean image to a specific timestep t_start.
    Used to initialize Phase 2 from bicubic upscale.
    
    Args:
        x_clean: Clean (bicubic upscaled) image [1, 1, H, W]
        t_start: Target timestep
        schedule: DDPMSchedule instance
    
    Returns:
        x_t: Noised image at timestep t_start
    """
    epsilon = torch.randn_like(x_clean)
    sqrt_alpha_bar_t = schedule.sqrt_alpha_bar[t_start]
    sqrt_one_minus_alpha_bar_t = schedule.sqrt_one_minus_alpha_bar[t_start]
    
    x_t = sqrt_alpha_bar_t * x_clean + sqrt_one_minus_alpha_bar_t * epsilon
    return x_t


def get_sinusoidal_embedding(t, dim=128):
    """
    Sinusoidal timestep embedding (Vaswani et al., 2017).
    
    Args:
        t: Timestep tensor [B] or scalar
        dim: Embedding dimension
    
    Returns:
        embedding: [B, dim] sinusoidal embedding
    """
    if isinstance(t, int) or (isinstance(t, torch.Tensor) and t.dim() == 0):
        t = torch.tensor([t], dtype=torch.float32)
    
    t = t.float()
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
    emb = t.unsqueeze(-1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb
