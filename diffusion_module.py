import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================
# Cosine Beta Schedule
# ==========================
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

# ==========================
# Time Embedding
# ==========================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        return self.linear(t.unsqueeze(1).float())

# ==========================
# UNet-like Diffusion Model
# ==========================
class DiffusionUNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.time_mlp = TimeEmbedding(dim)
        self.net = nn.Sequential(
            nn.Linear(dim + dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, dim)
        )

    def forward(self, x, t):
        t_embed = self.time_mlp(t)
        x_input = torch.cat([x, t_embed], dim=1)
        return self.net(x_input)

# ==========================
# Training DDPM
# ==========================
def train_diffusion_ddpm(data_vectors, timesteps=1000, epochs=1000, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = data_vectors.shape[1]
    model = DiffusionUNet(dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    betas = cosine_beta_schedule(timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    data = torch.tensor(data_vectors, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        idx = torch.randint(0, data.size(0), (batch_size,))
        x0 = data[idx]
        t = torch.randint(0, timesteps, (batch_size,), device=device)
        a = alphas_cumprod[t].unsqueeze(1)
        noise = torch.randn_like(x0)
        xt = torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise

        pred_noise = model(xt, t)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"[Diffusion-DDPM] Epoch {epoch+1}/{epochs} Loss: {loss.item():.6f}")

    return model, betas, alphas_cumprod

# ==========================
# Sampling from DDPM
# ==========================
def sample_diffusion_ddpm(model, betas, alphas_cumprod, dim=37, timesteps=1000, num_samples=10):
    device = next(model.parameters()).device
    x = torch.randn(num_samples, dim).to(device)

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((num_samples,), t, device=device)
        alpha = alphas_cumprod[t]
        beta = betas[t]

        noise_pred = model(x, t_tensor)
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(1 - beta)) * (x - ((1 - alpha).sqrt() / alpha.sqrt()) * noise_pred) + beta.sqrt() * noise

    return x.clamp(0, 1).detach().cpu().numpy()
