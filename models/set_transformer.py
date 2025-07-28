import torch
import torch.nn as nn
import torch.nn.functional as F

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim_out, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim_out)
        self.ff = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out)
        )

    def forward(self, x):
        h, _ = self.mha(x, x, x)
        x = self.ln(h + x)
        x = self.ln(self.ff(x) + x)
        return x

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds=1):
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(num_seeds, dim))
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        batch_size = x.size(0)
        seed = self.seed_vectors.unsqueeze(0).repeat(batch_size, 1, 1)
        h, _ = self.mha(seed, x, x)
        return self.ln(h + seed)

class SetTransformer(nn.Module):
    def __init__(self, input_dim=1, dim_hidden=128, num_heads=4, output_dim=37):
        super().__init__()
        self.embedding = nn.Embedding(38, dim_hidden)  # 1〜37を埋め込み
        self.encoder = nn.Sequential(
            SAB(dim_hidden, dim_hidden, num_heads),
            SAB(dim_hidden, dim_hidden, num_heads)
        )
        self.decoder = nn.Sequential(
            PMA(dim_hidden, num_heads),
            nn.Linear(dim_hidden, output_dim)
        )

    def forward(self, x):
        x = self.embedding(x)  # [B, 7, dim]
        x = self.encoder(x)
        x = self.decoder(x).squeeze(1)  # [B, 37]
        return torch.sigmoid(x)  # 各番号の出現確率
