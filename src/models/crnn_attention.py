"""Advanced model: residual SE-CNN front-end + BiGRU + multi-head attention pooling (written by the team).

CRNN: Cakir et al. 2017; squeeze-excitation: Hu et al. 2018; attention pooling: Kong et al. 2020 (PANNs).
`attention=False` replaces attention pooling with mean pooling over time (ablation).
"""
import torch
from torch import nn


class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(), nn.Linear(c // r, c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean((2, 3)))[:, :, None, None]


class ResSEBlock(nn.Module):
    def __init__(self, i, o, pool):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), SE(o))
        self.skip = nn.Sequential(nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o))
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x):
        return self.pool(torch.relu(self.conv(x) + self.skip(x)))


class AttentionPool(nn.Module):
    """Multi-head attention pooling over time: each head learns which frames matter (e.g. events above noise)."""

    def __init__(self, d, heads=4):
        super().__init__()
        self.heads, self.score, self.value, self.out = heads, nn.Linear(d, heads), nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, h):                                   # h: [B, T, d]
        a = self.score(h).softmax(dim=1)                    # [B, T, H] weights sum to 1 over time
        v = self.value(h).view(*h.shape[:2], self.heads, -1)  # [B, T, H, d/H]
        return self.out(torch.einsum("bth,bthk->bhk", a, v).flatten(1)), a


class CRNNAttention(nn.Module):
    def __init__(self, n_classes=50, attention=True, d=256):
        super().__init__()
        self.cnn = nn.Sequential(ResSEBlock(1, 32, (2, 2)), ResSEBlock(32, 64, (2, 2)),
                                 ResSEBlock(64, 128, (2, 1)), ResSEBlock(128, 128, (2, 1)))  # [B,128,8,54]
        self.proj = nn.Sequential(nn.Linear(128 * 8, d), nn.ReLU(), nn.Dropout(0.2))
        self.gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
        self.pool = AttentionPool(d) if attention else None
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, n_classes))

    def forward(self, z, return_attention=False):
        f = self.cnn(z)                                     # [B, C, F, T]
        h, _ = self.gru(self.proj(f.permute(0, 3, 1, 2).flatten(2)))  # [B, T, d]
        pooled, a = self.pool(h) if self.pool is not None else (h.mean(1), None)
        logits = self.head(pooled)
        return (logits, a) if return_attention else logits
