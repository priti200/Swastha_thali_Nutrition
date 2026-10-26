"""Colour- and texture-aware ViT - architecture only, to be trained once the dataset is ready.

The RGB input is expanded into extra channels before patch embedding:
    RGB (3, ImageNet-normalised)
    + colour: sin(hue)*sat, cos(hue)*sat, sat, value (4) - hue is circular, so it is encoded as an angle
    + texture: |Gabor| responses at n_orient orientations x len(wavelengths) scales
A timm ViT built with in_chans = 7 + n_texture consumes them; timm adapts the pretrained RGB
patch-embedding weights to the extra channels. The [CLS] feature feeds a normalised embedding
(for kNN prototype matching) and an optional classification head.
"""
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def colour_features(rgb: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """rgb in [0, 1], (B, 3, H, W) -> (B, 4, H, W)."""
    r, g, b = rgb.unbind(1)
    maxc, _ = rgb.max(1)
    minc, _ = rgb.min(1)
    delta = maxc - minc
    sat = delta / (maxc + eps)
    d = delta + eps
    hue = torch.where(maxc == r, ((g - b) / d) % 6, torch.where(maxc == g, (b - r) / d + 2, (r - g) / d + 4))
    angle = hue * (math.pi / 3)
    return torch.stack([torch.sin(angle) * sat, torch.cos(angle) * sat, sat, maxc], 1)


class GaborBank(nn.Module):
    """Fixed (non-learned) Gabor filters applied to the grayscale image."""

    def __init__(self, n_orient: int = 4, wavelengths=(4.0, 8.0), ksize: int = 15, aspect: float = 0.5):
        super().__init__()
        half = ksize // 2
        y, x = torch.meshgrid(torch.arange(-half, half + 1.0), torch.arange(-half, half + 1.0), indexing="ij")
        kernels = []
        for lam in wavelengths:
            sigma = 0.56 * lam
            for i in range(n_orient):
                theta = math.pi * i / n_orient
                xr = x * math.cos(theta) + y * math.sin(theta)
                yr = -x * math.sin(theta) + y * math.cos(theta)
                k = torch.exp(-(xr**2 + (aspect * yr) ** 2) / (2 * sigma**2)) * torch.cos(2 * math.pi * xr / lam)
                k = k - k.mean()
                kernels.append(k / k.abs().sum())
        self.register_buffer("weight", torch.stack(kernels).unsqueeze(1))

    @property
    def out_channels(self) -> int:
        return self.weight.shape[0]

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        return F.conv2d(gray, self.weight, padding=self.weight.shape[-1] // 2).abs()


class ColourTextureViT(nn.Module):
    def __init__(self, num_classes: int = 0, backbone: str = "vit_small_patch16_224", pretrained: bool = True,
                 embed_dim: int = 256, n_orient: int = 4, wavelengths=(4.0, 8.0)):
        super().__init__()
        self.gabor = GaborBank(n_orient, wavelengths)
        n_extra = 4 + self.gabor.out_channels
        self.extra_norm = nn.BatchNorm2d(n_extra)
        self.backbone = timm.create_model(backbone, pretrained=pretrained, in_chans=3 + n_extra, num_classes=0)
        width = self.backbone.num_features
        self.proj = nn.Linear(width, embed_dim)
        self.head = nn.Linear(width, num_classes) if num_classes else None
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, rgb: torch.Tensor):
        """rgb in [0, 1], (B, 3, H, W). Returns (embedding (B, embed_dim), logits or None)."""
        gray = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]).unsqueeze(1)
        extra = self.extra_norm(torch.cat([colour_features(rgb), self.gabor(gray)], 1))
        feats = self.backbone(torch.cat([(rgb - self.mean) / self.std, extra], 1))
        emb = F.normalize(self.proj(feats), dim=-1)
        return emb, self.head(feats) if self.head is not None else None
