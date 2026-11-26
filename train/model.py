"""Residual U-Net model and physical-unit autoregressive rollout helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)


class MonsoonResUNet(nn.Module):
    """Predict a normalized daily delta using weather plus two static maps and season."""

    def __init__(self, in_channels: int = 9, out_channels: int = 6, base_ch: int = 32) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)
        self.enc1 = ResidualBlock(base_ch, base_ch)
        self.enc2 = ResidualBlock(base_ch, base_ch * 2)
        self.bottleneck = ResidualBlock(base_ch * 2, base_ch * 4)
        self.dec2 = ResidualBlock(base_ch * 6, base_ch * 2)
        self.dec1 = ResidualBlock(base_ch * 3, base_ch)
        self.out_head = nn.Conv2d(base_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dynamic = x[:, :6]
        e1 = self.enc1(F.silu(self.in_proj(x)))
        e2 = self.enc2(F.avg_pool2d(e1, kernel_size=2, ceil_mode=True))
        z = self.bottleneck(F.avg_pool2d(e2, kernel_size=2, ceil_mode=True))
        z = F.interpolate(z, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        z = self.dec2(torch.cat([z, e2], dim=1))
        z = F.interpolate(z, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        z = self.dec1(torch.cat([z, e1], dim=1))
        return dynamic + self.out_head(z)


def normalize_static_maps(static_maps: np.ndarray | torch.Tensor, device=None) -> torch.Tensor:
    """Keep land/sea mask as supplied and scale elevation metres to roughly [-1, 1]."""
    static = torch.as_tensor(static_maps, dtype=torch.float32, device=device)
    if static.ndim == 4:
        if static.shape[0] != 1:
            raise ValueError("static maps must be [2,H,W] or [1,2,H,W]")
        static = static[0]
    if static.ndim != 3 or static.shape[0] != 2:
        raise ValueError("static maps must have shape [2,H,W]")
    static = static.clone()
    static[1] = static[1] / 3000.0
    return static


def _stats_tensors(stats: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=device).view(1, 6, 1, 1)
    std = torch.as_tensor(stats["std"], dtype=torch.float32, device=device).view(1, 6, 1, 1)
    if torch.any(std <= 0):
        raise ValueError("all channel standard deviations must be positive")
    return mean, std


@torch.no_grad()
def rollout_10_days(
    model: MonsoonResUNet,
    x0_physical: np.ndarray | torch.Tensor,
    static_maps: np.ndarray | torch.Tensor,
    stats: dict[str, Any],
    doy: float | np.ndarray | torch.Tensor,
    steps: int = 10,
) -> torch.Tensor:
    """Return [steps,6,H,W] or [B,steps,6,H,W] values in physical units.

    ``doy`` is the day of year for the last observed input day. Forecast step 1
    uses the following day, matching the API contract.
    """
    if not 1 <= int(steps) <= 10:
        raise ValueError("steps must be between 1 and 10")
    device = next(model.parameters()).device
    x = torch.as_tensor(x0_physical, dtype=torch.float32, device=device)
    single = x.ndim == 3
    if single:
        x = x.unsqueeze(0)
    if x.ndim != 4 or x.shape[1] != 6:
        raise ValueError("x0 must have shape [6,H,W] or [B,6,H,W]")

    batch, _, height, width = x.shape
    mean, std = _stats_tensors(stats, device)
    static = normalize_static_maps(static_maps, device=device)
    if static.shape[-2:] != (height, width):
        raise ValueError("static map grid must match the weather grid")
    static = static.unsqueeze(0).expand(batch, 2, height, width)
    doy_t = torch.as_tensor(doy, dtype=torch.float32, device=device).reshape(-1)
    if doy_t.numel() == 1 and batch > 1:
        doy_t = doy_t.expand(batch)
    if doy_t.numel() != batch:
        raise ValueError("doy must be scalar or have one value per batch item")
    doy_t = doy_t.view(batch, 1, 1, 1)

    current = (x - mean) / std
    outputs = []
    for step in range(int(steps)):
        forecast_doy = doy_t + step + 1
        seasonal = torch.sin(2.0 * math.pi * forecast_doy / 365.25).expand(batch, 1, height, width)
        model_input = torch.cat([current, static, seasonal], dim=1)
        current = model(model_input)
        physical = current * std + mean
        physical = physical.clone()
        physical[:, 1] = torch.clamp(physical[:, 1], min=0.0)
        current = (physical - mean) / std
        outputs.append(physical)

    result = torch.stack(outputs, dim=1)
    return result[0] if single else result
