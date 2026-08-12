from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    b, c, h, w = x.shape
    if h % scale or w % scale:
        pad_h = (-h) % scale
        pad_w = (-w) % scale
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        h, w = x.shape[-2:]
    return x.view(b, c, h // scale, scale, w // scale, scale).permute(0, 1, 3, 5, 2, 4).reshape(
        b, c * scale * scale, h // scale, w // scale
    )


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int = 64, growth: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + growth * 2, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + growth * 3, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + growth * 4, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.act(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.act(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, channels: int = 64, growth: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth)
        self.rdb2 = ResidualDenseBlock(channels, growth)
        self.rdb3 = ResidualDenseBlock(channels, growth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """BasicSR-compatible RealESRGAN RRDBNet for native 2x inference."""

    def __init__(self, blocks: int = 23, channels: int = 64, growth: int = 32):
        super().__init__()
        self.scale = 2
        self.conv_first = nn.Conv2d(12, channels, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(channels, growth) for _ in range(blocks)])
        self.conv_body = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv_hr = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(channels, 3, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        feat = self.conv_first(pixel_unshuffle(x, 2))
        body = self.conv_body(self.body(feat)) + feat
        body = self.act(self.conv_up1(F.interpolate(body, scale_factor=2, mode="nearest")))
        body = self.act(self.conv_up2(F.interpolate(body, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.act(self.conv_hr(body)))
        return out[..., : original_size[0] * 2, : original_size[1] * 2]


class CompactRRDBNet(RRDBNet):
    """Small test fixture; production uses the 23-block network."""

    def __init__(self):
        super().__init__(blocks=2, channels=16, growth=8)


def unwrap_checkpoint(value: dict) -> dict:
    for key in ("params_ema", "params", "state_dict"):
        if key in value:
            value = value[key]
    return {key.removeprefix("module."): tensor for key, tensor in value.items()}


def load_weights(model: nn.Module, path: str | Path, strict: bool = True) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(unwrap_checkpoint(checkpoint), strict=strict)


def output_4_by_3(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    native = model(x)
    return F.interpolate(
        native,
        size=(round(x.shape[-2] * 4 / 3), round(x.shape[-1] * 4 / 3)),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0, 1)
