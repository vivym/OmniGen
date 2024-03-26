import torch
import torch.nn as nn
from einops import rearrange, repeat

from .common import CausalConv3d


class Upsampler(nn.Module):
    def __init__(
        self,
        spatial_upsample_factor: int = 1,
        temporal_upsample_factor: int = 1,
    ):
        super().__init__()

        self.spatial_upsample_factor = spatial_upsample_factor
        self.temporal_upsample_factor = temporal_upsample_factor


class SpatialUpsampler3D(Upsampler):
    def __init__(self, in_channels: int, out_channels: int | None = None):
        super().__init__(spatial_upsample_factor=2)

        if out_channels is None:
            out_channels = in_channels

        self.conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels * 4,
            kernel_size=3,
            stride=(1, 2, 2),
        )

        o, i, t, h, w = self.conv.weight.shape
        conv_weight = torch.empty(o // 4, i, t, h, w)
        nn.init.kaiming_normal_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 4) ...")
        self.conv.conv.weight.data.copy_(conv_weight)

        nn.init.zeros_(self.conv.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = rearrange(x, "b (c p1 p2) t h w -> b c t (h p1) (w p2)", p1=2, p2=2)
        return x


class TemporalUpsampler3D(Upsampler):
    def __init__(self, in_channels: int, out_channels: int | None = None):
        super().__init__(
            spatial_upsample_factor=1,
            temporal_upsample_factor=2,
        )

        if out_channels is None:
            out_channels = in_channels

        self.conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels * 2,
            kernel_size=3,
            stride=(2, 1, 1),
        )

        o, i, t, h, w = self.conv.weight.shape
        conv_weight = torch.empty(o // 2, i, t, h, w)
        nn.init.kaiming_normal_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 2) ...")
        self.conv.conv.weight.data.copy_(conv_weight)

        nn.init.zeros_(self.conv.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = rearrange(x, "b (c p1) t h w -> b c (t p1) h w", p1=2)
        return x


class SpatialTemporalUpsampler3D(Upsampler):
    def __init__(self, in_channels: int, out_channels: int | None = None):
        super().__init__(
            spatial_upsample_factor=2,
            temporal_upsample_factor=2,
        )

        if out_channels is None:
            out_channels = in_channels

        self.conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels * 8,
            kernel_size=3,
            stride=(2, 2, 2),
        )

        o, i, t, h, w = self.conv.weight.shape
        conv_weight = torch.empty(o // 8, i, t, h, w)
        nn.init.kaiming_normal_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 8) ...")
        self.conv.conv.weight.data.copy_(conv_weight)

        nn.init.zeros_(self.conv.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = rearrange(x, "b (c p1 p2 p3) t h w -> b c (t p1) (h p2) (w p3)", p1=2, p2=2, p3=2)
        return x
