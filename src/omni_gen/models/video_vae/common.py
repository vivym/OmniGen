import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .activations import get_activation


class CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        pad_mode: str = "constant",
        **kwargs,
    ):
        super().__init__()

        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        assert len(kernel_size) == 3, f"Kernel size must be a 3-tuple, got {kernel_size} instead."

        t_ks, h_ks, w_ks = kernel_size
        assert h_ks % 2 == 1 and w_ks % 2 == 1, f"Kernel size must be odd, got {kernel_size} instead."

        dilation: int = kwargs.get("dilation", 1)
        stride: int = kwargs.get("stride", 1)

        self.pad_mode = pad_mode

        t_pad = (t_ks - 1) * dilation + (1 - stride)
        h_pad = h_ks // 2
        w_pad = w_ks // 2

        self.temporal_padding = t_pad
        self.causal_padding = (w_pad, w_pad, h_pad, h_pad, t_pad, 0)

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(stride, 1, 1),
            dilation=(dilation, 1, 1),
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        pad_mode = self.pad_mode if self.temporal_padding < x.shape[2] else "constant"
        x = F.pad(x, self.causal_padding, mode=pad_mode)
        return self.conv(x)


class ResidualBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        non_linearity: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()

        self.output_scale_factor = output_scale_factor

        self.norm1 = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            eps=norm_eps,
            affine=True,
        )

        self.nonlinearity = get_activation(non_linearity)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=out_channels,
            eps=norm_eps,
            affine=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)

        x = self.dropout(x)
        x = self.conv2(x)

        return (x + shortcut) / self.output_scale_factor


class ResidualBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        non_linearity: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()

        self.output_scale_factor = output_scale_factor

        self.norm1 = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            eps=norm_eps,
            affine=True,
        )

        self.nonlinearity = get_activation(non_linearity)

        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)

        self.norm2 = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=out_channels,
            eps=norm_eps,
            affine=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)

        x = self.dropout(x)
        x = self.conv2(x)

        return (x + shortcut) / self.output_scale_factor


class SpatialDownsample2x(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_video = x.ndim == 5

        if is_video:
            # b, c, t, h, w -> (b t), c, h, w
            batch_size = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")

        x = self.conv(x)

        if is_video:
            # (b t), c, h, w -> b, c, t, h, w
            x = rearrange(x, "(b t) c h w -> b c t h w", b=batch_size)

        return x


class SpatialUpsample2x(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1)

        o, i, h, w = self.conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_normal_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 4) ...")
        self.conv.weight.data.copy_(conv_weight)

        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_video = x.ndim == 5

        if is_video:
            # b, c, t, h, w -> (b t), c, h, w
            batch_size = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")

        x = self.conv(x)
        x = F.silu(x)
        x = rearrange(x, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=2, p2=2)

        if is_video:
            # (b t), c, h, w -> b, c, t, h, w
            x = rearrange(x, "(b t) c h w -> b c t h w", b=batch_size)

        return x


class TemporalDownsample2x(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.causal_padding = (kernel_size - 1, 0)

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b, c, t, h, w -> (b h w), c, t
        batch_size, height = x.shape[0], x.shape[3]
        x = rearrange(x, "b c t h w -> (b h w) c t")

        x = F.pad(x, pad=self.causal_padding)
        x = self.conv(x)

        # (b h w), c, t -> b, c, t, h, w
        x = rearrange(x, "(b h w) c t -> b c t h w", b=batch_size, h=height)
        return x


class TemporalUpsample2x(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels * 2, kernel_size=1)

        o, i, t = self.conv.weight.shape
        conv_weight = torch.empty(o // 2, i, t)
        nn.init.kaiming_normal_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 2) ...")
        self.conv.weight.data.copy_(conv_weight)

        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b, c, t, h, w -> (b h w), c, t
        batch_size, height = x.shape[0], x.shape[3]
        x = rearrange(x, "b c t h w -> (b h w) c t")

        x = self.conv(x)
        x = F.silu(x)
        x = rearrange(x, "b (c p) t -> b c (t p)", p=2)

        # (b h w), c, t -> b, c, t, h, w
        x = rearrange(x, "(b h w) c t -> b c t h w", b=batch_size, h=height)
        return x


class SpatialNorm2D(nn.Module):
    """
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
    ):
        super().__init__()

        self.norm = nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
        self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f: torch.FloatTensor, zq: torch.FloatTensor) -> torch.FloatTensor:
        f_size = f.shape[-2:]
        zq = F.interpolate(zq, size=f_size, mode="nearest")
        norm_f = self.norm(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


class SpatialNorm3D(SpatialNorm2D):
    def forward(self, f: torch.FloatTensor, zq: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = f.shape[0]
        f = rearrange(f, "b c t h w -> (b t) c h w")
        zq = rearrange(zq, "b c t h w -> (b t) c h w")

        x = super().forward(f, zq)

        x = rearrange(x, "(b t) c h w -> b c t h w", b=batch_size)

        return x
