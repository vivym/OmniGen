import math

import torch
import torch.nn as nn

from .downsamplers import BlurPooling2D, BlurPooling3D


class DiscriminatorBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
    ):
        super().__init__()

        self.output_scale_factor = output_scale_factor

        self.norm1 = nn.BatchNorm2d(in_channels)

        self.nonlinearity = nn.LeakyReLU(0.2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        if add_downsample:
            self.downsampler = BlurPooling2D(out_channels, out_channels)
        else:
            self.downsampler = nn.Identity()

        self.norm2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if add_downsample:
            self.shortcut = nn.Sequential(
                BlurPooling2D(in_channels, in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.shortcut = nn.Identity()

        self.spatial_downsample_factor = 2
        self.temporal_downsample_factor = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)

        x = self.dropout(x)
        x = self.downsampler(x)
        x = self.conv2(x)

        return (x + shortcut) / self.output_scale_factor


class Discriminator2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        block_out_channels: tuple[int] = (64,),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        output_channels = block_out_channels[0]
        for i, out_channels in enumerate(block_out_channels):
            input_channels = output_channels
            output_channels = out_channels
            is_final_block = i == len(block_out_channels) - 1

            self.blocks.append(
                DiscriminatorBlock2D(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    output_scale_factor=math.sqrt(2),
                    add_downsample=not is_final_block,
                )
            )

        self.conv_norm_out = nn.BatchNorm2d(block_out_channels[-1])
        self.conv_act = nn.LeakyReLU(0.2)

        self.conv_out = nn.Conv2d(block_out_channels[-1], 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.conv_in(x)

        for block in self.blocks:
            x = block(x)

        x = self.conv_out(x)

        return x


class DiscriminatorBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
    ):
        super().__init__()

        self.output_scale_factor = output_scale_factor

        self.norm1 = nn.BatchNorm3d(in_channels)

        self.nonlinearity = nn.LeakyReLU(0.2)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        if add_downsample:
            self.downsampler = BlurPooling3D(out_channels, out_channels)
        else:
            self.downsampler = nn.Identity()

        self.norm2 = nn.BatchNorm3d(out_channels)

        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        if add_downsample:
            self.shortcut = nn.Sequential(
                BlurPooling3D(in_channels, in_channels),
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.shortcut = nn.Identity()

        self.spatial_downsample_factor = 2
        self.temporal_downsample_factor = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)

        x = self.dropout(x)
        x = self.downsampler(x)
        x = self.conv2(x)

        return (x + shortcut) / self.output_scale_factor


class Discriminator3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        block_out_channels: tuple[int] = (64,),
    ):
        super().__init__()

        self.conv_in = nn.Conv3d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        output_channels = block_out_channels[0]
        for i, out_channels in enumerate(block_out_channels):
            input_channels = output_channels
            output_channels = out_channels
            is_final_block = i == len(block_out_channels) - 1

            self.blocks.append(
                DiscriminatorBlock3D(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    output_scale_factor=math.sqrt(2),
                    add_downsample=not is_final_block,
                )
            )

        self.conv_norm_out = nn.BatchNorm3d(block_out_channels[-1])
        self.conv_act = nn.LeakyReLU(0.2)

        self.conv_out = nn.Conv3d(block_out_channels[-1], 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.conv_in(x)

        for block in self.blocks:
            x = block(x)

        x = self.conv_out(x)

        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        block_out_channels: tuple[int] = (64,),
        use_2d: bool = True,
        use_3d: bool = False,
    ):
        super().__init__()

        self.use_2d = use_2d
        self.use_3d = use_3d

        if use_2d:
            self.discriminator_2d = Discriminator2D(in_channels, block_out_channels)

        if use_3d:
            self.discriminator_3d = Discriminator3D(in_channels, block_out_channels)

        self.apply(weight_init)

    def forward(
        self,
        x_2d: torch.Tensor | None = None,
        x_3d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if x_2d is not None:
            assert self.use_2d
            logits_2d = self.discriminator_2d(x_2d)
        else:
            logits_2d = None

        if x_3d is not None:
            assert self.use_3d
            logits_3d = self.discriminator_3d(x_3d)
        else:
            logits_3d = None

        return logits_2d, logits_3d


def weight_init(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
