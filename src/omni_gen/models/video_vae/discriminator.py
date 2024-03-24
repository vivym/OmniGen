import torch
import torch.nn as nn

from .attention import SpatialAttention
from .common import ResidualBlock2D, SpatialDownsample2x
from .gc_block import GlobalContextBlock


class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        add_attention: bool = False,
        add_gc_block: bool = False,
        add_downsample: bool = True,
    ):
        super().__init__()

        self.convs = nn.ModuleList([])
        self.attentions = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.convs.append(
                ResidualBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    non_linearity=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                )
            )

            if add_attention:
                self.attentions.append(
                    SpatialAttention(
                        out_channels,
                        nheads=out_channels // attention_head_dim,
                        head_dim=attention_head_dim,
                        bias=True,
                        upcast_softmax=True,
                        eps=norm_eps,
                        rescale_output_factor=output_scale_factor,
                        residual_connection=True,
                    )
                )
            else:
                self.attentions.append(None)

        if add_gc_block:
            self.gc_block = GlobalContextBlock(out_channels, out_channels, fusion_type="mul")
        else:
            self.gc_block = None

        if add_downsample:
            self.downsampler = SpatialDownsample2x(out_channels, out_channels, kernel_size=3)
            self.spatial_downsample_factor = 2
        else:
            self.downsampler = None
            self.spatial_downsample_factor = 1

        self.temporal_downsample_factor = 1

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for conv, attn in zip(self.convs, self.attentions):
            x = conv(x)
            if attn is not None:
                x = attn(x)

        if self.gc_block is not None:
            x = self.gc_block(x)

        if self.downsampler is not None:
            x = self.downsampler(x)

        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        block_out_channels: tuple[int] = (64,),
        use_gc_blocks: tuple[bool] | None = None,
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        num_attention_heads: int = 1,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        output_channels = block_out_channels[0]
        for i, out_channels in enumerate(block_out_channels):
            input_channels = output_channels
            output_channels = out_channels
            is_final_block = i == len(block_out_channels) - 1   # TODO: whether it is necessary

            self.blocks.append(
                DiscriminatorBlock(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    num_layers=layers_per_block,
                    act_fn=act_fn,
                    norm_num_groups=norm_num_groups,
                    norm_eps=1e-6,
                    attention_head_dim=out_channels // num_attention_heads,
                    add_attention=is_final_block,
                    add_gc_block=use_gc_blocks[i] if use_gc_blocks is not None else False,
                    add_downsample=not is_final_block,
                )
            )

        self.conv_out = nn.Conv2d(block_out_channels[-1], 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.conv_in(x)

        for block in self.blocks:
            x = block(x)

        x = self.conv_out(x)

        return x
