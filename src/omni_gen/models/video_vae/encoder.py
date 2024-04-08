import torch
import torch.nn as nn

from .activations import get_activation
from .common import CausalConv3d
from .down_blocks import get_down_block
from .mid_blocks import get_mid_block


class Encoder(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 8):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("SpatialDownBlock3D",)`):
            The types of down blocks to use. See `~omni_gen.models.video_vae.down_blocks.get_down_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        use_gc_blocks (`Tuple[bool, ...]`, *optional*, defaults to `None`):
            Whether to use global context blocks for each down block.
        mid_block_type (`str`, *optional*, defaults to `"MidBlock3D"`):
            The type of mid block to use. See `~omni_gen.models.video_vae.mid_blocks.get_mid_block` for available options.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        num_attention_heads (`int`, *optional*, defaults to 1):
            The number of attention heads to use.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 8,
        down_block_types: tuple[str] = ("SpatialDownBlock3D",),
        block_out_channels: tuple[int] = (64,),
        use_gc_blocks: tuple[bool] | None = None,
        mid_block_type: str = "MidBlock3D",
        mid_block_use_attention: bool = True,
        mid_block_attention_type: str = "3d",
        mid_block_num_attention_heads: int = 1,
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        num_attention_heads: int = 1,
        double_z: bool = True,
        image_mode: bool = False,
    ):
        super().__init__()

        assert len(down_block_types) == len(block_out_channels), (
            "Number of down block types must match number of block output channels."
        )
        if use_gc_blocks is not None:
            assert len(use_gc_blocks) == len(down_block_types), (
                "Number of GC blocks must match number of down block types."
            )
        else:
            use_gc_blocks = [False] * len(down_block_types)

        if image_mode:
            self.conv_in = nn.Conv2d(
                in_channels,
                block_out_channels[0],
                kernel_size=3,
                padding=1,
            )
        else:
            self.conv_in = CausalConv3d(
                in_channels,
                block_out_channels[0],
                kernel_size=3,
            )

        self.down_blocks = nn.ModuleList([])

        spatial_downsample_factor = 1
        temporal_downsample_factor = 1

        output_channels = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channels = output_channels
            output_channels = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                in_channels=input_channels,
                out_channels=output_channels,
                num_layers=layers_per_block,
                act_fn=act_fn,
                norm_num_groups=norm_num_groups,
                norm_eps=1e-6,
                num_attention_heads=num_attention_heads,
                add_gc_block=use_gc_blocks[i],
                add_downsample=not is_final_block,
            )
            spatial_downsample_factor *= down_block.spatial_downsample_factor
            temporal_downsample_factor *= down_block.temporal_downsample_factor
            self.down_blocks.append(down_block)

        self.mid_block = get_mid_block(
            mid_block_type,
            in_channels=block_out_channels[-1],
            num_layers=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=1e-6,
            add_attention=mid_block_use_attention,
            attention_type=mid_block_attention_type,
            num_attention_heads=mid_block_num_attention_heads,
        )

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1],
            num_groups=norm_num_groups,
            eps=1e-6,
        )
        self.conv_act = get_activation(act_fn)

        conv_out_channels = 2 * out_channels if double_z else out_channels
        if image_mode:
            self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, kernel_size=3, padding=1)
        else:
            self.conv_out = CausalConv3d(block_out_channels[-1], conv_out_channels, kernel_size=3)

        self.spatial_downsample_factor = spatial_downsample_factor
        self.temporal_downsample_factor = temporal_downsample_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.conv_in(x)

        for down_block in self.down_blocks:
            x = down_block(x)

        x = self.mid_block(x)

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x
