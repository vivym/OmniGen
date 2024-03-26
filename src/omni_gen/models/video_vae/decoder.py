import torch
import torch.nn as nn

from .activations import get_activation
from .common import CausalConv3d
from .up_blocks import get_up_block
from .mid_blocks import get_mid_block


class Decoder(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 8):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("SpatialUpBlock3D",)`):
            The types of up blocks to use. See `~omni_gen.models.video_vae.up_blocks.get_up_block` for available options.
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
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 3,
        up_block_types: tuple[str] = ("SpatialUpBlock3D",),
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
    ):
        super().__init__()

        assert len(up_block_types) == len(block_out_channels), (
            "Number of up block types must match number of block output channels."
        )
        if use_gc_blocks is not None:
            assert len(use_gc_blocks) == len(up_block_types), (
                "Number of GC blocks must match number of up block types."
            )
        else:
            use_gc_blocks = [False] * len(up_block_types)

        self.conv_in = CausalConv3d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
        )

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

        self.up_blocks = nn.ModuleList([])

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channels = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            input_channels = output_channels
            output_channels = reversed_block_out_channels[i]
            is_first_block = i == 0

            up_block = get_up_block(
                up_block_type,
                in_channels=input_channels,
                out_channels=output_channels,
                num_layers=layers_per_block + 1,    # TODO: check this
                act_fn=act_fn,
                norm_num_groups=norm_num_groups,
                norm_eps=1e-6,
                num_attention_heads=num_attention_heads,
                add_gc_block=use_gc_blocks[i],
                add_upsample=not is_first_block,
            )
            self.up_blocks.append(up_block)

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=norm_num_groups,
            eps=1e-6,
        )
        self.conv_act = get_activation(act_fn)

        self.conv_out = CausalConv3d(block_out_channels[0], out_channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.conv_in(x)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        x = self.mid_block(x)
        x = x.to(upscale_dtype)

        for up_block in self.up_blocks:
            x = up_block(x)

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x
