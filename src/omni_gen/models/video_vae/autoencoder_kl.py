from dataclasses import dataclass

import torch
import torch.nn as nn

from omni_gen.utils.distributions import DiagonalGaussianDistribution
from .decoder import Decoder
from .encoder import Encoder


@dataclass
class EncoderOutput:
    latent_dist: DiagonalGaussianDistribution


@dataclass
class DecoderOutput:
    sample: torch.Tensor


class AutoencoderKL(nn.Module):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input video.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("SpatialDownBlock3D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("SpatialUpBlock3D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        use_gc_blocks (`Tuple[bool, ...]`, *optional*, defaults to `None`):
            Tuple of booleans indicating whether to use global context block in the corresponding down/up block.
        mid_block_type: (`str`, *optional*, defaults to `"MidBlock3D"`): Type of the middle block.
        mid_block_use_attention (`bool`, *optional*, defaults to `True`):
            Whether to use attention in the middle block.
        mid_block_attention_type (`str`, *optional*, defaults to `"3d"`):
            Type of attention to use in the middle block.
        mid_block_num_attention_heads (`int`, *optional*, defaults to 1):
            Number of attention heads to use in the middle block.
        layers_per_block (`int`, *optional*, defaults to 1): Number of layers in each block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        num_attention_heads (`int`, *optional*, defaults to 1):
            Number of attention heads to use in the encoder and decoder blocks.
        latent_channels (`int`, *optional*, defaults to 8): Number of channels in the latent space.
        norm_num_groups (`int`, *optional*, defaults to 32):
            Number of groups to use for group normalization in the encoder and decoder blocks.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str, ...] = ("SpatialDownBlock3D",),
        up_block_types: tuple[str, ...] = ("SpatialUpBlock3D",),
        block_out_channels: tuple[int, ...] = (64,),
        use_gc_blocks: tuple[bool, ...] | None = None,
        mid_block_type: str = "MidBlock3D",
        mid_block_use_attention: bool = False,
        mid_block_attention_type: str = "3d",
        mid_block_num_attention_heads: int = 1,
        layers_per_block: int = 1,
        act_fn: str = "silu",
        num_attention_heads: int = 1,
        latent_channels: int = 8,
        norm_num_groups: int = 32,
        scaling_factor: float = 0.18215,
        image_mode: bool = False,
        tile_sample_size: tuple[int, ...] = (17, 256, 256),
        tile_overlap_factor: float = 0.25,
    ):
        super().__init__()

        self.scaling_factor = scaling_factor

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            use_gc_blocks=use_gc_blocks,
            mid_block_type=mid_block_type,
            mid_block_use_attention=mid_block_use_attention,
            mid_block_attention_type=mid_block_attention_type,
            mid_block_num_attention_heads=mid_block_num_attention_heads,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            num_attention_heads=num_attention_heads,
            double_z=True,
            image_mode=image_mode,
        )

        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            use_gc_blocks=use_gc_blocks,
            mid_block_type=mid_block_type,
            mid_block_use_attention=mid_block_use_attention,
            mid_block_attention_type=mid_block_attention_type,
            mid_block_num_attention_heads=mid_block_num_attention_heads,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            num_attention_heads=num_attention_heads,
            image_mode=image_mode,
        )

        if image_mode:
            self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
            self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        else:
            self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
            self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)

        self.use_tiling = False
        self.use_slicing = False

        assert len(tile_sample_size) == 3, tile_sample_size
        self.tile_sample_size = tile_sample_size
        self.tile_latent_size = (
            (tile_sample_size[0] - 1) // self.encoder.temporal_downsample_factor + 1,
            tile_sample_size[1] // self.encoder.spatial_downsample_factor,
            tile_sample_size[2] // self.encoder.spatial_downsample_factor,
        )
        self.tile_overlap_factor = tile_overlap_factor

    def enable_tiling(self, use_tiling: bool = True):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.use_tiling = use_tiling

    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.enable_tiling(False)

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def encode(self, x: torch.Tensor) -> EncoderOutput:
        if (
            self.use_tiling and
            (x.shape[-1] > self.tile_sample_size[-1] or x.shape[-2] > self.tile_sample_size[-2])
        ):
            return self._tiled_encode(x)

        h = self.encoder(x)

        moments: torch.Tensor = self.quant_conv(h)
        mean, logvar = moments.chunk(2, dim=1)
        posterior = DiagonalGaussianDistribution(mean, logvar)

        return EncoderOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor) -> DecoderOutput:
        if (
            self.use_tiling and
            (z.shape[-1] > self.tile_latent_size[-1] or z.shape[-2] > self.tile_latent_size[-2])
        ):
            return self._tiled_decode(z)

        z = self.post_quant_conv(z)

        decoded = self.decoder(z)

        return DecoderOutput(sample=decoded)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        posteriors = self.encode(x).latent_dist

        z = posteriors.sample()

        rec_samples = self.decode(z).sample

        return posteriors, rec_samples

    def _blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[..., y, :] = a[..., -blend_extent + y, :] * (1 - y / blend_extent) + b[..., y, :] * (y / blend_extent)
        return b

    def _blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[..., :, x] = a[..., :, -blend_extent + x] * (1 - x / blend_extent) + b[..., :, x] * (x / blend_extent)
        return b

    def _tiled_encode(self, x: torch.Tensor) -> EncoderOutput:
        overlap_size = (
            int(self.tile_sample_size[-2] * (1 - self.tile_overlap_factor)),
            int(self.tile_sample_size[-1] * (1 - self.tile_overlap_factor)),
        )
        blend_extent = (
            int(self.tile_latent_size[-2] * self.tile_overlap_factor),
            int(self.tile_latent_size[-1] * self.tile_overlap_factor),
        )
        row_limit = (
            self.tile_latent_size[-2] - blend_extent[-2],
            self.tile_latent_size[-1] - blend_extent[-1],
        )

        # Split the image into 256x256 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[-2], overlap_size[-2]):
            row = []
            for j in range(0, x.shape[-1], overlap_size[-1]):
                tile = x[..., i:i + self.tile_sample_size[-2], j:j + self.tile_sample_size[-1]]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self._blend_v(rows[i - 1][j], tile, blend_extent[-2])
                if j > 0:
                    tile = self._blend_h(row[j - 1], tile, blend_extent[-1])
                result_row.append(tile[..., :row_limit[-2], :row_limit[-1]])
            result_rows.append(torch.cat(result_row, dim=-1))

        moments = torch.cat(result_rows, dim=-2)
        mean, logvar = moments.chunk(2, dim=1)
        posterior = DiagonalGaussianDistribution(mean, logvar)

        return EncoderOutput(latent_dist=posterior)

    def _tiled_decode(self, z: torch.Tensor) -> DecoderOutput:
        overlap_size = (
            int(self.tile_latent_size[-2] * (1 - self.tile_overlap_factor)),
            int(self.tile_latent_size[-1] * (1 - self.tile_overlap_factor)),
        )
        blend_extent = (
            int(self.tile_sample_size[-2] * self.tile_overlap_factor),
            int(self.tile_sample_size[-1] * self.tile_overlap_factor),
        )
        row_limit = (
            self.tile_sample_size[-2] - blend_extent[-2],
            self.tile_sample_size[-1] - blend_extent[-1],
        )

        # Split z into overlapping 32x32 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[-2], overlap_size[-2]):
            row = []
            for j in range(0, z.shape[-1], overlap_size[-1]):
                tile = z[..., i:i + self.tile_latent_size[-2], j:j + self.tile_latent_size[-1]]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self._blend_v(rows[i - 1][j], tile, blend_extent[-2])
                if j > 0:
                    tile = self._blend_h(row[j - 1], tile, blend_extent[-1])
                result_row.append(tile[..., :row_limit[-2], :row_limit[-1]])
            result_rows.append(torch.cat(result_row, dim=-1))

        decoded = torch.cat(result_rows, dim=-2)

        return DecoderOutput(sample=decoded)
