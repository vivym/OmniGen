import itertools
from dataclasses import dataclass

import torch
import torch.nn as nn

from omni_gen.utils.distributions import DiagonalGaussianDistribution
from omni_gen.utils.accelerate import apply_forward_hook
from .decoder import Decoder
from .encoder import Encoder
from .vae_loss import VAELoss


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
        with_loss (`bool`, *optional*, defaults to `False`):
            Whether to compute the loss during forward pass. If `True`, the forward pass returns a dictionary with
            the loss and the output. If `False`, the forward pass returns the output.
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
        with_loss: bool = False,
        lpips_model_name_or_path: str = "vivym/lpips",
        init_logvar: float = 0.0,
        reconstruction_loss_type: str = "l1",
        reconstruction_loss_weight: float = 1.0,
        perceptual_loss_weight: float = 1.0,
        nll_loss_weight: float = 1.0,
        kl_loss_weight: float = 1.0,
        discriminator_loss_weight: float = 0.5,
        disc_block_out_channels: tuple[int] = (64,),
    ):
        super().__init__()

        self.scaling_factor = scaling_factor
        self.with_loss = with_loss

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
        )

        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)

        if with_loss:
            self.loss = VAELoss(
                lpips_model_name_or_path=lpips_model_name_or_path,
                init_logvar=init_logvar,
                reconstruction_loss_type=reconstruction_loss_type,
                reconstruction_loss_weight=reconstruction_loss_weight,
                perceptual_loss_weight=perceptual_loss_weight,
                nll_loss_weight=nll_loss_weight,
                kl_loss_weight=kl_loss_weight,
                discriminator_loss_weight=discriminator_loss_weight,
                disc_in_channels=in_channels,
                disc_block_out_channels=disc_block_out_channels,
            )

    @apply_forward_hook
    def encode(self, x: torch.Tensor) -> EncoderOutput:
        h = self.encoder(x)

        moments: torch.Tensor = self.quant_conv(h)
        mean, logvar = moments.chunk(2, dim=1)
        posterior = DiagonalGaussianDistribution(mean, logvar)

        return EncoderOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(self, z: torch.Tensor) -> DecoderOutput:
        z = self.post_quant_conv(z)

        decoded = self.decoder(z)

        return DecoderOutput(sample=decoded)

    def parameters_without_loss(self):
        return itertools.chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
            self.quant_conv.parameters(),
            self.post_quant_conv.parameters(),
        )

    @apply_forward_hook
    def training_step(
        self,
        samples: torch.Tensor,
        gan_stage: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        posteriors = self.encode(samples).latent_dist

        z = posteriors.sample()

        rec_samples = self.decode(z).sample

        return self.loss(
            samples=samples,
            posteriors=posteriors,
            rec_samples=rec_samples,
            last_layer_weight=self.decoder.conv_out.weight,
            gan_stage=gan_stage,
        )

    @apply_forward_hook
    def validation_step(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        posteriors = self.encode(samples).latent_dist

        z = posteriors.sample()

        rec_samples = self.decode(z).sample

        loss, log_dict = self.loss(
            samples=samples,
            posteriors=posteriors,
            rec_samples=rec_samples,
            last_layer_weight=self.decoder.conv_out.weight,
            gan_stage="none",
        )

        return rec_samples, loss, log_dict
