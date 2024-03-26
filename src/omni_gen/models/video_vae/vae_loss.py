import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from omni_gen.utils.distributions import DiagonalGaussianDistribution
from .discriminator import Discriminator
from .lpips import LPIPSMetric


class VAELoss(nn.Module):
    def __init__(
        self,
        lpips_model_name_or_path: str = "vivym/lpips",
        init_logvar: float = 0.0,
        reconstruction_loss_weight: float = 1.0,
        perceptual_loss_weight: float = 1.0,
        nll_loss_weight: float = 1.0,
        kl_loss_weight: float = 1.0,
        discriminator_loss_weight: float = 0.5,
        disc_in_channels: int = 3,
        disc_block_out_channels: tuple[int] = (64,),
        disc_use_gc_blocks: tuple[bool] | None = None,
        disc_layers_per_block: int = 2,
        disc_norm_num_groups: int = 32,
        disc_act_fn: str = "silu",
        disc_num_attention_heads: int = 1,
    ):
        super().__init__()

        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.nll_loss_weight = nll_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.discriminator_loss_weight = discriminator_loss_weight

        self.lpips_metric = LPIPSMetric.from_pretrained(lpips_model_name_or_path)

        self.logvar = nn.Parameter(torch.full((), init_logvar, dtype=torch.float32))

        self.discriminator = Discriminator(
            in_channels=disc_in_channels,
            block_out_channels=disc_block_out_channels,
            use_gc_blocks=disc_use_gc_blocks,
            layers_per_block=disc_layers_per_block,
            norm_num_groups=disc_norm_num_groups,
            act_fn=disc_act_fn,
            num_attention_heads=disc_num_attention_heads,
        )

    def to(self, *args, **kwargs) -> "VAELoss":
        device = None
        dtype = None
        for arg in args:
            if isinstance(arg, torch.device):
                device = arg
            elif isinstance(arg, torch.dtype):
                dtype = arg

        for k, v in kwargs.items():
            if k == "device":
                device = v
            elif k == "dtype":
                dtype = v

        self.lpips_metric.to(device=device, dtype=dtype)

        return super().to(*args, **kwargs)

    def forward(
        self,
        samples: torch.Tensor,
        posteriors: DiagonalGaussianDistribution,
        rec_samples: torch.Tensor,
        last_layer_weight: nn.Parameter | None = None,
        gan_stage: str = "generator",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if samples.ndim == 5:
            samples = rearrange(samples, "b c t h w -> (b t) c h w")
            rec_samples = rearrange(rec_samples, "b c t h w -> (b t) c h w")

        loss_rec = torch.abs(samples - rec_samples)

        loss_perceptual = self.lpips_metric(samples, rec_samples)

        loss_nll = (
            self.reconstruction_loss_weight * loss_rec + self.perceptual_loss_weight * loss_perceptual
        ) / torch.exp(self.logvar) + self.logvar
        loss_nll = loss_nll.sum() / loss_nll.shape[0]

        loss_kl = posteriors.kl()
        loss_kl = loss_kl.sum() / loss_kl.shape[0]

        loss = self.nll_loss_weight * loss_nll + self.kl_loss_weight * loss_kl

        loss_dict = {
            "loss_rec": loss_rec.detach().mean(),
            "loss_perceptual": loss_perceptual.detach().mean(),
            "loss_nll": loss_nll.detach(),
            "loss_kl": loss_kl.detach(),
            "logvar": self.logvar.detach(),
        }

        if gan_stage == "generator":
            # Update the Generator
            logits_fake = self.discriminator(rec_samples)
            loss_g = -torch.mean(logits_fake)

            if last_layer_weight is not None:
                disc_weight = compute_adaptive_disc_weight(
                    loss_nll, loss_g, last_layer_weight
                ) * self.discriminator_loss_weight
            else:
                disc_weight = 0.0

            loss += disc_weight * loss_g

            loss_dict["loss_g"] = loss_g.detach()
            loss_dict["disc_weight"] = disc_weight
        elif gan_stage == "discriminator":
            # Update the Discriminator
            logits_real = self.discriminator(samples.detach())
            logits_fake = self.discriminator(rec_samples.detach())

            loss_real = torch.mean(F.relu(1. - logits_real))
            loss_fake = torch.mean(F.relu(1. + logits_fake))
            loss_d = (loss_real + loss_fake) * 0.5

            loss += loss_d

            loss_dict["loss_real"] = loss_real.detach()
            loss_dict["loss_fake"] = loss_fake.detach()
            loss_dict["loss_d"] = loss_d.detach()

        loss_dict["loss"] = loss.detach()

        return loss, loss_dict


def compute_adaptive_disc_weight(
    loss_nll: torch.Tensor, loss_g: torch.Tensor, last_layer_weight: nn.Parameter
) -> torch.Tensor:
    nll_grads = torch.autograd.grad(loss_nll, last_layer_weight, retain_graph=True)[0]
    g_grads = torch.autograd.grad(loss_g, last_layer_weight, retain_graph=True)[0]

    with torch.no_grad():
        disc_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        disc_weight = torch.clamp(disc_weight, min=0.0, max=1e4)

    return disc_weight
