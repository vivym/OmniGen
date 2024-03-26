import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from omni_gen.utils.distributions import DiagonalGaussianDistribution
from .discriminator import Discriminator2D, Discriminator3D
from .lpips import LPIPSMetric


class VAELoss(nn.Module):
    def __init__(
        self,
        lpips_model_name_or_path: str = "vivym/lpips",
        init_logvar: float = 0.0,
        reconstruction_loss_type: str = "l1",   # Choose from ["l1", "l2"]
        reconstruction_loss_weight: float = 1.0,
        perceptual_loss_weight: float = 1.0,
        nll_loss_weight: float = 1.0,
        kl_loss_weight: float = 1.0,
        discriminator_loss_weight: float = 0.5,
        disc_in_channels: int = 3,
        disc_block_out_channels: tuple[int] = (64,),
    ):
        super().__init__()

        self.reconstruction_loss_type = reconstruction_loss_type
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.nll_loss_weight = nll_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.discriminator_loss_weight = discriminator_loss_weight

        # TODO: video perception loss
        # TODO: LeCAM
        # TODO: Discriminator gradient penalty

        self.lpips_metric = LPIPSMetric.from_pretrained(lpips_model_name_or_path)

        self.logvar = nn.Parameter(torch.full((), init_logvar, dtype=torch.float32))

        self.discriminator_2d = Discriminator2D(
            in_channels=disc_in_channels,
            block_out_channels=disc_block_out_channels,
        )

        self.discriminator_3d = Discriminator3D(
            in_channels=disc_in_channels,
            block_out_channels=disc_block_out_channels,
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
        flattend_samples = rearrange(samples, "b c t h w -> (b t) c h w")
        flattend_rec_samples = rearrange(rec_samples, "b c t h w -> (b t) c h w")

        if self.reconstruction_loss_type == "l1":
            loss_rec = torch.abs(flattend_samples - flattend_rec_samples)
        elif self.reconstruction_loss_type == "l2":
            loss_rec = (flattend_samples - flattend_rec_samples) ** 2
        else:
            raise ValueError(f"Invalid reconstruction loss type: {self.reconstruction_loss_type}")

        loss_perceptual = self.lpips_metric(flattend_samples, flattend_rec_samples)

        loss_nll = (
            self.reconstruction_loss_weight * loss_rec + self.perceptual_loss_weight * loss_perceptual
        ) / torch.exp(self.logvar) + self.logvar
        loss_nll = loss_nll.sum() / loss_nll.shape[0]

        loss_kl = posteriors.kl()
        loss_kl = loss_kl.sum() / loss_kl.shape[0]

        loss = self.nll_loss_weight * loss_nll + self.kl_loss_weight * loss_kl

        log_dict = {
            "loss_rec": loss_rec.detach().mean(),
            "loss_perceptual": loss_perceptual.detach().mean(),
            "loss_nll": loss_nll.detach(),
            "loss_kl": loss_kl.detach(),
            "logvar": self.logvar.detach(),
        }

        if gan_stage == "generator":
            # Update the Generator
            logits_fake_2d = self.discriminator_2d(rec_samples[:, :, 0])
            loss_g_2d = -torch.mean(logits_fake_2d)

            if last_layer_weight is not None:
                disc_weight_2d = compute_adaptive_disc_weight(
                    loss_nll, loss_g_2d, last_layer_weight
                ) * self.discriminator_loss_weight
                log_dict["disc_weight_2d"] = disc_weight_2d
            else:
                disc_weight_2d = 0.0

            loss += disc_weight_2d * loss_g_2d

            log_dict["loss_g_2d"] = loss_g_2d.detach()

            if samples.shape[2] > 1:
                logits_fake_3d = self.discriminator_3d(rec_samples[:, :, 1:])
                loss_g_3d = -torch.mean(logits_fake_3d[:, :, 1:])

                if last_layer_weight is not None:
                    disc_weight_3d = compute_adaptive_disc_weight(
                        loss_nll, loss_g_3d, last_layer_weight
                    ) * self.discriminator_loss_weight
                    log_dict["disc_weight_3d"] = disc_weight_3d
                else:
                    disc_weight_3d = 0.0

                loss += disc_weight_3d * loss_g_3d

                log_dict["loss_g_3d"] = loss_g_3d.detach()

        elif gan_stage == "discriminator":
            # Update the Discriminator
            logits_real_2d = self.discriminator_2d(samples[:, :, 0].detach())
            logits_fake_2d = self.discriminator_2d(rec_samples[:, :, 0].detach())

            loss_real_2d = torch.mean(F.relu(1. - logits_real_2d))
            loss_fake_2d = torch.mean(F.relu(1. + logits_fake_2d))
            loss_d_2d = (loss_real_2d + loss_fake_2d) * 0.5

            loss += loss_d_2d

            log_dict["loss_real_2d"] = loss_real_2d.detach()
            log_dict["loss_fake_2d"] = loss_real_2d.detach()
            log_dict["loss_d_2d"] = loss_d_2d.detach()

            if samples.shape[2] > 1:
                logits_real_3d = self.discriminator_3d(samples[:, :, 1:].detach())
                logits_fake_3d = self.discriminator_3d(rec_samples[:, :, 1:].detach())

                loss_real_3d = torch.mean(F.relu(1. - logits_real_3d))
                loss_fake_3d = torch.mean(F.relu(1. + logits_fake_3d))
                loss_d_3d = (loss_real_3d + loss_fake_3d) * 0.5

                loss += loss_d_3d

                log_dict["loss_real_3d"] = loss_real_3d.detach()
                log_dict["loss_fake_3d"] = loss_real_3d.detach()
                log_dict["loss_d_3d"] = loss_d_3d.detach()

        log_dict["loss"] = loss.detach()

        return loss, log_dict


def compute_adaptive_disc_weight(
    loss_nll: torch.Tensor, loss_g: torch.Tensor, last_layer_weight: nn.Parameter
) -> torch.Tensor:
    nll_grads = torch.autograd.grad(loss_nll, last_layer_weight, retain_graph=True)[0]
    g_grads = torch.autograd.grad(loss_g, last_layer_weight, retain_graph=True)[0]

    with torch.no_grad():
        disc_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        disc_weight = torch.clamp(disc_weight, min=0.0, max=1e4)

    return disc_weight
