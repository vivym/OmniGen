import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from omni_gen.utils.accelerate import apply_forward_hook
from omni_gen.utils.distributions import DiagonalGaussianDistribution
from .discriminator import Discriminator2D, Discriminator3D
from .lpips import LPIPSMetric


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        import functools
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


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
        image_mode: bool = False,
    ):
        super().__init__()

        self.reconstruction_loss_type = reconstruction_loss_type
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.nll_loss_weight = nll_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.discriminator_loss_weight = discriminator_loss_weight
        self.image_mode = image_mode

        # TODO: video perception loss
        # TODO: LeCAM
        # TODO: Discriminator gradient penalty

        self.lpips_metric = LPIPSMetric.from_pretrained(lpips_model_name_or_path)

        self.logvar = nn.Parameter(torch.full((), init_logvar, dtype=torch.float32))

        self.logvar.requires_grad = False

        # self.discriminator_2d = Discriminator2D(
        #     in_channels=disc_in_channels,
        #     block_out_channels=disc_block_out_channels,
        # )

        self.discriminator_2d = NLayerDiscriminator(
            input_nc=3,
            n_layers=3,
        ).apply(weights_init)

        if not self.image_mode:
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

    @apply_forward_hook
    def compute_ae_loss(
        self,
        samples: torch.Tensor,
        posteriors: DiagonalGaussianDistribution,
        rec_samples: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.image_mode:
            samples = samples[:, :, None]
            rec_samples = rec_samples[:, :, None]

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

        return loss, loss_nll, {
            "loss_rec": loss_rec.detach().mean(),
            "loss_perceptual": loss_perceptual.detach().mean(),
            "loss_nll": loss_nll.detach(),
            "loss_kl": loss_kl.detach(),
            "logvar": self.logvar.detach(),
        }

    @apply_forward_hook
    def compute_generator_loss(
        self,
        rec_samples: torch.Tensor,
        loss_nll: torch.Tensor,
        last_layer_weight: nn.Parameter | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.image_mode:
            rec_samples = rec_samples[:, :, None]

        log_dict: dict[str, torch.Tensor] = {}

        logits_fake_2d = self.discriminator_2d(rec_samples[:, :, 0])
        loss_g_2d = -torch.mean(logits_fake_2d)

        if last_layer_weight is not None:
            disc_weight_2d = compute_adaptive_disc_weight(
                loss_nll, loss_g_2d, last_layer_weight
            )
            log_dict["disc_weight_2d"] = disc_weight_2d
            disc_weight_2d = disc_weight_2d * self.discriminator_loss_weight
        else:
            disc_weight_2d = 0.0

        loss = disc_weight_2d * loss_g_2d

        log_dict["loss_g_2d"] = loss_g_2d.detach()

        if rec_samples.shape[2] > 1:
            logits_fake_3d = self.discriminator_3d(rec_samples[:, :, 1:])
            loss_g_3d = -torch.mean(logits_fake_3d)

            if last_layer_weight is not None:
                disc_weight_3d = compute_adaptive_disc_weight(
                    loss_nll, loss_g_3d, last_layer_weight
                )
                log_dict["disc_weight_3d"] = disc_weight_3d
                disc_weight_3d = disc_weight_3d * self.discriminator_loss_weight
            else:
                disc_weight_3d = 0.0

            loss += disc_weight_3d * loss_g_3d

            log_dict["loss_g_3d"] = loss_g_3d.detach()

        return loss, log_dict

    @apply_forward_hook
    def compute_discriminator_loss(
        self,
        samples: torch.Tensor,
        rec_samples: torch.Tensor,
    ):
        if self.image_mode:
            samples = samples[:, :, None]
            rec_samples = rec_samples[:, :, None]

        log_dict: dict[str, torch.Tensor] = {}

        logits_real_2d = self.discriminator_2d(samples[:, :, 0].detach())
        logits_fake_2d = self.discriminator_2d(rec_samples[:, :, 0].detach())

        loss_real_2d = torch.mean(F.relu(1. - logits_real_2d))
        loss_fake_2d = torch.mean(F.relu(1. + logits_fake_2d))
        loss_d_2d = (loss_real_2d + loss_fake_2d) * 0.5

        loss = loss_d_2d

        log_dict["loss_real_2d"] = loss_real_2d.detach()
        log_dict["loss_fake_2d"] = loss_fake_2d.detach()
        log_dict["loss_d_2d"] = loss_d_2d.detach()

        if samples.shape[2] > 1:
            logits_real_3d = self.discriminator_3d(samples[:, :, 1:].detach())
            logits_fake_3d = self.discriminator_3d(rec_samples[:, :, 1:].detach())

            loss_real_3d = torch.mean(F.relu(1. - logits_real_3d))
            loss_fake_3d = torch.mean(F.relu(1. + logits_fake_3d))
            loss_d_3d = (loss_real_3d + loss_fake_3d) * 0.5

            loss += loss_d_3d

            log_dict["loss_real_3d"] = loss_real_3d.detach()
            log_dict["loss_fake_3d"] = loss_fake_3d.detach()
            log_dict["loss_d_3d"] = loss_d_3d.detach()

        return loss, log_dict


def compute_adaptive_disc_weight(
    loss_nll: torch.Tensor, loss_g: torch.Tensor, last_layer_weight: nn.Parameter
) -> torch.Tensor:
#     nll_grads = torch.autograd.grad(loss_nll, last_layer_weight, retain_graph=True)[0]
#     g_grads = torch.autograd.grad(loss_g, last_layer_weight, retain_graph=True)[0]

#     with torch.no_grad():
#         disc_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
#         disc_weight = torch.clamp(disc_weight, min=0.0, max=1e4)

#     return disc_weight


# def calculate_adaptive_weight(nll_loss, g_loss, last_layer=None):
    nll_grads = torch.autograd.grad(loss_nll, last_layer_weight, retain_graph=True)[0]
    g_grads = torch.autograd.grad(loss_g, last_layer_weight, retain_graph=True)[0]
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight
