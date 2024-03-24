from typing import Optional

import numpy as np
import torch


class DiagonalGaussianDistribution:
    def __init__(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        deterministic: bool = False,
    ):
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.deterministic = deterministic

        if deterministic:
            self.var = self.std = torch.zeros_like(self.mean)
        else:
            self.std = torch.exp(0.5 * self.logvar)
            self.var = torch.exp(self.logvar)

    def sample(self, generator: torch.Generator | None = None) -> torch.FloatTensor:
        x = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * x

    def mode(self):
        return self.mean

    def kl(self, other: Optional["DiagonalGaussianDistribution"] = None) -> torch.Tensor:
        dims = list(range(1, self.mean.ndim))

        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=dims,
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=dims,
                )

    def nll(self, sample: torch.Tensor) -> torch.Tensor:
        dims = list(range(1, self.mean.ndim))

        if self.deterministic:
            return torch.Tensor([0.0])

        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )
