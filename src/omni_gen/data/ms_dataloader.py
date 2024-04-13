from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
from torch.utils.data import DataLoader


@dataclass
class DataLoaderConfig:
    data_loader: DataLoader

    weight: float = 1.0

    loss_scale: float = 1.0


class MultiSourceIterator:
    def __init__(self, sources: list[tuple[Iterator, float, float]]):
        self.source_iters = [s[0] for s in sources]

        weights = np.array([s[1] for s in sources])
        self.probs: np.ndarray =  weights / weights.sum()

        self.loss_scales = [s[2] for s in sources]

        self.indices = np.arange(len(sources))

        self.remaining_iters = len(sources)

    def __next__(self) -> tuple[Any, float]:
        data = None
        loss_scale = None
        while self.remaining_iters > 0:
            try:
                idx = np.random.choice(self.indices, p=self.probs)
                data = next(self.source_iters[idx])
                loss_scale = self.loss_scales[idx]
                break
            except StopIteration:
                self.remaining_iters -= 1
                self.probs[idx] = 0.0
                if self.remaining_iters > 0:
                    self.probs = self.probs / self.probs.sum()

        if data is None:
            raise StopIteration
        else:
            return data, loss_scale


class MultiSourceDataLoader:
    def __init__(self, data_loaders: list[DataLoaderConfig]):
        self.data_loaders = data_loaders

    def __iter__(self) -> MultiSourceIterator:
        return MultiSourceIterator([
            (iter(config.data_loader), config.weight, config.loss_scale)
            for config in self.data_loaders
        ])
