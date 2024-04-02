import random
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TrF
from torchvision.transforms.functional import InterpolationMode
from PIL import Image


class Resize(nn.Module):
    def __init__(
        self,
        size: int | tuple[int, int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        max_size: int | None = None,
        antialias: bool = True,
        random_scale: bool = False,
    ):
        super().__init__()

        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")

        self.size = size
        self.max_size = max_size

        self.interpolation = interpolation
        self.antialias = antialias
        self.random_scale = random_scale

    def forward(self, x):
        if isinstance(x, Image.Image):
            min_size = min(x.size)
        elif isinstance(x, torch.Tensor):
            min_size = x.shape[-2:]
        else:
            raise TypeError(f"Input type {type(x)} not supported")

        if min_size > self.size and self.random_scale:
            scale = random.uniform(self.size / min_size, 1.0)
            target_size = max(int(round(min_size * scale)), self.size)
        else:
            target_size = self.size

        return TrF.resize(x, target_size, self.interpolation, self.max_size, self.antialias)


class ImageDataset(Dataset):
    def __init__(
        self,
        image_paths: list[str],
        size: int | tuple[int, int],
        training: bool = True,
    ):
        self.image_paths = image_paths
        self.training = training

        self.transform = T.Compose([
            Resize(size=size, random_scale=True),
            T.RandomCrop(size=size) if training else T.CenterCrop(size=size),
            T.RandomHorizontalFlip(p=0.5) if training else T.Lambda(lambda x: x),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)
        # Normalize to [-1, 1]
        image = image * 2 - 1

        return {"pixel_values": image}
