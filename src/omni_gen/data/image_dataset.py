import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


class ImageDataset(Dataset):
    def __init__(
        self,
        image_paths: list[str],
        size: int | tuple[int, int],
    ):
        self.image_paths = image_paths

        self.transform = T.Compose([
            T.RandomResizedCrop(size=size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        # Normalize to [-1, 1]
        return image * 2 - 1
