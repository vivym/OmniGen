import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


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
            T.Resize(size=size),
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
