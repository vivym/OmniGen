import io

from litdata import StreamingDataset
from PIL import Image
from torchvision import transforms as T

from .image_dataset import Resize


class StreamingImageDataset(StreamingDataset):
    def __init__(
        self,
        data_dir: str,
        spatial_size: int | tuple[int, int],
        training: bool = True,
        seed: int = 233,
        max_cache_size: int | str = "100GB",
    ):
        super().__init__(
            input_dir=data_dir,
            seed=seed,
            max_cache_size=max_cache_size,
        )

        self.transform = T.Compose([
            Resize(size=spatial_size, random_scale=True) if training else T.Lambda(lambda x: x),
            T.RandomCrop(size=spatial_size) if training else T.CenterCrop(size=spatial_size),
            T.RandomHorizontalFlip(p=0.5) if training else T.Lambda(lambda x: x),
            T.ToTensor(),
        ])

    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)

        buf = io.BytesIO(sample["image"])
        try:
            image = Image.open(buf).convert("RGB")
        except Exception:
            image = Image.new("RGB", (256, 256))
        image = self.transform(image)
        # Normalize to [-1, 1]
        image = image * 2 - 1

        return {"pixel_values": image}
