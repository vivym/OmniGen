from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torchvision.models import vgg16, VGG16_Weights
from safetensors import safe_open


class LPIPSMetric:
    def __init__(self):
        super().__init__()

        self.vgg16 = VGG16()

        self.projs = nn.ModuleList([
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
            for in_channels in (64, 128, 256, 512, 512)
        ])

        self.mean = torch.as_tensor([-.030, -.088, -.188], dtype=torch.float32)[None, :, None, None]
        self.std = torch.as_tensor([.458, .448, .450], dtype=torch.float32)[None, :, None, None]

        self.vgg16.eval()
        for param in self.vgg16.parameters():
            param.requires_grad = False

        self.projs.eval()
        for param in self.projs.parameters():
            param.requires_grad = False

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "LPIPSMetric":
        _model_dir = Path(model_name_or_path)
        _model_path = _model_dir / "vgg_lpips_linear.safetensors"
        if _model_dir.exists() and _model_dir.is_dir() and _model_path.exists():
            model_path = str(_model_path.resolve())
        else:
            model_path = hf_hub_download(model_name_or_path, filename="vgg_lpips_linear.safetensors")

        state_dict = {}
        with safe_open(model_path, framework="pt") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)

        lpips = cls()

        lpips.projs.load_state_dict(state_dict)

        return lpips

    def to(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "LPIPSMetric":
        self.vgg16.to(device=device, dtype=dtype)
        self.projs.to(device, dtype=dtype)
        self.mean = self.mean.to(device, dtype=dtype)
        self.std = self.std.to(device, dtype=dtype)

        return self

    def _norm_inputs(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs, targets = self._norm_inputs(inputs), self._norm_inputs(targets)

        inputs_features = self.vgg16(inputs)
        targets_features = self.vgg16(targets)

        res = 0
        for proj, input_features, target_features in zip(
            self.projs, inputs_features, targets_features
        ):
            input_features = F.normalize(input_features, p=2, dim=1)
            target_features = F.normalize(target_features, p=2, dim=1)

            scores: torch.Tensor = proj((input_features - target_features) ** 2)
            scores = scores.mean(dim=[2, 3], keepdim=True)
            res += scores

        return res


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()

        pretrained_features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), pretrained_features[x])

        for x in range(4, 9):
            self.slice2.add_module(str(x), pretrained_features[x])

        for x in range(9, 16):
            self.slice3.add_module(str(x), pretrained_features[x])

        for x in range(16, 23):
            self.slice4.add_module(str(x), pretrained_features[x])

        for x in range(23, 30):
            self.slice5.add_module(str(x), pretrained_features[x])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h

        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3
