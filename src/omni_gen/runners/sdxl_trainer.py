import copy
import os

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils.import_utils import is_xformers_available
from ray.train import CheckpointConfig, ScalingConfig, SyncConfig, RunConfig
from ray.train.torch import TorchTrainer
from packaging import version
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, PretrainedConfig, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
)
from tqdm import tqdm

from omni_gen.utils import logging
from .runner import Runner

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SDXLTrainer(Runner):
    def __call__(self):
        trainer = TorchTrainer(
            self.train_loop_per_worker,
            scaling_config=ScalingConfig(
                trainer_resources={"CPU": self.runner_config.num_cpus_per_worker},
                num_workers=self.runner_config.num_devices,
                use_gpu=self.runner_config.num_gpus_per_worker > 0,
                resources_per_worker={
                    "CPU": self.runner_config.num_cpus_per_worker,
                    "GPU": self.runner_config.num_gpus_per_worker,
                },
            ),
            run_config=RunConfig(
                name=self.runner_config.name,
                # storage_path=self.runner_config.storage_path,
                # TODO: failure_config
                checkpoint_config=CheckpointConfig(),
                sync_config=SyncConfig(
                    sync_artifacts=True,
                ),
                verbose=self.runner_config.verbose_mode,
                log_to_file=True,
            ),
        )

        trainer.fit()

    def train_loop_per_worker(self):
        if self.runner_config.allow_tf32:
            torch.set_float32_matmul_precision("high")

        accelerator = self.setup_accelerator()

        train_dataloader = self.setup_dataloader()

        (
            unet, ema_unet, vae, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
        ) = self.setup_models(accelerator)

        noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            self.model_config.pretrained_model_name_or_path, subfolder="scheduler"
        )

        # Prepare everything with our `accelerator`.
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    def setup_dataloader(self) -> DataLoader:
        ...

    def setup_models(
        self, accelerator: Accelerator
    ) -> tuple[
        UNet2DConditionModel,
        EMAModel | None,
        AutoencoderKL,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
        CLIPTokenizer,
    ]:
        # Load the tokenizers
        tokenizer_one = AutoTokenizer.from_pretrained(
            self.model_config.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.model_config.revision,
            use_fast=False,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            self.model_config.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=self.model_config.revision,
            use_fast=False,
        )

        # import correct text encoder classes
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            self.model_config.pretrained_model_name_or_path, revision=self.model_config.revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            self.model_config.pretrained_model_name_or_path,
            revision=self.model_config.revision,
            subfolder="text_encoder_2",
        )

        # Check for terminal SNR in combination with SNR Gamma
        text_encoder_one = text_encoder_cls_one.from_pretrained(
            self.model_config.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.model_config.revision,
            variant=self.model_config.variant,
        )
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            self.model_config.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            revision=self.model_config.revision,
            variant=self.model_config.variant,
        )

        vae_path = (
            self.model_config.pretrained_model_name_or_path
            if self.model_config.pretrained_vae_model_name_or_path is None
            else self.model_config.pretrained_vae_model_name_or_path
        )
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if self.model_config.pretrained_vae_model_name_or_path is None else None,
            revision=self.model_config.revision,
            variant=self.model_config.variant,
        )

        unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            self.model_config.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.model_config.revision,
            variant=self.model_config.variant,
        )

        # Freeze vae and text encoders.
        vae.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        # Set unet as trainable.
        unet.train()

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to weight_dtype
        # The VAE is in float32 to avoid NaN losses.
        vae.to(accelerator.device, dtype=torch.float32)
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)

        # Create EMA for the unet.
        if self.runner_config.use_ema:
            ema_unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
                self.model_config.pretrained_model_name_or_path,
                subfolder="unet",
                revision=self.model_config.revision,
                variant=self.model_config.variant,
            )
            ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

        if self.runner_config.use_xformers:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models: list, weights: list, output_dir: str):
            if accelerator.is_main_process:
                if self.runner_config.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for model in models:
                    model: UNet2DConditionModel
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models: list, input_dir: str):
            if self.runner_config.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model: UNet2DConditionModel = models.pop()

                # load diffusers style into model
                load_model: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        if self.runner_config.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        if self.runner_config.use_ema:
            ema_unet.to(accelerator.device)

        return (
            unet, ema_unet, vae, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
        )


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,
    revision: str | None = None,
    subfolder: str = "text_encoder",
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")
