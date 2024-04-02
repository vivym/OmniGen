from dataclasses import dataclass
from pathlib import Path


def normalize_path(original_path: str) -> str:
    path = Path(original_path)

    if path.exists():
        return str(path.resolve())
    else:
        return original_path


@dataclass
class DataConfig:
    dataset_name_or_path: str | list[str]

    revision: str | None = None

    num_workers: int = 8

    spatial_size: int | tuple[int, int] = 256

    num_frames: int = 17

    frame_intervals: int | tuple[int, ...] = 1

    def __post_init__(self):
        if isinstance(self.dataset_name_or_path, str):
            self.dataset_name_or_path = normalize_path(self.dataset_name_or_path)
        else:
            self.dataset_name_or_path = [normalize_path(path) for path in self.dataset_name_or_path]


@dataclass
class ModelConfig:
    pretrained_model_name_or_path: str | None = None

    revision: str | None = None

    variant: str | None = None

    pretrained_vae_model_name_or_path: str | None = None

    in_channels: int = 3

    out_channels: int = 3

    down_block_types: tuple[str, ...] = ("SpatialDownBlock3D",)

    up_block_types: tuple[str, ...] = ("SpatialUpBlock3D",)

    block_out_channels: tuple[int, ...] = (64,)

    use_gc_blocks: tuple[bool, ...] | None = None

    mid_block_type: str = "MidBlock3D"

    mid_block_use_attention: bool = False

    mid_block_attention_type: str = "3d"

    mid_block_num_attention_heads: int = 1

    layers_per_block: int = 1

    act_fn: str = "silu"

    num_attention_heads: int = 1

    latent_channels: int = 8

    norm_num_groups: int = 32

    scaling_factor: float = 0.18215

    lpips_model_name_or_path: str = "vivym/lpips"

    init_logvar: float = 0.0

    reconstruction_loss_type: str = "l1"

    reconstruction_loss_weight: float = 1.0

    perceptual_loss_weight: float = 1.0

    nll_loss_weight: float = 1.0

    kl_loss_weight: float = 1.0

    discriminator_loss_weight: float = 0.5

    disc_block_out_channels: tuple[int, ...] = (64,)

    load_from_2d_vae: str | None = None

    def __post_init__(self):
        if self.pretrained_model_name_or_path is not None:
            self.pretrained_model_name_or_path = normalize_path(self.pretrained_model_name_or_path)

        if self.pretrained_vae_model_name_or_path is not None:
            self.pretrained_vae_model_name_or_path = normalize_path(self.pretrained_vae_model_name_or_path)

        if self.lpips_model_name_or_path is not None:
            self.lpips_model_name_or_path = normalize_path(self.lpips_model_name_or_path)

        if self.load_from_2d_vae is not None:
            self.load_from_2d_vae = normalize_path(self.load_from_2d_vae)


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4

    use_8bit_adam: bool = False

    weight_decay: float = 1e-2

    adam_beta: tuple[float, float] = (0.9, 0.999)

    adam_epsilon: float = 1e-8

    # The scheduler type to use.
    # Choose between "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "piecewise_constant"
    lr_scheduler: str = "constant"

    lr_warmup_steps: int = 0


@dataclass
class RunnerConfig:
    runner_cls_path: str

    name: str = "omni-gen"

    storage_path: str = "./outputs"

    num_devices: int = 1

    max_steps: int = 1000000

    train_batch_size: int = 1

    val_batch_size: int = 1

    seed: int | None = None

    use_ema: bool = False

    ema_decay: float = 0.9999

    ema_power: float = 0.666666

    use_xformers: bool = False

    # Path to DeepSpeed config file
    hf_ds_config_path: str | None = None

    # Number of steps to accumulate gradients before updating optimizer states.
    gradient_accumulation_steps: int = 1

    # Enable gradient clipping with value.
    gradient_clipping: float | None = None

    # Possible options are 0,1,2,3; Default will be taken from environment variable.
    zero_stage: int | None = None

    gradient_checkpointing: bool = False

    mixed_precision: str | None = None  # Choose from 'no','fp16','bf16 or 'fp8'.

    allow_tf32: bool = False

    log_with: str | list[str] | None = None

    num_cpus_per_worker: int = 1

    num_gpus_per_worker: int = 1

    validation_every_n_steps: int | None = None

    checkpointing_every_n_steps: int | None = None

    num_checkpoints_to_keep: int | None = None

    checkpointing_score_attribute: str | None = None

    checkpointing_score_order: str = "max"

    verbose_mode: int = 1   #  0 = silent, 1 = default, 2 = verbose. Defaults to 1.

    discriminator_start_steps: int = 1000

    resume_from_checkpoint: str | None = None

    max_failures: int = 0

    def __post_init__(self):
        self.storage_path = normalize_path(self.storage_path)
        if self.hf_ds_config_path is not None:
            self.hf_ds_config_path = normalize_path(self.hf_ds_config_path)
        if self.resume_from_checkpoint is not None:
            self.resume_from_checkpoint = normalize_path(self.resume_from_checkpoint)
