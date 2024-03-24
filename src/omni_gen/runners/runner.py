import abc

import datasets
import diffusers
import torch
import transformers
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import ProjectConfiguration, set_seed

from omni_gen.configs import DataConfig, ModelConfig, OptimizerConfig, RunnerConfig
from omni_gen.utils import logging
from omni_gen.utils.lr_schedulers import get_lr_scheduler

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Runner(abc.ABC):
    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        runner_config: RunnerConfig,
    ):
        self.data_config = data_config
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.runner_config = runner_config

    @abc.abstractmethod
    def __call__(self):
        ...

    def setup_accelerator(self) -> Accelerator:
        project_config = ProjectConfiguration(
            project_dir=self.runner_config.storage_path,
        )

        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=self.runner_config.hf_ds_config_path,
            gradient_accumulation_steps=self.runner_config.gradient_accumulation_steps,
            gradient_clipping=self.runner_config.gradient_clipping,
            zero_stage=self.runner_config.zero_stage,
        )
        deepspeed_plugin.hf_ds_config.config["train_micro_batch_size_per_gpu"] = self.runner_config.train_batch_size

        accelerator = Accelerator(
            mixed_precision=self.runner_config.mixed_precision,
            gradient_accumulation_steps=self.runner_config.gradient_accumulation_steps,
            deepspeed_plugin=deepspeed_plugin,
            log_with=self.runner_config.log_with,
            project_config=project_config,
        )

        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        if self.runner_config.seed is not None:
            set_seed(self.runner_config.seed)

        return accelerator

    def setup_optimizer(
        self,
        parameters: list[torch.nn.Parameter],
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        if self.optimizer_config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "You need to install the bitsandbytes package to use 8-bit AdamW: `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            parameters,
            lr=self.optimizer_config.learning_rate,
            betas=self.optimizer_config.adam_beta,
            weight_decay=self.optimizer_config.weight_decay,
            eps=self.optimizer_config.adam_epsilon,
        )

        lr_scheduler = get_lr_scheduler(
            self.optimizer_config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.optimizer_config.lr_warmup_steps * self.runner_config.gradient_accumulation_steps,
            num_training_steps=self.runner_config.max_steps * self.runner_config.gradient_accumulation_steps,
        )

        return optimizer, lr_scheduler
