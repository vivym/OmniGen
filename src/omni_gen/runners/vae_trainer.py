import copy
import os

import torch
from accelerate import Accelerator
from ray.train import CheckpointConfig, ScalingConfig, SyncConfig, RunConfig
from ray.train.torch import TorchTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm

from omni_gen.data.video_dataset import VideoDataset
from omni_gen.models.video_vae import AutoencoderKL
from omni_gen.utils import logging
from omni_gen.utils.ema import EMAModel
from .runner import Runner

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class VAETrainer(Runner):
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

        vae, ema_vae = self.setup_models(accelerator)
        params_to_optimize = list(vae.parameters_without_loss())

        optimizer, lr_scheduler = self.setup_optimizer(vae.parameters_without_loss())

        disc_optimizer, disc_lr_scheduler = self.setup_optimizer(vae.loss.parameters())

        (
            train_dataloader, vae, optimizer, lr_scheduler, disc_optimizer, disc_lr_scheduler
        ) = accelerator.prepare(
            train_dataloader, vae, optimizer, lr_scheduler, disc_optimizer, disc_lr_scheduler
        )
        train_dataloader: DataLoader
        vae: AutoencoderKL
        optimizer: torch.optim.Optimizer
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler
        disc_optimizer: torch.optim.Optimizer
        disc_lr_scheduler: torch.optim.lr_scheduler.LRScheduler

        if accelerator.is_main_process:
            accelerator.init_trackers(
                project_name=self.runner_config.name,
                config=self.runner_config,
            )

        total_batch_size = (
            self.runner_config.train_batch_size * accelerator.num_processes * self.runner_config.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Instantaneous batch size per device = {self.runner_config.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.runner_config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.runner_config.max_steps}")
        global_step = 0
        initial_global_step = 0
        gan_stage = "none" if self.runner_config.discriminator_start_steps > global_step else "generator"

        progress_bar = tqdm(
            range(0, self.runner_config.max_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        # TODO: skip initial steps if resuming from checkpoint

        done = False
        while not done:
            vae.train()

            loss_dict: dict[str, torch.Tensor] = {}

            for batch in train_dataloader:
                with accelerator.accumulate(vae):
                    loss, loss_dict_i = vae.training_step(
                        samples=batch["pixel_values"],
                        gan_stage=gan_stage,
                    )

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            params_to_optimize, self.runner_config.gradient_clipping
                        )

                    # Gather the losses across all processes for logging (if we use distributed training).
                    for k, v in loss_dict_i.items():
                        if k not in loss_dict:
                            loss_dict[k] = 0.0

                        losses: torch.Tensor = accelerator.gather(v)
                        loss_dict[k] += losses.mean()

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    disc_optimizer.step()
                    disc_lr_scheduler.step()
                    disc_optimizer.zero_grad()

                logs = {"lr": lr_scheduler.get_last_lr()[0]}

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if self.runner_config.use_ema:
                        ema_vae.step(vae.parameters())

                    progress_bar.update(1)
                    global_step += 1

                    for k, v in loss_dict.items():
                        loss_dict[k] = v.item() / self.runner_config.gradient_accumulation_steps
                    # logs.update({"loss": loss_dict["loss"]})
                    logs.update(loss_dict)
                    accelerator.log(loss_dict, step=global_step)
                    loss_dict: dict[str, torch.Tensor] = {}

                    if gan_stage == "generator":
                        gan_stage = "discriminator"
                    elif gan_stage == "discriminator":
                        gan_stage = "generator"

                    if gan_stage == "none" and global_step >= self.runner_config.discriminator_start_steps:
                        gan_stage = "generator"

                progress_bar.set_postfix(**logs)

                if (
                    accelerator.sync_gradients and \
                    accelerator.is_main_process and \
                    global_step % self.runner_config.checkpointing_every_n_steps == 0
                ):
                    accelerator.wait_for_everyone()

                    save_path = os.path.join(self.runner_config.storage_path, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                if global_step >= self.runner_config.max_steps:
                    done = True
                    break

        accelerator.end_training()

    def setup_dataloader(self) -> DataLoader:
        video_paths = []
        with open(self.data_config.dataset_name_or_path, "r") as f:
            for line in f:
                line: str = line.strip()
                if line:
                    video_paths.append(line.strip())

        dataset = VideoDataset(video_paths)

        return DataLoader(
            dataset,
            batch_size=self.runner_config.train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

    def setup_models(self, accelerator: Accelerator) -> tuple[AutoencoderKL, EMAModel | None]:
        vae = AutoencoderKL(
            in_channels=self.model_config.in_channels,
            out_channels=self.model_config.out_channels,
            down_block_types=self.model_config.down_block_types,
            up_block_types=self.model_config.up_block_types,
            block_out_channels=self.model_config.block_out_channels,
            use_gc_blocks=self.model_config.use_gc_blocks,
            mid_block_type=self.model_config.mid_block_type,
            mid_block_use_attention=self.model_config.mid_block_use_attention,
            mid_block_num_attention_heads=self.model_config.mid_block_num_attention_heads,
            layers_per_block=self.model_config.layers_per_block,
            act_fn=self.model_config.act_fn,
            num_attention_heads=self.model_config.num_attention_heads,
            latent_channels=self.model_config.latent_channels,
            norm_num_groups=self.model_config.norm_num_groups,
            scaling_factor=self.model_config.scaling_factor,
            with_loss=True,
        )

        vae.train()

        # TODO: dtype? (for lpips_metric)
        vae.loss.to(device=accelerator.device)

        # Create EMA for the vae.
        if self.runner_config.use_ema:
            ema_vae = copy.deepcopy(vae)
            ema_vae = EMAModel(
                ema_vae.parameters(),
            )
            ema_vae.to(accelerator.device)
        else:
            ema_vae = None

        return vae, ema_vae
