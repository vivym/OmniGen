import copy
import os
import random
import tempfile
from pathlib import Path

import ray.train
import torch
from accelerate import Accelerator
from ray.train import (
    Checkpoint, CheckpointConfig, FailureConfig, ScalingConfig, SyncConfig, RunConfig
)
from ray.train.torch import TorchTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm

from omni_gen.data.video_dataset import VideoDataset
from omni_gen.models.video_vae import AutoencoderKL
from omni_gen.utils import logging
from omni_gen.utils.ema import EMAModel
from omni_gen.utils.inflation import inflate_params_from_2d_vae
from .runner import Runner

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class VAETrainer(Runner):
    def __call__(self):
        storage_path = Path(self.runner_config.storage_path).resolve()
        experiment_path = storage_path / self.runner_config.name

        if (
            self.runner_config.resume_from_checkpoint == "latest" and
            TorchTrainer.can_restore(experiment_path)
        ):
            trainer = TorchTrainer.restore(experiment_path)
        else:
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
                    storage_path=str(storage_path),
                    failure_config=FailureConfig(max_failures=self.runner_config.max_failures),
                    checkpoint_config=CheckpointConfig(
                        num_to_keep=self.runner_config.num_checkpoints_to_keep,
                        checkpoint_score_attribute=self.runner_config.checkpointing_score_attribute,
                        checkpoint_score_order=self.runner_config.checkpointing_score_order,
                    ),
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

        try:
            self._train_loop_per_worker(accelerator)
        finally:
            accelerator.end_training()

    def _train_loop_per_worker(self, accelerator: Accelerator):
        train_dataloader, val_dataloader = self.setup_dataloader()

        vae, ema_vae = self.setup_models(accelerator)

        params_to_optimize = list(vae.parameters())
        optimizer, lr_scheduler = self.setup_optimizer(params_to_optimize)

        (
            train_dataloader, val_dataloader, vae, optimizer, lr_scheduler
        ) = accelerator.prepare(
            train_dataloader, val_dataloader, vae, optimizer, lr_scheduler
        )
        train_dataloader: DataLoader
        val_dataloader: DataLoader
        vae: AutoencoderKL
        optimizer: torch.optim.Optimizer
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler

        if accelerator.is_main_process:
            configs = {}
            for prefix, config in [
                ("data", self.data_config),
                ("model", self.model_config),
                ("optimizer", self.optimizer_config),
                ("runner", self.runner_config),
            ]:
                for k, v in config.__dict__.items():
                    configs[f"{prefix}_{k}"] = v

            accelerator.init_trackers(
                project_name=self.runner_config.name,
                config=configs,
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

        if (
            self.runner_config.resume_from_checkpoint is not None and
            self.runner_config.resume_from_checkpoint != "latest" and
            Path(self.runner_config.resume_from_checkpoint).exists()
        ):
            checkpoint = Checkpoint.from_directory(self.runner_config.resume_from_checkpoint)
        else:
            checkpoint: Checkpoint | None = ray.train.get_checkpoint()

        if checkpoint is not None:
            with checkpoint.as_directory() as checkpoint_dir:
                accelerator.load_state(checkpoint_dir)

        if lr_scheduler.last_epoch > 0:
            global_step = lr_scheduler.last_epoch // self.runner_config.gradient_accumulation_steps
            initial_global_step = global_step

        gan_stage = "none" if self.runner_config.discriminator_start_steps > global_step else "generator"

        progress_bar = tqdm(
            range(0, self.runner_config.max_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        # TODO: skip initial steps if resuming from checkpoint

        dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            dtype = torch.bfloat16

        done = False
        while not done:
            vae.train()

            log_dict: dict[str, torch.Tensor] = {}

            for batch in train_dataloader:
                with accelerator.accumulate(vae):
                    loss, log_dict_i = vae.training_step(
                        samples=batch["pixel_values"].to(dtype=dtype),
                        gan_stage=gan_stage,
                    )

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            params_to_optimize, self.runner_config.gradient_clipping
                        )

                    # Gather the values across all processes for logging (if we use distributed training).
                    for k, v in log_dict_i.items():
                        if k not in log_dict:
                            log_dict[k] = 0.0

                        values: torch.Tensor = accelerator.gather(v)
                        log_dict[k] += values.mean()

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                logs = {"lr": lr_scheduler.get_last_lr()[0]}

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if self.runner_config.use_ema:
                        ema_vae.step(filter(lambda x: x.requires_grad, vae.parameters()))

                    progress_bar.update(1)
                    global_step += 1

                    for k, v in log_dict.items():
                        log_dict[k] = v.item() / self.runner_config.gradient_accumulation_steps
                    logs.update(log_dict)
                    accelerator.log({"lr": logs["lr"]}, step=global_step)
                    accelerator.log(
                        {
                            f"train/{k}": v
                            for k, v in log_dict.items()
                        },
                        step=global_step,
                    )
                    log_dict: dict[str, torch.Tensor] = {}

                    if gan_stage == "generator":
                        gan_stage = "discriminator"
                    elif gan_stage == "discriminator":
                        gan_stage = "generator"

                    if gan_stage == "none" and global_step >= self.runner_config.discriminator_start_steps:
                        gan_stage = "generator"

                progress_bar.set_postfix(**logs)

                do_validation = (
                    accelerator.sync_gradients and \
                    self.runner_config.validation_every_n_steps is not None and \
                    global_step % self.runner_config.validation_every_n_steps == 0
                )

                do_checkpointing = (
                    accelerator.sync_gradients and \
                    self.runner_config.checkpointing_every_n_steps is not None and \
                    global_step % self.runner_config.checkpointing_every_n_steps == 0
                )

                if do_validation or do_checkpointing:
                    val_metrics = self.validation_loop(
                        accelerator=accelerator,
                        val_dataloader=val_dataloader,
                        vae=vae,
                        ema_vae=ema_vae,
                        dtype=dtype,
                        global_step=global_step,
                    )
                    val_metrics["global_step"] = global_step

                if do_checkpointing:
                    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                        accelerator.save_state(temp_checkpoint_dir)
                        logger.info(f"Saved state to {temp_checkpoint_dir}")
                        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                        ray.train.report(val_metrics, checkpoint=checkpoint)

                if global_step >= self.runner_config.max_steps:
                    done = True
                    break

            do_validation = self.runner_config.checkpointing_every_n_steps is None

            do_checkpointing = self.runner_config.checkpointing_every_n_steps is None

            if do_validation or do_checkpointing:
                val_metrics = self.validation_loop(
                    accelerator=accelerator,
                    val_dataloader=val_dataloader,
                    vae=vae,
                    ema_vae=ema_vae,
                    dtype=dtype,
                    global_step=global_step,
                )
                val_metrics["global_step"] = global_step

            if do_checkpointing:
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    accelerator.save_state(temp_checkpoint_dir)
                    logger.info(f"Saved state to {temp_checkpoint_dir}")
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                    ray.train.report(val_metrics, checkpoint=checkpoint)

    def validation_loop(
        self,
        accelerator: Accelerator,
        val_dataloader: DataLoader,
        vae: AutoencoderKL,
        ema_vae: EMAModel | None,
        dtype: torch.dtype,
        global_step: int,
    ) -> dict[str, float]:
        logger.info("Running validation...")

        if self.runner_config.use_ema:
            # Store the VAE parameters temporarily and load the EMA parameters to perform inference.
            ema_vae.store(vae.parameters())
            ema_vae.copy_to(vae.parameters())

        if vae.training:
            vae.eval()
            restore_training = True
        else:
            restore_training = False

        num_samples_to_log = 0
        samples_to_log = []
        rec_samples_to_log = []
        log_dict: dict[str, torch.Tensor] = {}
        num_samples = 0

        with torch.inference_mode():
            for batch in tqdm(val_dataloader, desc="Validation", disable=not accelerator.is_main_process):
                rec_samples, loss, log_dict_i = vae.validation_step(batch["pixel_values"].to(dtype=dtype))

                if accelerator.is_main_process:
                    if num_samples_to_log < 16:
                        samples_to_log.append(batch["pixel_values"])
                        rec_samples_to_log.append(rec_samples.to(dtype=torch.float32))
                        num_samples_to_log += rec_samples.shape[0]

                for k, v in log_dict_i.items():
                    if k not in log_dict:
                        log_dict[k] = 0.0
                    log_dict[k] += v
                num_samples += 1

            log_dict = accelerator.gather(log_dict)
            num_samples = torch.as_tensor(num_samples, device=loss.device)
            num_samples: torch.Tensor = accelerator.gather(num_samples)
            num_samples = num_samples.sum().cpu()

            log_dict_cpu = {k: (v.cpu().sum() / num_samples).item() for k, v in log_dict.items()}
            accelerator.log(
                {
                    f"val/{k}": v
                    for k, v in log_dict_cpu.items()
                },
                step=global_step,
            )

            if accelerator.is_main_process:
                samples_to_log = torch.cat(samples_to_log, dim=0)
                rec_samples_to_log = torch.cat(rec_samples_to_log, dim=0)

                samples_to_log = torch.clamp((samples_to_log + 1) / 2 * 255, min=0, max=255)
                rec_samples_to_log = torch.clamp((rec_samples_to_log + 1) / 2 * 255, min=0, max=255)

                # B, C, T, H, W -> B, T, C, H, W
                samples_to_log = samples_to_log.to(dtype=torch.uint8).transpose(1, 2).cpu().numpy()
                rec_samples_to_log = rec_samples_to_log.to(dtype=torch.uint8).transpose(1, 2).cpu().numpy()

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        tracker.writer.add_video(
                            "val/samples",
                            samples_to_log,
                            fps=4,
                            global_step=global_step,
                        )
                        tracker.writer.add_video(
                            "val/rec_samples",
                            rec_samples_to_log,
                            fps=4,
                            global_step=global_step,
                        )
                    elif tracker.name == "wandb":
                        import wandb

                        tracker.log(
                            {
                                "val/samples": [
                                    wandb.Video(video, fps=4, caption=f"#{i:02d}")
                                    for i, video in enumerate(samples_to_log)
                                ],
                                "val/rec_samples": [
                                    wandb.Video(video, fps=4, caption=f"#{i:02d}")
                                    for i, video in enumerate(rec_samples_to_log)
                                ],
                            },
                            step=global_step,
                        )

        if restore_training:
            vae.train()

        if self.runner_config.use_ema:
            ema_vae.restore(vae.parameters())

        return log_dict_cpu

    def setup_dataloader(self) -> tuple[DataLoader, DataLoader]:
        video_paths = []
        with open(self.data_config.dataset_name_or_path, "r") as f:
            for line in f:
                line: str = line.strip()
                if line:
                    video_paths.append(line.strip())

        random.shuffle(video_paths)

        # TODO: make this configurable
        train_video_paths = video_paths[:-4096]
        val_video_paths = video_paths[len(train_video_paths):]

        train_dataset = VideoDataset(
            train_video_paths,
            spatial_size=self.data_config.spatial_size,
            num_frames=self.data_config.num_frames,
            frame_intervals=self.data_config.frame_intervals,
            training=True,
        )

        val_dataset = VideoDataset(
            val_video_paths,
            spatial_size=self.data_config.spatial_size,
            num_frames=self.data_config.num_frames,
            frame_intervals=self.data_config.frame_intervals,
            training=False,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.runner_config.train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.runner_config.val_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        return train_dataloader, val_dataloader

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
            mid_block_attention_type=self.model_config.mid_block_attention_type,
            mid_block_num_attention_heads=self.model_config.mid_block_num_attention_heads,
            layers_per_block=self.model_config.layers_per_block,
            act_fn=self.model_config.act_fn,
            num_attention_heads=self.model_config.num_attention_heads,
            latent_channels=self.model_config.latent_channels,
            norm_num_groups=self.model_config.norm_num_groups,
            scaling_factor=self.model_config.scaling_factor,
            with_loss=True,
            lpips_model_name_or_path=self.model_config.lpips_model_name_or_path,
            init_logvar=self.model_config.init_logvar,
            reconstruction_loss_weight=self.model_config.reconstruction_loss_weight,
            perceptual_loss_weight=self.model_config.perceptual_loss_weight,
            nll_loss_weight=self.model_config.nll_loss_weight,
            kl_loss_weight=self.model_config.kl_loss_weight,
            discriminator_loss_weight=self.model_config.discriminator_loss_weight,
            disc_block_out_channels=self.model_config.disc_block_out_channels,
        )
        vae.train()

        if self.model_config.load_from_2d_vae is not None:
            state_dict = torch.load(self.model_config.load_from_2d_vae, map_location="cpu")["state_dict"]
            state_dict = inflate_params_from_2d_vae(vae.state_dict(), state_dict)
            missing_keys, unexpected_keys = vae.load_state_dict(state_dict, strict=False)

            missing_keys: list[str]
            missing_keys = filter(lambda x: not x.startswith("loss."), missing_keys)
            missing_keys = list(missing_keys)

            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                logger.warning(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}.")
                print(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}.")

        vae.loss.to(device=accelerator.device)

        # Create EMA for the vae.
        if self.runner_config.use_ema:
            params = list(filter(lambda x: x.requires_grad, vae.parameters()))
            ema_vae = EMAModel(copy.deepcopy(params), use_ema_warmup=True)
            ema_vae.to(accelerator.device)
        else:
            ema_vae = None

        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models: list, weights: list, output_dir: str):
            if accelerator.is_main_process:
                if self.runner_config.use_ema:
                    state_dict = ema_vae.state_dict()
                    model_path = os.path.join(output_dir, "ema_vae.bin")
                    torch.save(state_dict, model_path)

                # TODO: save vae

        def load_model_hook(models: list, input_dir: str):
            if self.runner_config.use_ema:
                model_path = os.path.join(input_dir, "ema_vae.bin")
                state_dict = torch.load(model_path, map_location=accelerator.device)
                ema_vae.load_state_dict(state_dict)

                # TODO: load vae

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        return vae, ema_vae
