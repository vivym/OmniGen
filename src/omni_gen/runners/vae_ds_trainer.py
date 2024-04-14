import copy
import json
import os
import tempfile
from pathlib import Path

import deepspeed
import ray.train
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.accelerator import get_accelerator
from deepspeed.monitor.monitor import MonitorMaster
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.lr_schedules import WarmupCosineLR
from einops import rearrange
from ray.train import (
    Checkpoint, CheckpointConfig, FailureConfig, ScalingConfig, SyncConfig, RunConfig
)
from ray.train.torch import TorchTrainer
from safetensors import safe_open
from torch.utils.data import DataLoader
from tqdm import tqdm

import omni_gen.utils.distributed as dist
from omni_gen.data.streaming_image_dataset import StreamingImageDataset
from omni_gen.data.streaming_video_dataset import StreamingVideoDataset
from omni_gen.models.video_vae import AutoencoderKL, Discriminator2D, Discriminator3D, LPIPSMetric
from omni_gen.utils import logging
from omni_gen.utils.distributions import DiagonalGaussianDistribution
from omni_gen.utils.ema import EMAModel
from omni_gen.utils.inflation import inflate_params_from_2d_vae
from .deepspeed_runner import DeepSpeedRunner

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class VAEDeepSpeedTrainer(DeepSpeedRunner):
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

        train_dataloader, val_dataloader = self.setup_datasets()

        vae, ema_vae, lpips_metric, discriminator = self.setup_models()

        total_batch_size = (
            self.runner_config.train_batch_size
            * self.runner_config.num_devices
            * self.runner_config.gradient_accumulation_steps
        )

        if self.runner_config.gradient_clipping is None:
            gradient_clipping = 0.
        else:
            gradient_clipping = self.runner_config.gradient_clipping
        deepspeed_config = {
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.optimizer_config.learning_rate,
                    "betas": self.optimizer_config.adam_beta,
                    "eps": self.optimizer_config.adam_epsilon,
                    "weight_decay": self.optimizer_config.weight_decay,
                },
            },
            "gradient_accumulation_steps": self.runner_config.gradient_accumulation_steps,
            "gradient_clipping": gradient_clipping,
            "train_batch_size": total_batch_size,
            "train_micro_batch_size_per_gpu": self.runner_config.train_batch_size,
            "wall_clock_breakdown": False,
            "wandb": {
                "enabled": self.runner_config.log_with == "wandb",
                "project": self.runner_config.name,
            },
        }

        if self.optimizer_config.lr_scheduler == "WarmupCosineLR":
            deepspeed_config["scheduler"] = {
                "type": "WarmupCosineLR",
                "params": {
                    "total_num_steps": self.runner_config.max_steps,
                    "warmup_min_ratio": 0,
                    "warmup_num_steps": self.optimizer_config.lr_warmup_steps,
                },
            }
        elif self.optimizer_config.lr_scheduler == "WarmupLR":
            deepspeed_config["scheduler"] = {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.optimizer_config.learning_rate,
                    "warmup_num_steps": self.optimizer_config.lr_warmup_steps,
                },
            }
        else:
            raise ValueError(f"Invalid lr_scheduler: {self.optimizer_config.lr_scheduler}")

        if self.runner_config.mixed_precision is not None:
            if self.runner_config.mixed_precision == "fp16":
                deepspeed_config["fp16"] = {"enabled": True}
                dtype = torch.float16
            elif self.runner_config.mixed_precision == "bf16":
                deepspeed_config["bf16"] = {"enabled": True}
                dtype = torch.bfloat16
            elif self.runner_config.mixed_precision == "no":
                dtype = torch.float32
            else:
                raise ValueError(f"Invalid mixed precision mode: {self.runner_config.mixed_precision}")
        else:
            dtype = torch.float32

        if self.runner_config.zero_stage is not None:
            deepspeed_config["zero_optimization"] = {
                "stage": self.runner_config.zero_stage,
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "contiguous_gradients": True,
            }

        if self.runner_config.hf_ds_config_path is not None:
            with open(self.runner_config.hf_ds_config_path, "r") as f:
                deepspeed_config.update(json.load(f))

        vae_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=vae,
            model_parameters=vae.parameters(),
            config=deepspeed_config,
        )
        vae_engine: deepspeed.DeepSpeedEngine
        optimizer: torch.optim.Optimizer
        lr_scheduler: WarmupCosineLR

        disc_deepspeed_config = copy.deepcopy(deepspeed_config)
        disc_deepspeed_config["wandb"]["enabled"] = False
        discriminator_engine, optimizer_disc, _, _ = deepspeed.initialize(
            model=discriminator,
            model_parameters=discriminator.parameters(),
            config=disc_deepspeed_config,
        )
        discriminator_engine: deepspeed.DeepSpeedEngine
        optimizer_disc: torch.optim.Optimizer

        device = get_accelerator().device_name(vae_engine.local_rank)
        is_main_process = vae_engine.global_rank == 0
        is_local_main_process = vae_engine.local_rank == 0

        lpips_metric.to(device=device)
        ema_vae.to(device=device)

        ds_config = DeepSpeedConfig(deepspeed_config)
        monitor = MonitorMaster(ds_config.monitor_config)

        if is_main_process:
            configs = {}
            for prefix, config in [
                ("data", self.data_config),
                ("model", self.model_config),
                ("optimizer", self.optimizer_config),
                ("runner", self.runner_config),
            ]:
                for k, v in config.__dict__.items():
                    configs[f"{prefix}_{k}"] = v

            if monitor.wandb_monitor is not None:
                import wandb

                wandb.config.update(configs)

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
                # TODO: check return values (`client_state`)
                _, state = vae_engine.load_checkpoint(checkpoint_dir)
                if "global_step" in state:
                    global_step = state["global_step"]
                    initial_global_step = global_step

                if self.runner_config.use_ema:
                    ema_ckpt_path = os.path.join(checkpoint_dir, "ema_vae.bin")
                    if os.path.exists(ema_ckpt_path):
                        ema_vae.load_state_dict(torch.load(ema_ckpt_path, map_location=device))

                disc_ckpt_path = os.path.join(checkpoint_dir, "discriminator")
                if os.path.exists(disc_ckpt_path):
                    discriminator_engine.load_checkpoint(checkpoint_dir)

        progress_bar = tqdm(
            range(0, self.runner_config.max_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not is_local_main_process,
        )

        # TODO: skip initial steps if resuming from checkpoint

        done = False
        while not done:
            vae_engine.train()
            discriminator_engine.train()

            log_dict: dict[str, torch.Tensor] = {}
            log_dict_denom: dict[str, int] = {}

            for batch in train_dataloader:
                batch: dict[str, torch.Tensor]
                samples = batch["pixel_values"].to(device=device, dtype=dtype, non_blocking=True)

                optimizer.zero_grad()

                posteriors, rec_samples = vae_engine(samples)

                loss_ae, loss_nll, log_dict_ae = self.compute_ae_loss(
                    samples=samples,
                    posteriors=posteriors,
                    rec_samples=rec_samples,
                    lpips_metric=lpips_metric,
                )

                if global_step >= self.runner_config.discriminator_start_steps:
                    for param in discriminator.parameters():
                        param.requires_grad = False

                    loss_gen, log_dict_gen = self.compute_generator_loss(
                        rec_samples=rec_samples,
                        loss_nll=loss_nll,
                        discriminator=discriminator,
                        last_layer_weight=vae.decoder.conv_out.weight,
                    )

                    for param in discriminator.parameters():
                        param.requires_grad = True
                else:
                    loss_gen = 0.
                    log_dict_gen = {}

                loss = loss_ae + loss_gen

                vae_engine.backward(loss)
                vae_engine.step()

                if global_step >= self.runner_config.discriminator_start_steps:
                    optimizer_disc.zero_grad()

                    loss_disc, log_dict_disc = self.compute_discriminator_loss(
                        samples=samples,
                        rec_samples=rec_samples,
                        discriminator=discriminator,
                    )

                    discriminator_engine.backward(loss_disc)
                    discriminator_engine.step()
                else:
                    log_dict_disc = {}

                for log_dict_i in [log_dict_ae, log_dict_gen, log_dict_disc]:
                    log_dict_i: dict[str, torch.Tensor]
                    for k, v in log_dict_i.items():
                        log_dict[k] = log_dict.get(k, 0) + v
                        log_dict_denom[k] = log_dict_denom.get(k, 0) + 1

                if vae_engine.is_gradient_accumulation_boundary():
                    if self.runner_config.use_ema:
                        params = list(vae.parameters()) + list(discriminator.parameters())
                        ema_vae.step(params)

                    progress_bar.update(1)
                    global_step += 1

                    if global_step % self.runner_config.log_every_n_steps == 0:
                        gathered_log_dict: dict[str, torch.Tensor] = dist.gather(log_dict, dst=0)

                        if is_main_process:
                            log_dict: dict[str, torch.Tensor] = {}
                            for k, v in gathered_log_dict.items():
                                log_dict[k] = v.mean().cpu().item() / log_dict_denom[k]

                            logs = {"lr": lr_scheduler.get_last_lr()[0]}
                            logs.update(log_dict)
                            progress_bar.set_postfix(**logs)

                            events = [
                                (f"train/{k}", v, vae_engine.global_samples)
                                for k, v in log_dict.items()
                            ]
                            monitor.write_events(events)

                        log_dict: dict[str, torch.Tensor] = {}
                        log_dict_denom: dict[str, int] = {}
                    else:
                        progress_bar.set_postfix()

                do_validation = (
                    vae_engine.is_gradient_accumulation_boundary() and \
                    self.runner_config.validation_every_n_steps is not None and \
                    global_step % self.runner_config.validation_every_n_steps == 0
                )

                do_checkpointing = (
                    vae_engine.is_gradient_accumulation_boundary() and \
                    self.runner_config.checkpointing_every_n_steps is not None and \
                    global_step % self.runner_config.checkpointing_every_n_steps == 0
                )

                if do_validation or do_checkpointing:
                    val_metrics = self.validation_loop(
                        val_dataloader=val_dataloader,
                        vae_engine=vae_engine,
                        vae=vae,
                        ema_vae=ema_vae,
                        discriminator=discriminator,
                        lpips_metric=lpips_metric,
                        monitor=monitor,
                        device=device,
                        dtype=dtype,
                        is_main_process=is_main_process,
                        is_local_main_process=is_local_main_process,
                    )
                    val_metrics["global_step"] = global_step

                if do_checkpointing:
                    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                        vae_engine.save_checkpoint(
                            temp_checkpoint_dir,
                            client_state={"global_step": global_step},
                        )

                        if self.runner_config.use_ema:
                            ema_ckpt_path = os.path.join(temp_checkpoint_dir, "ema_vae.bin")
                            torch.save(ema_vae.state_dict(), ema_ckpt_path)

                        disc_ckpt_path = os.path.join(temp_checkpoint_dir, "discriminator")
                        discriminator_engine.save_checkpoint(
                            disc_ckpt_path,
                            client_state={"global_step": global_step},
                        )

                        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                        ray.train.report(val_metrics, checkpoint=checkpoint)

                        if is_main_process:
                            logger.info(f"Saved state to {temp_checkpoint_dir}")

                if global_step >= self.runner_config.max_steps:
                    done = True
                    break

            do_validation = self.runner_config.checkpointing_every_n_steps is None

            do_checkpointing = self.runner_config.checkpointing_every_n_steps is None

            if do_validation or do_checkpointing:
                val_metrics = self.validation_loop(
                    val_dataloader=val_dataloader,
                    vae_engine=vae_engine,
                    vae=vae,
                    ema_vae=ema_vae,
                    discriminator=discriminator,
                    lpips_metric=lpips_metric,
                    monitor=monitor,
                    device=device,
                    dtype=dtype,
                    is_main_process=is_main_process,
                    is_local_main_process=is_local_main_process,
                )
                val_metrics["global_step"] = global_step

            if do_checkpointing:
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    vae_engine.save_checkpoint(temp_checkpoint_dir)

                    if self.runner_config.use_ema:
                        ema_ckpt_path = os.path.join(temp_checkpoint_dir, "ema_vae.bin")
                        torch.save(ema_vae.state_dict(), ema_ckpt_path)

                    disc_ckpt_path = os.path.join(temp_checkpoint_dir, "discriminator")
                    discriminator_engine.save_checkpoint(disc_ckpt_path)

                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                    ray.train.report(val_metrics, checkpoint=checkpoint)

                    logger.info(f"Saved state to {temp_checkpoint_dir}")

    def validation_loop(
        self,
        val_dataloader: DataLoader,
        vae_engine: deepspeed.DeepSpeedEngine,
        vae: AutoencoderKL,
        ema_vae: EMAModel | None,
        discriminator: Discriminator2D | Discriminator3D,
        lpips_metric: LPIPSMetric,
        monitor: MonitorMaster,
        device: torch.device,
        dtype: torch.dtype,
        is_main_process: bool,
        is_local_main_process: bool,
    ) -> dict[str, float]:
        logger.info("Running validation...")

        if self.runner_config.use_ema:
            # Store the VAE parameters temporarily and load the EMA parameters to perform inference.
            params = list(vae.parameters()) + list(discriminator.parameters())
            ema_vae.store(params)
            ema_vae.copy_to(params)

        if vae.training:
            vae_engine.eval()
            restore_training = True
        else:
            restore_training = False

        num_samples_to_log = 0
        samples_to_log = []
        rec_samples_to_log = []
        log_dict: dict[str, torch.Tensor] = {}
        num_batches = 0

        with torch.inference_mode():
            for batch in tqdm(val_dataloader, desc="Validation", disable=not is_local_main_process):
                batch: dict[str, torch.Tensor]
                samples = batch["pixel_values"].to(device=device, dtype=dtype, non_blocking=True)

                posteriors, rec_samples = vae_engine(samples)
                rec_samples: torch.Tensor

                _, _, log_dict_i = self.compute_ae_loss(
                    samples=samples,
                    posteriors=posteriors,
                    rec_samples=rec_samples,
                    lpips_metric=lpips_metric,
                )

                if is_main_process:
                    if num_samples_to_log < 64:
                        samples_to_log.append(samples)
                        rec_samples_to_log.append(rec_samples.to(dtype=torch.float32))
                        num_samples_to_log += rec_samples.shape[0]

                for k, v in log_dict_i.items():
                    if k not in log_dict:
                        log_dict[k] = 0.0
                    log_dict[k] += v

                num_batches += 1

            gathered_log_dict: dict[str, torch.Tensor] = dist.gather(log_dict, dst=0)
            if is_main_process:
                log_dict: dict[str, torch.Tensor] = {}
                for k, v in gathered_log_dict.items():
                    log_dict[k] = v.mean().cpu().item() / num_batches

                events =[
                    (f"val/{k}", v, vae_engine.global_samples)
                    for k, v in log_dict.items()
                ]
                monitor.write_events(events)

                samples_to_log = torch.cat(samples_to_log, dim=0)
                rec_samples_to_log = torch.cat(rec_samples_to_log, dim=0)

                if self.model_config.image_mode:
                    samples_to_log = samples_to_log[:, :, None]
                    rec_samples_to_log = rec_samples_to_log[:, :, None]

                samples_to_log = torch.clamp((samples_to_log + 1) / 2 * 255, min=0, max=255)
                rec_samples_to_log = torch.clamp((rec_samples_to_log + 1) / 2 * 255, min=0, max=255)

                # B, C, T, H, W -> B, T, C, H, W
                samples_to_log = samples_to_log.to(dtype=torch.uint8).transpose(1, 2).cpu().numpy()
                rec_samples_to_log = rec_samples_to_log.to(dtype=torch.uint8).transpose(1, 2).cpu().numpy()

                if monitor.wandb_monitor is not None:
                    import wandb

                    monitor.wandb_monitor.log(
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
                        step=vae_engine.global_samples,
                    )

                if monitor.tb_monitor is not None:
                    monitor.tb_monitor.summary_writer.add_video(
                        "val/samples",
                        samples_to_log,
                        fps=4,
                        global_step=vae_engine.global_samples,
                    )
                    monitor.tb_monitor.summary_writer.add_video(
                        "val/rec_samples",
                        rec_samples_to_log,
                        fps=4,
                        global_step=vae_engine.global_samples,
                    )
            else:
                log_dict = {}

        if restore_training:
            vae_engine.train()

        if self.runner_config.use_ema:
            ema_vae.restore(params)

        return log_dict

    def compute_ae_loss(
        self,
        samples: torch.Tensor,
        posteriors: DiagonalGaussianDistribution,
        rec_samples: torch.Tensor,
        lpips_metric: LPIPSMetric,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.model_config.image_mode:
            samples = samples[:, :, None]
            rec_samples = rec_samples[:, :, None]

        flattend_samples = rearrange(samples, "b c t h w -> (b t) c h w")
        flattend_rec_samples = rearrange(rec_samples, "b c t h w -> (b t) c h w")

        reconstruction_loss_type = self.model_config.reconstruction_loss_type
        if reconstruction_loss_type == "l1":
            loss_rec = torch.abs(flattend_samples - flattend_rec_samples)
        elif reconstruction_loss_type == "l2":
            loss_rec = (flattend_samples - flattend_rec_samples) ** 2
        else:
            raise ValueError(f"Invalid reconstruction loss type: {reconstruction_loss_type}")

        loss_perceptual = lpips_metric(flattend_samples, flattend_rec_samples)

        loss_nll = (
            self.model_config.reconstruction_loss_weight * loss_rec
            + self.model_config.perceptual_loss_weight * loss_perceptual
        )
        loss_nll = loss_nll.sum() / loss_nll.shape[0]

        loss_kl = posteriors.kl()
        loss_kl = loss_kl.sum() / loss_kl.shape[0]

        loss = (
            self.model_config.nll_loss_weight * loss_nll
            + self.model_config.kl_loss_weight * loss_kl
        )

        return loss, loss_nll, {
            "loss_ae": loss.detach(),
            "loss_rec": loss_rec.detach().mean(),
            "loss_perceptual": loss_perceptual.detach().mean(),
            "loss_nll": loss_nll.detach(),
            "loss_kl": loss_kl.detach(),
        }

    def compute_generator_loss(
        self,
        rec_samples: torch.Tensor,
        loss_nll: torch.Tensor,
        discriminator: Discriminator2D | Discriminator3D,
        last_layer_weight: nn.Parameter,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.model_config.image_mode:
            rec_samples = rec_samples[:, :, None]

        log_dict: dict[str, torch.Tensor] = {}

        if rec_samples.shape[2] == 1:
            rec_samples_2d = rec_samples[:, :, 0]

            logits_fake_2d = discriminator(rec_samples_2d)

            loss_g_2d = -torch.mean(logits_fake_2d)
            log_dict["loss_g_2d"] = loss_g_2d.detach()

            disc_weight_2d = compute_adaptive_disc_weight(
                loss_nll, loss_g_2d, last_layer_weight
            )
            log_dict["disc_weight_2d"] = disc_weight_2d
            disc_weight_2d = disc_weight_2d * self.model_config.discriminator_loss_weight

            loss = loss_g_2d * disc_weight_2d
        else:
            logits_fake_3d = discriminator(rec_samples)

            loss_g_3d = -torch.mean(logits_fake_3d)
            log_dict["loss_g_3d"] = loss_g_3d.detach()

            disc_weight_3d = compute_adaptive_disc_weight(
                loss_nll, loss_g_3d, last_layer_weight
            )
            log_dict["disc_weight_3d"] = disc_weight_3d
            disc_weight_3d = disc_weight_3d * self.model_config.discriminator_loss_weight

            loss = loss_g_3d * disc_weight_3d

        return loss, log_dict

    def compute_discriminator_loss(
        self,
        samples: torch.Tensor,
        rec_samples: torch.Tensor,
        discriminator: Discriminator2D | Discriminator3D,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.model_config.image_mode:
            samples = samples[:, :, None]
            rec_samples = rec_samples[:, :, None]

        log_dict: dict[str, torch.Tensor] = {}

        if samples.shape[2] == 1:
            samples_2d = samples[:, :, 0].detach()
            rec_samples_2d = rec_samples[:, :, 0].detach()

            logits_real_2d = discriminator(samples_2d)
            logits_fake_2d = discriminator(rec_samples_2d)

            loss_real_2d = torch.mean(F.relu(1. - logits_real_2d))
            loss_fake_2d = torch.mean(F.relu(1. + logits_fake_2d))
            loss_d_2d = (loss_real_2d + loss_fake_2d) * 0.5

            log_dict["loss_real_2d"] = loss_real_2d.detach()
            log_dict["loss_fake_2d"] = loss_fake_2d.detach()
            log_dict["loss_d_2d"] = loss_d_2d.detach()

            loss = loss_d_2d
        else:
            logits_real_3d = discriminator(samples.detach())
            logits_fake_3d = discriminator(rec_samples.detach())

            loss_real_3d = torch.mean(F.relu(1. - logits_real_3d))
            loss_fake_3d = torch.mean(F.relu(1. + logits_fake_3d))
            loss_d_3d = (loss_real_3d + loss_fake_3d) * 0.5

            log_dict["loss_real_3d"] = loss_real_3d.detach()
            log_dict["loss_fake_3d"] = loss_fake_3d.detach()
            log_dict["loss_d_3d"] = loss_d_3d.detach()

            loss = loss_d_3d

        return loss, log_dict

    def setup_datasets(self) -> tuple[DataLoader, DataLoader]:
        with open(self.data_config.dataset_name_or_path, "r") as f:
            meta = json.load(f)

        if self.model_config.image_mode:
            train_dataset = StreamingImageDataset(
                meta["train"][0][0],    # TODO: support multiple datasets
                spatial_size=self.data_config.spatial_size,
                training=True,
            )

            val_dataset = StreamingImageDataset(
                meta["val"][0][0],      # TODO: support multiple datasets
                spatial_size=self.data_config.spatial_size,
                training=False,
            )
        else:
            train_dataset = StreamingVideoDataset(
                meta["train"][0][0],
                spatial_size=self.data_config.spatial_size,
                num_frames=self.data_config.num_frames,
                frame_intervals=self.data_config.frame_intervals,
                training=True,
            )

            val_dataset = StreamingVideoDataset(
                meta["val"][0][0],
                spatial_size=self.data_config.spatial_size,
                num_frames=self.data_config.num_frames,
                frame_intervals=self.data_config.frame_intervals,
                training=False,
            )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.runner_config.train_batch_size,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.runner_config.val_batch_size,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        return train_dataloader, val_dataloader

    def setup_models(self) -> tuple[AutoencoderKL, EMAModel | None, LPIPSMetric, Discriminator2D | Discriminator3D]:
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
            image_mode=self.model_config.image_mode,
        )
        vae.train()

        if self.model_config.load_from_3d_vae is not None:
            if self.model_config.load_from_3d_vae.endswith(".safetensors"):
                state_dict = {}
                with safe_open(self.model_config.load_from_3d_vae, framework="pt") as f:
                    for k in f.keys():
                        state_dict[k] = f.get_tensor(k)
            else:
                state_dict = torch.load(self.model_config.load_from_3d_vae, map_location="cpu")
        elif self.model_config.load_from_2d_vae is not None:
            if self.model_config.load_from_2d_vae.endswith(".safetensors"):
                state_dict = {}
                with safe_open(self.model_config.load_from_2d_vae, framework="pt") as f:
                    for k in f.keys():
                        state_dict[k] = f.get_tensor(k)
            else:
                state_dict = torch.load(self.model_config.load_from_2d_vae, map_location="cpu")["state_dict"]
            state_dict = inflate_params_from_2d_vae(
                vae.state_dict(), state_dict, image_mode=self.model_config.image_mode
            )
        else:
            state_dict = None

        if state_dict is not None:
            missing_keys, unexpected_keys = vae.load_state_dict(state_dict, strict=False)

            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                logger.warning(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}.")
                print(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}.")

        lpips_metric = LPIPSMetric.from_pretrained(self.model_config.lpips_model_name_or_path)

        if self.model_config.image_mode:
            discriminator = Discriminator2D(
                in_channels=self.model_config.in_channels,
                block_out_channels=self.model_config.disc_block_out_channels,
            )
        else:
            discriminator = Discriminator3D(
                in_channels=self.model_config.in_channels,
                block_out_channels=self.model_config.disc_block_out_channels,
            )

            if self.model_config.load_from_3d_discriminator is not None:
                if self.model_config.load_from_3d_discriminator.endswith(".safetensors"):
                    state_dict = {}
                    with safe_open(self.model_config.load_from_3d_discriminator, framework="pt") as f:
                        for k in f.keys():
                            state_dict[k] = f.get_tensor(k)
                else:
                    state_dict = torch.load(self.model_config.load_from_3d_discriminator, map_location="cpu")
            else:
                state_dict = None

            if state_dict is not None:
                missing_keys, unexpected_keys = discriminator.load_state_dict(state_dict, strict=False)

                if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                    logger.warning(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}.")
                    print(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}.")

        # Create EMA for the vae and discriminator.
        if self.runner_config.use_ema:
            params = list(vae.parameters()) + list(discriminator.parameters())
            ema_vae = EMAModel(copy.deepcopy(params), use_ema_warmup=True)
        else:
            ema_vae = None

        return vae, ema_vae, lpips_metric, discriminator


def compute_adaptive_disc_weight(
    loss_nll: torch.Tensor, loss_g: torch.Tensor, last_layer_weight: nn.Parameter
) -> torch.Tensor:
    return torch.tensor(10000., dtype=loss_nll.dtype, device=loss_nll.device)

    nll_grads = torch.autograd.grad(loss_nll, last_layer_weight, retain_graph=True)[0]
    g_grads = torch.autograd.grad(loss_g, last_layer_weight, retain_graph=True)[0]

    with torch.no_grad():
        disc_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        disc_weight = torch.clamp(disc_weight, min=0.0, max=1e4)

    return disc_weight
