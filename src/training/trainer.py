"""
Unified ControlNet Trainer for DSM2DTM.

Supports both pixel-loss (prediction_type="sample") and noise-loss
(prediction_type="epsilon") training through configurable loss strategies.
"""

import os
import math
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer

from ..utils.config import Config, config_to_dict
from ..utils.logging_utils import setup_logger
from ..utils.prompt import encode_prompt
from ..data.normalization import get_normalizer
from ..data.dataset import PairedDSMDataset, collate_fn
from ..data.split import load_split
from ..data.preprocessing import extract_index
from ..models.vae_modifier import load_vae_with_lora
from ..models.controlnet_loader import init_controlnet_from_unet
from ..losses.registry import get_loss
from .validation import run_validation, run_inference_preview


class ControlNetTrainer:
    """
    Unified trainer for ControlNet DSM→DTM conversion.

    Supports:
    - Pixel loss (MAE + Gradient) with prediction_type="sample"
    - Noise loss (MSE) with prediction_type="epsilon"
    - Configurable via YAML config
    - Validation with RMSE in real units
    - Best model tracking
    - Training history visualization
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device

        # Setup output
        os.makedirs(cfg.output.dir, exist_ok=True)
        self.logger = setup_logger(cfg.output.dir, prefix="training")
        self.logger.info(f"Configuration: {config_to_dict(cfg)}")

        # Seed
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Normalizer
        self.normalizer = get_normalizer(cfg)
        self.logger.info(f"Normalization: {cfg.data.normalization}")

        # Tracking
        self.train_loss_history = []
        self.val_rmse_history = []
        self.best_val_rmse = float("inf")
        self.global_step = 0

    def setup(self):
        """Load models, datasets, optimizer, etc."""
        cfg = self.cfg

        # --- Load Data ---
        self.logger.info("Loading dataset split...")
        if os.path.exists(cfg.data.split_json):
            split_data = load_split(cfg.data.split_json)
            train_pairs = split_data["train"]
            val_pairs = split_data["val"]
        else:
            self.logger.warning(f"Split file not found: {cfg.data.split_json}. Falling back to auto-scan.")
            train_pairs, val_pairs = self._auto_scan_split()

        self.logger.info(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

        self.train_ds = PairedDSMDataset(
            cfg.data.data_root, train_pairs, self.normalizer,
            cfg.data.dsm_dir, cfg.data.dtm_dir, cfg.data.crop_size,
        )
        self.val_ds = PairedDSMDataset(
            cfg.data.data_root, val_pairs, self.normalizer,
            cfg.data.dsm_dir, cfg.data.dtm_dir, cfg.data.crop_size,
        )

        self.train_loader = DataLoader(
            self.train_ds, batch_size=cfg.training.batch_size, shuffle=True,
            num_workers=cfg.training.num_workers, collate_fn=collate_fn,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=cfg.training.batch_size, shuffle=False,
            num_workers=cfg.training.num_workers, collate_fn=collate_fn,
        )

        # --- Load Models ---
        self.logger.info("Loading models...")

        # VAE
        self.vae = load_vae_with_lora(cfg.model.base_model, cfg.model.vae_path, self.device)
        self.vae.requires_grad_(False)
        self.vae.eval()

        # UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            cfg.model.base_model, subfolder="unet"
        ).to(self.device)
        self.unet.requires_grad_(False)
        self.unet.eval()

        # Text Encoder + Tokenizer
        self.text_encoder = CLIPTextModel.from_pretrained(
            cfg.model.base_model, subfolder="text_encoder"
        ).to(self.device)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        self.tokenizer = CLIPTokenizer.from_pretrained(
            cfg.model.base_model, subfolder="tokenizer"
        )

        # ControlNet
        if cfg.model.controlnet_path:
            self.logger.info(f"Loading ControlNet from {cfg.model.controlnet_path}...")
            self.controlnet = ControlNetModel.from_pretrained(cfg.model.controlnet_path).to(self.device)
        else:
            self.logger.info("Initializing ControlNet from UNet...")
            self.controlnet = init_controlnet_from_unet(cfg.model.base_model, self.device)
        self.controlnet.train()

        # Scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            cfg.model.base_model, subfolder="scheduler",
            prediction_type=cfg.model.prediction_type,
        )

        # Optimizer
        self.optimizer = AdamW(
            self.controlnet.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        # Loss
        self.loss_fn = get_loss(cfg.loss)
        self.logger.info(f"Loss function: {cfg.loss.type}")

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler(
            "cuda" if self.device.type == "cuda" else "cpu"
        )

        self.logger.info("Setup complete.")

    def _auto_scan_split(self):
        """Fallback: auto-scan directories and split 80/10/10."""
        from pathlib import Path

        cfg = self.cfg
        all_dsm = sorted([p.name for p in (Path(cfg.data.data_root) / cfg.data.dsm_dir).glob("*.TIF")])
        all_dtm = sorted([p.name for p in (Path(cfg.data.data_root) / cfg.data.dtm_dir).glob("*.TIF")])

        dsm_by_idx = {extract_index(f): f for f in all_dsm if extract_index(f) is not None}
        dtm_by_idx = {extract_index(f): f for f in all_dtm if extract_index(f) is not None}

        common = sorted(set(dsm_by_idx.keys()) & set(dtm_by_idx.keys()))
        paired = [(dsm_by_idx[i], dtm_by_idx[i]) for i in common]

        n = len(paired)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        return paired[:n_train], paired[n_train : n_train + n_val]

    def train_step(self, batch: dict) -> dict:
        """
        Execute a single training step.

        Returns dict with loss values.
        """
        cfg = self.cfg

        dsm_cond = batch["dsm"].to(self.device, dtype=torch.float32)
        dtm_target = batch["dtm"].to(self.device, dtype=torch.float32)
        bsz = dtm_target.shape[0]

        # Encode target to latent space
        with torch.no_grad():
            latents = self.vae.encode(dtm_target).latent_dist.sample()
            latents = latents * cfg.vae_scale_factor

        # Add noise
        noise = torch.randn_like(latents)
        t = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, t)

        # Prompt with dropout for CFG training
        if np.random.rand() < cfg.prompt_dropout_prob:
            prompt = ""
        else:
            prompt = cfg.inference.prompt

        encoder_hidden_states = encode_prompt(
            self.tokenizer, self.text_encoder, self.device, prompt, bsz
        )

        # Forward pass
        with torch.autocast(device_type=self.device.type, enabled=cfg.fp16):
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                sample=noisy_latents,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=dsm_cond,
                conditioning_scale=1.0,
                return_dict=False,
            )

            model_output = self.unet(
                noisy_latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # Compute loss based on prediction type
            if cfg.model.prediction_type == "sample":
                # model_output is predicted clean latents x0
                loss_dict = self.loss_fn(
                    pred_latents=model_output,
                    target_latents=latents,
                )
            else:
                # model_output is predicted noise
                loss_dict = self.loss_fn(
                    noise_pred=model_output,
                    noise=noise,
                )

        return loss_dict

    def train(self):
        """Run the full training loop."""
        cfg = self.cfg

        pbar = tqdm(total=cfg.training.total_steps)

        num_update_steps_per_epoch = math.ceil(
            len(self.train_loader) / cfg.training.gradient_accumulation_steps
        )
        num_epochs = math.ceil(cfg.training.total_steps / num_update_steps_per_epoch)
        self.logger.info(f"Total Epochs: {num_epochs} | Steps per Epoch: {num_update_steps_per_epoch}")

        current_epoch = 0
        epoch_loss = 0.0
        epoch_steps = 0

        while self.global_step < cfg.training.total_steps:
            current_epoch += 1
            self.logger.info(f"--- Starting Epoch {current_epoch} ---")

            for step, batch in enumerate(self.train_loader):
                if self.global_step >= cfg.training.total_steps:
                    break

                loss_dict = self.train_step(batch)
                loss = loss_dict["loss"]

                self.scaler.scale(loss).backward()
                epoch_loss += loss.item()
                epoch_steps += 1

                if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    pbar.update(1)

                    self.train_loss_history.append((self.global_step, loss.item()))

                    current_lr = self.optimizer.param_groups[0]["lr"]
                    pbar.set_description(f"Ep {current_epoch} | Loss: {loss.item():.4f}")

                    # Periodic logging
                    if self.global_step % cfg.training.log_every == 0:
                        loss_info = " | ".join(
                            f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                            for k, v in loss_dict.items() if k != "loss"
                        )
                        self.logger.info(
                            f"Step {self.global_step}/{cfg.training.total_steps} "
                            f"[Epoch {current_epoch}] - Loss: {loss.item():.6f} "
                            f"- LR: {current_lr:.2e} | {loss_info}"
                        )

                    # Save checkpoint
                    if self.global_step % cfg.training.save_every == 0:
                        save_path = os.path.join(cfg.output.dir, f"checkpoint-{self.global_step}")
                        self.controlnet.save_pretrained(save_path)
                        self.logger.info(f"Saved checkpoint to {save_path}")

                    # Inference preview
                    if cfg.inference.infer_every > 0 and self.global_step % cfg.inference.infer_every == 0:
                        self.logger.info(f"Running inference preview at step {self.global_step}...")

                        fixed_index = max(0, min(cfg.inference.fixed_infer_pair_index, len(self.val_ds) - 1))
                        fixed_sample = self.val_ds[fixed_index]
                        fixed_batch = collate_fn([fixed_sample] * cfg.inference.num_samples)

                        run_inference_preview(
                            cfg=cfg,
                            vae=self.vae,
                            unet=self.unet,
                            controlnet=self.controlnet,
                            tokenizer=self.tokenizer,
                            text_encoder=self.text_encoder,
                            noise_scheduler=self.noise_scheduler,
                            normalizer=self.normalizer,
                            batch=fixed_batch,
                            step=self.global_step,
                        )

                    # Validation
                    if self.global_step % cfg.training.val_every == 0:
                        self.logger.info("Running validation...")

                        mean_rmse = run_validation(
                            cfg=cfg,
                            vae=self.vae,
                            unet=self.unet,
                            controlnet=self.controlnet,
                            tokenizer=self.tokenizer,
                            text_encoder=self.text_encoder,
                            noise_scheduler=self.noise_scheduler,
                            val_loader=self.val_loader,
                            normalizer=self.normalizer,
                            logger=self.logger,
                        )

                        self.val_rmse_history.append((self.global_step, mean_rmse))
                        self.logger.info(
                            f">>> Validation Step {self.global_step}: "
                            f"Mean RMSE ({cfg.output.unit}) = {mean_rmse:.6f}"
                        )

                        # Save best model
                        if mean_rmse < self.best_val_rmse:
                            self.best_val_rmse = mean_rmse
                            best_save_path = os.path.join(cfg.output.dir, "best_model")
                            self.controlnet.save_pretrained(best_save_path)
                            self.logger.info(
                                f"!!! New BEST model (RMSE: {self.best_val_rmse:.6f}). "
                                f"Saved to {best_save_path}"
                            )

                        self.controlnet.train()

            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
            self.logger.info(f"End of Epoch {current_epoch} - Avg Loss: {avg_epoch_loss:.6f}")
            epoch_loss = 0.0
            epoch_steps = 0

        pbar.close()
        self.logger.info("Training finished successfully.")

        # Save final model
        self.controlnet.save_pretrained(os.path.join(cfg.output.dir, "final_model"))

        # Plot training history
        self._plot_history()

    def _plot_history(self):
        """Plot and save training loss / validation RMSE curves."""
        if not self.train_loss_history:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Training Loss
        ax1 = axes[0]
        train_steps, train_losses = zip(*self.train_loss_history)
        ax1.plot(train_steps, train_losses, label="Train Loss", alpha=0.6)
        if len(train_losses) > 50:
            ma_loss = np.convolve(train_losses, np.ones(50) / 50, mode="valid")
            ax1.plot(train_steps[49:], ma_loss, color="red", label="Moving Avg (50)")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"Training Loss ({self.cfg.loss.type})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Validation RMSE
        if self.val_rmse_history:
            ax2 = axes[1]
            val_steps, val_rmses = zip(*self.val_rmse_history)
            ax2.plot(val_steps, val_rmses, marker="o", color="orange", label=f"Val RMSE ({self.cfg.output.unit})")
            ax2.set_xlabel("Steps")
            ax2.set_ylabel(f"RMSE ({self.cfg.output.unit})")
            ax2.set_title(f"Validation RMSE (Best: {min(val_rmses):.4f})")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.cfg.output.dir, "training_history.png")
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Training history plot saved to {plot_path}")
