"""
Validation logic for DSM2DTM training.

Handles validation metric computation and inference previews during training.
"""

import os
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDPMScheduler

from ..utils.prompt import encode_prompt
from ..data.normalization import NormalizationStrategy


@torch.no_grad()
def run_validation(
    cfg,
    vae,
    unet,
    controlnet,
    tokenizer,
    text_encoder,
    noise_scheduler,
    val_loader,
    normalizer: NormalizationStrategy,
    logger=None,
):
    """
    Run validation and compute RMSE on the validation set.

    For pixel-loss (prediction_type="sample"):
        - Forward pass predicts clean latents x0
        - Decode x0 to pixel space, denormalize, compute RMSE in real units

    For noise-loss (prediction_type="epsilon"):
        - Use scheduler.step() to get the clean latent estimate (FIXED from original bug)
        - Decode and compute RMSE

    Returns:
        Mean RMSE across validation batches.
    """
    controlnet.eval()
    rmses = []

    prediction_type = cfg.model.prediction_type

    for val_batch in val_loader:
        dsm_val = val_batch["dsm"].to(cfg.device, dtype=torch.float32)
        dtm_val = val_batch["dtm"].to(cfg.device, dtype=torch.float32)
        bsz_val = dtm_val.shape[0]

        lat_val = vae.encode(dtm_val).latent_dist.sample() * cfg.vae_scale_factor
        noise_val = torch.randn_like(lat_val)
        t_val = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz_val,), device=cfg.device
        ).long()
        noisy_lat_val = noise_scheduler.add_noise(lat_val, noise_val, t_val)

        enc_val = encode_prompt(
            tokenizer, text_encoder, cfg.device, cfg.inference.prompt, bsz_val
        )

        with torch.autocast(device_type=cfg.device.type, enabled=cfg.fp16):
            down_res_val, mid_res_val = controlnet(
                sample=noisy_lat_val,
                timestep=t_val,
                encoder_hidden_states=enc_val,
                controlnet_cond=dsm_val,
                conditioning_scale=1.0,
                return_dict=False,
            )

            model_output = unet(
                noisy_lat_val,
                t_val,
                encoder_hidden_states=enc_val,
                down_block_additional_residuals=down_res_val,
                mid_block_additional_residual=mid_res_val,
            ).sample

            # Get clean latent estimate based on prediction type
            if prediction_type == "sample":
                # Model directly predicts x0
                lat_est_val = model_output
            else:
                # Model predicts noise (epsilon) — use scheduler formula to recover x0
                # x0 = (x_t - sqrt(1-alpha_bar) * epsilon) / sqrt(alpha_bar)
                alpha_bar = noise_scheduler.alphas_cumprod.to(cfg.device)[t_val]
                alpha_bar = alpha_bar.view(-1, 1, 1, 1)
                lat_est_val = (noisy_lat_val - torch.sqrt(1 - alpha_bar) * model_output) / torch.sqrt(alpha_bar)

            recon_val = vae.decode(lat_est_val / cfg.vae_scale_factor).sample
            recon_val = recon_val.clamp(-1.0, 1.0)

            # Denormalize to real units for metric calculation
            pred_real = normalizer.denormalize_tensor(recon_val)
            target_real = normalizer.denormalize_tensor(dtm_val)

            rmse_val = torch.sqrt(F.mse_loss(pred_real, target_real))

        rmses.append(rmse_val.item())

    mean_rmse = np.mean(rmses) if len(rmses) else float("nan")
    controlnet.train()

    return mean_rmse


@torch.no_grad()
def run_inference_preview(
    cfg,
    vae,
    unet,
    controlnet,
    tokenizer,
    text_encoder,
    noise_scheduler,
    normalizer: NormalizationStrategy,
    batch: dict,
    step: int,
):
    """
    Run inference on a fixed batch for visualization during training.

    Saves:
    - Original DSM/DTM images (once)
    - Predicted DTM image
    - Error map
    """
    out_dir = os.path.join(cfg.output.dir, "inference")
    os.makedirs(out_dir, exist_ok=True)

    controlnet.eval()
    unet.eval()
    vae.eval()

    def to_vis(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-6:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    # Save originals once
    original_dsm_path = os.path.join(out_dir, "original_image_dsm.png")
    original_dtm_path = os.path.join(out_dir, "original_image_dtm.png")

    if not os.path.exists(original_dsm_path) or not os.path.exists(original_dtm_path):
        dsm_vis = batch["dsm"][0].detach().cpu().numpy()
        dtm_vis = batch["dtm"][0].detach().cpu().numpy()

        dsm_vis_01 = to_vis(dsm_vis[0])
        dtm_vis_01 = to_vis(dtm_vis[0])

        dsm_u16 = (dsm_vis_01 * 65535.0).round().clip(0, 65535).astype(np.uint16)
        dtm_u16 = (dtm_vis_01 * 65535.0).round().clip(0, 65535).astype(np.uint16)

        if not os.path.exists(original_dsm_path):
            Image.fromarray(dsm_u16, mode="I;16").save(original_dsm_path)
        if not os.path.exists(original_dtm_path):
            Image.fromarray(dtm_u16, mode="I;16").save(original_dtm_path)

    num_samples = cfg.inference.num_samples
    dsm_cond = batch["dsm"][:num_samples].to(cfg.device, dtype=torch.float32)
    bsz = dsm_cond.shape[0]

    encoder_hidden_states = encode_prompt(
        tokenizer, text_encoder, cfg.device, cfg.inference.prompt, bsz
    )

    latents = torch.randn(
        (bsz, unet.config.in_channels, cfg.data.crop_size // 8, cfg.data.crop_size // 8),
        device=cfg.device,
        dtype=torch.float32,
    )
    latents = latents * noise_scheduler.init_noise_sigma

    infer_scheduler = DDPMScheduler.from_config(noise_scheduler.config)
    infer_scheduler.set_timesteps(cfg.inference.num_steps, device=cfg.device)

    for t in infer_scheduler.timesteps:
        with torch.autocast(device_type=cfg.device.type, enabled=cfg.fp16):
            down_block_res_samples, mid_block_res_sample = controlnet(
                sample=latents,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=dsm_cond,
                conditioning_scale=1.0,
                return_dict=False,
            )

            noise_pred = unet(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

        step_out = infer_scheduler.step(noise_pred, t, latents)
        latents = step_out.prev_sample

    recon = vae.decode(latents / cfg.vae_scale_factor).sample
    recon = recon.clamp(-1, 1)

    recon_np = recon.detach().cpu().numpy()  # [B, 1, H, W]
    target_np = batch["dtm"][:bsz].detach().cpu().numpy()

    recon_raw = recon.squeeze(1).detach().cpu().numpy()  # [B, H, W]

    for i in range(bsz):
        # Save preview
        img_vis_01 = to_vis(recon_raw[i])
        img_u16 = (img_vis_01 * 65535.0).round().clip(0, 65535).astype(np.uint16)
        save_path = os.path.join(out_dir, f"step_{step}_sample_{i}.png")
        Image.fromarray(img_u16, mode="I;16").save(save_path)

        # Save error map
        pred_h = normalizer.denormalize(recon_np[i, 0], data_type="dtm")
        target_h = normalizer.denormalize(target_np[i, 0], data_type="dtm")
        diff = np.abs(pred_h - target_h)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        vmax_val = max(20.0, np.percentile(diff, 95))
        plt.imshow(diff, cmap="jet", vmin=0, vmax=vmax_val)
        error_path = os.path.join(out_dir, f"step_{step}_sample_{i}_error.png")
        plt.savefig(error_path, bbox_inches="tight", dpi=100)
        plt.close()

    controlnet.train()
