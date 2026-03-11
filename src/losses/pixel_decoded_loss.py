"""
Pixel-space decoded loss for DSM2DTM.

Decodes predicted latents through frozen VAE to pixel space [B,1,H,W],
then computes MAE + Sobel gradient loss on actual elevation values.

Uses VAE gradient checkpointing + slicing to fit in 24GB VRAM.
Gradients flow through the VAE decode back to ControlNet.

Used when prediction_type="sample" (model predicts x0, not noise).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_loss
from .pixel_loss import sobel_gradients, gradient_loss


@register_loss("pixel_decoded")
class PixelDecodedLoss(nn.Module):
    """
    Pixel-level loss computed in decoded pixel space.

    Decodes predicted clean latents through the frozen VAE decoder to
    pixel space [B, 1, H, W], then computes MAE + Sobel gradient loss
    on actual elevation values rather than abstract latent channels.

    Memory optimizations (enabled automatically via set_vae()):
    - VAE gradient checkpointing: discards intermediate activations during
      forward, recomputes during backward. Reduces ~50GB to ~0.9GB.
    - VAE slicing: decodes batch images one at a time internally.

    Loss = mae_weight * L1(pred_decoded, target_decoded)
         + grad_weight * GradientLoss(pred_decoded, target_decoded)
    """

    def __init__(self, mae_weight: float = 1.0, grad_weight: float = 0.5, **kwargs):
        super().__init__()
        self.mae_weight = mae_weight
        self.grad_weight = grad_weight
        self._vae = None
        self._vae_scale_factor = 0.18215

    def set_vae(self, vae, vae_scale_factor: float):
        """
        Provide VAE reference and enable memory optimizations.

        Called by the trainer after loss construction. Enables gradient
        checkpointing and slicing on the VAE for memory-efficient
        differentiable decoding.

        Args:
            vae: The frozen AutoencoderKL instance.
            vae_scale_factor: Latent scaling factor (default 0.18215).
        """
        self._vae = vae
        self._vae_scale_factor = vae_scale_factor

        # Enable memory optimizations for differentiable decode.
        # Gradient checkpointing: recompute ResNet block intermediates
        # during backward instead of storing them (~50GB → ~0.9GB).
        self._vae.enable_gradient_checkpointing()

        # Slicing: decode images one at a time internally,
        # preventing batch-size multiplication of peak memory.
        self._vae.enable_slicing()

    def forward(
        self,
        pred_latents: torch.Tensor,
        target_latents: torch.Tensor,
        noise_pred: torch.Tensor = None,
        noise: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        """
        Compute pixel-space decoded loss.

        Args:
            pred_latents: Predicted clean latents from UNet [B, 4, H/8, W/8].
            target_latents: Ground truth clean latents [B, 4, H/8, W/8].
            noise_pred: Unused (API compatibility).
            noise: Unused (API compatibility).

        Returns:
            Dict with 'loss' (total), 'loss_mae', 'loss_grad'.
        """
        if self._vae is None:
            raise RuntimeError(
                "VAE not set on PixelDecodedLoss. "
                "Call set_vae(vae, scale_factor) before training."
            )

        # Decode predictions to pixel space — NO torch.no_grad()!
        # Gradients must flow: loss → VAE decode → model_output → ControlNet.
        # enable_gradient_checkpointing() handles memory by recomputing
        # intermediates during backward.
        # enable_slicing() processes B=1 internally.
        pred_decoded = self._vae.decode(
            pred_latents / self._vae_scale_factor
        ).sample

        # Decode targets — no gradients needed (ground truth).
        # Saves ~12.5GB per image by not retaining decode graph.
        with torch.no_grad():
            target_decoded = self._vae.decode(
                target_latents / self._vae_scale_factor
            ).sample

        # Clamp to normalized range [-1, 1]
        pred_decoded = pred_decoded.clamp(-1.0, 1.0)
        target_decoded = target_decoded.clamp(-1.0, 1.0)

        # MAE loss in pixel space
        loss_mae = F.l1_loss(pred_decoded, target_decoded)

        # Sobel gradient loss in pixel space
        loss_grad = gradient_loss(pred_decoded, target_decoded)

        total_loss = self.mae_weight * loss_mae + self.grad_weight * loss_grad

        return {
            "loss": total_loss,
            "loss_mae": loss_mae.item(),
            "loss_grad": loss_grad.item(),
        }
