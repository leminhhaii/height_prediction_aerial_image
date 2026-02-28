"""
Noise prediction MSE loss for DSM2DTM.

Standard diffusion training objective: MSE between predicted noise and actual noise.
Used when prediction_type="epsilon" (model predicts noise).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_loss


@register_loss("noise")
class NoiseLoss(nn.Module):
    """
    Standard noise prediction MSE loss.

    Loss = MSE(noise_pred, noise)

    This is the standard diffusion training objective where the model
    predicts the noise added to the latents.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self,
        pred_latents: torch.Tensor = None,
        target_latents: torch.Tensor = None,
        noise_pred: torch.Tensor = None,
        noise: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        """
        Compute noise MSE loss.

        Args:
            pred_latents: Unused (for API compatibility with PixelLoss).
            target_latents: Unused.
            noise_pred: Predicted noise from the model.
            noise: Actual noise that was added.

        Returns:
            Dict with 'loss' (total), 'loss_mse'.
        """
        if noise_pred is None or noise is None:
            raise ValueError("NoiseLoss requires 'noise_pred' and 'noise' arguments")

        loss = F.mse_loss(noise_pred, noise)

        return {
            "loss": loss,
            "loss_mse": loss.item(),
        }
