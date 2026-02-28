"""
Pixel-level composite loss for DSM2DTM.

Combines MAE (L1) loss with Sobel gradient loss on predicted clean latents.
Used when prediction_type="sample" (model predicts x0, not noise).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_loss


def sobel_gradients(x: torch.Tensor):
    """
    Compute Sobel gradients (horizontal and vertical) on a tensor.

    Args:
        x: Tensor of shape [B, C, H, W].

    Returns:
        Tuple of (grad_x, grad_y) tensors.
    """
    fx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 3)
    fy = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 3)

    if x.shape[1] > 1:
        fx = fx.repeat(x.shape[1], 1, 1, 1)
        fy = fy.repeat(x.shape[1], 1, 1, 1)

    grad_x = F.conv2d(x, fx, padding=1, groups=x.shape[1])
    grad_y = F.conv2d(x, fy, padding=1, groups=x.shape[1])
    return grad_x, grad_y


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Sobel gradient matching loss.

    Args:
        pred: Predicted tensor [B, C, H, W].
        target: Target tensor [B, C, H, W].

    Returns:
        Scalar loss value.
    """
    pred_gx, pred_gy = sobel_gradients(pred)
    target_gx, target_gy = sobel_gradients(target)
    return F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)


@register_loss("pixel")
class PixelLoss(nn.Module):
    """
    Pixel-level composite loss: MAE + Gradient.

    Loss = mae_weight * L1(pred, target) + grad_weight * GradientLoss(pred, target)

    This loss operates on predicted clean latents (x0) vs target latents.
    """

    def __init__(self, mae_weight: float = 1.0, grad_weight: float = 0.5, **kwargs):
        super().__init__()
        self.mae_weight = mae_weight
        self.grad_weight = grad_weight

    def forward(
        self,
        pred_latents: torch.Tensor,
        target_latents: torch.Tensor,
        noise_pred: torch.Tensor = None,
        noise: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        """
        Compute pixel-level loss.

        Args:
            pred_latents: Predicted clean latents (= noise_pred when prediction_type="sample").
            target_latents: Ground truth clean latents.
            noise_pred: Unused (for API compatibility with NoiseLoss).
            noise: Unused.

        Returns:
            Dict with 'loss' (total), 'loss_mae', 'loss_grad'.
        """
        loss_mae = F.l1_loss(pred_latents, target_latents)
        loss_grad = gradient_loss(pred_latents, target_latents)

        total_loss = self.mae_weight * loss_mae + self.grad_weight * loss_grad

        return {
            "loss": total_loss,
            "loss_mae": loss_mae.item(),
            "loss_grad": loss_grad.item(),
        }
