"""
CPU-only verification for PixelDecodedLoss.

Tests the loss module without GPU or real model weights by using a
lightweight mock VAE that mimics the AutoencoderKL decode interface.

Verifies:
  1. Registry registration ("pixel_decoded" is available)
  2. set_vae() integration (accepts VAE + scale factor)
  3. Forward pass produces correct output dict keys
  4. Gradient flow (loss.backward() produces grads on input)
  5. Output values are finite and positive
  6. Existing losses (pixel, noise) still work unchanged

Usage:
    python tools/verify_pixel_decoded_loss.py
"""

import sys
import os
from dataclasses import dataclass

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Mock VAE that mimics diffusers AutoencoderKL.decode() interface
# ---------------------------------------------------------------------------
class MockVAEOutput:
    """Mimics diffusers DecoderOutput with a .sample attribute."""

    def __init__(self, sample: torch.Tensor):
        self.sample = sample


class MockVAEDecoder(nn.Module):
    """
    Lightweight mock VAE decoder for CPU testing.

    Upsamples [B, 4, H, W] latents to [B, 1, H*8, W*8] pixel space
    using a single transposed convolution (differentiable).
    """

    def __init__(self):
        super().__init__()
        # Simple learnable upsample: 4 channels -> 1 channel, 8x upscale
        self.upsample = nn.ConvTranspose2d(4, 1, kernel_size=8, stride=8, bias=False)

    def decode(self, latents: torch.Tensor) -> MockVAEOutput:
        pixel = self.upsample(latents)
        return MockVAEOutput(sample=pixel)

    def enable_gradient_checkpointing(self):
        """No-op for mock — real VAE uses this for memory optimization."""
        pass

    def enable_slicing(self):
        """No-op for mock — real VAE uses this for batch-wise decoding."""
        pass


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------
def test_registry():
    """Test that pixel_decoded is registered and can be instantiated."""
    from src.losses import get_loss, list_losses

    available = list_losses()
    assert "pixel_decoded" in available, (
        f"'pixel_decoded' not in registry. Available: {available}"
    )

    @dataclass
    class MockLossConfig:
        type: str = "pixel_decoded"
        mae_weight: float = 1.0
        grad_weight: float = 0.5

    loss_fn = get_loss(MockLossConfig())
    assert loss_fn is not None
    assert hasattr(loss_fn, "set_vae"), "PixelDecodedLoss missing set_vae method"
    print("  [PASS] Registry: 'pixel_decoded' registered and instantiable")


def test_set_vae():
    """Test that set_vae accepts mock VAE without errors."""
    from src.losses.pixel_decoded_loss import PixelDecodedLoss

    loss_fn = PixelDecodedLoss(mae_weight=1.0, grad_weight=0.5)
    mock_vae = MockVAEDecoder()
    loss_fn.set_vae(mock_vae, vae_scale_factor=0.18215)
    assert loss_fn._vae is not None
    assert loss_fn._vae_scale_factor == 0.18215
    print("  [PASS] set_vae: VAE reference stored correctly")


def test_forward_pass():
    """Test forward pass produces correct output format."""
    from src.losses.pixel_decoded_loss import PixelDecodedLoss

    loss_fn = PixelDecodedLoss(mae_weight=1.0, grad_weight=0.5)
    mock_vae = MockVAEDecoder()
    loss_fn.set_vae(mock_vae, vae_scale_factor=0.18215)

    # Small tensors: B=2, C=4, H=8, W=8 latents -> decoded to B=2, 1, 64, 64
    pred = torch.randn(2, 4, 8, 8, requires_grad=True)
    target = torch.randn(2, 4, 8, 8)

    result = loss_fn(pred_latents=pred, target_latents=target)

    # Check output keys
    assert "loss" in result, "Missing 'loss' key in output"
    assert "loss_mae" in result, "Missing 'loss_mae' key in output"
    assert "loss_grad" in result, "Missing 'loss_grad' key in output"

    # Check values are finite
    assert torch.isfinite(result["loss"]), f"Loss is not finite: {result['loss']}"
    assert result["loss"].item() > 0, f"Loss should be positive: {result['loss'].item()}"
    assert isinstance(result["loss_mae"], float), "loss_mae should be a Python float"
    assert isinstance(result["loss_grad"], float), "loss_grad should be a Python float"

    print(f"  [PASS] Forward pass: loss={result['loss'].item():.6f}, "
          f"mae={result['loss_mae']:.6f}, grad={result['loss_grad']:.6f}")


def test_gradient_flow():
    """Test that gradients flow back through the VAE decode to the input."""
    from src.losses.pixel_decoded_loss import PixelDecodedLoss

    loss_fn = PixelDecodedLoss(mae_weight=1.0, grad_weight=0.5)
    mock_vae = MockVAEDecoder()
    loss_fn.set_vae(mock_vae, vae_scale_factor=0.18215)

    pred = torch.randn(2, 4, 8, 8, requires_grad=True)
    target = torch.randn(2, 4, 8, 8)

    result = loss_fn(pred_latents=pred, target_latents=target)
    result["loss"].backward()

    assert pred.grad is not None, "No gradients on pred_latents — gradient flow broken!"
    assert pred.grad.abs().sum() > 0, "Gradients are all zeros"
    print(f"  [PASS] Gradient flow: grad norm = {pred.grad.norm().item():.6f}")


def test_no_vae_raises():
    """Test that forward without set_vae raises RuntimeError."""
    from src.losses.pixel_decoded_loss import PixelDecodedLoss

    loss_fn = PixelDecodedLoss()
    pred = torch.randn(1, 4, 8, 8)
    target = torch.randn(1, 4, 8, 8)

    try:
        loss_fn(pred_latents=pred, target_latents=target)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "VAE not set" in str(e)
        print("  [PASS] Error handling: RuntimeError raised when VAE not set")


def test_existing_losses_unchanged():
    """Verify existing pixel and noise losses still work identically."""
    from src.losses import get_loss

    @dataclass
    class PixelConfig:
        type: str = "pixel"
        mae_weight: float = 1.0
        grad_weight: float = 0.5

    @dataclass
    class NoiseConfig:
        type: str = "noise"
        mae_weight: float = 1.0
        grad_weight: float = 0.5

    # Test PixelLoss
    pixel_loss = get_loss(PixelConfig())
    pred = torch.randn(2, 4, 8, 8)
    target = torch.randn(2, 4, 8, 8)
    result = pixel_loss(pred_latents=pred, target_latents=target)
    assert "loss" in result and torch.isfinite(result["loss"])
    assert not hasattr(pixel_loss, "set_vae"), "PixelLoss should NOT have set_vae"

    # Test NoiseLoss
    noise_loss = get_loss(NoiseConfig())
    noise_pred = torch.randn(2, 4, 8, 8)
    noise = torch.randn(2, 4, 8, 8)
    result = noise_loss(noise_pred=noise_pred, noise=noise)
    assert "loss" in result and torch.isfinite(result["loss"])

    print("  [PASS] Backward compat: PixelLoss and NoiseLoss work unchanged")


def test_trainer_set_vae_integration():
    """Verify the hasattr pattern in trainer works for all loss types."""
    from src.losses import get_loss
    from src.losses.pixel_decoded_loss import PixelDecodedLoss
    from src.losses.pixel_loss import PixelLoss
    from src.losses.noise_loss import NoiseLoss

    mock_vae = MockVAEDecoder()

    @dataclass
    class PDConfig:
        type: str = "pixel_decoded"
        mae_weight: float = 1.0
        grad_weight: float = 0.5

    @dataclass
    class PConfig:
        type: str = "pixel"
        mae_weight: float = 1.0
        grad_weight: float = 0.5

    @dataclass
    class NConfig:
        type: str = "noise"
        mae_weight: float = 1.0
        grad_weight: float = 0.5

    # Simulate what trainer.setup() does
    for config_cls, expect_set_vae in [(PDConfig, True), (PConfig, False), (NConfig, False)]:
        loss_fn = get_loss(config_cls())
        if hasattr(loss_fn, "set_vae"):
            loss_fn.set_vae(mock_vae, 0.18215)
            assert expect_set_vae, f"{config_cls.type} should not have set_vae"
        else:
            assert not expect_set_vae, f"{config_cls.type} should have set_vae"

    print("  [PASS] Trainer integration: hasattr(set_vae) pattern works for all loss types")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  PixelDecodedLoss — CPU Verification")
    print("=" * 60)
    print()

    tests = [
        ("Registry registration", test_registry),
        ("set_vae integration", test_set_vae),
        ("Forward pass output", test_forward_pass),
        ("Gradient flow through VAE decode", test_gradient_flow),
        ("Error when VAE not set", test_no_vae_raises),
        ("Existing losses unchanged", test_existing_losses_unchanged),
        ("Trainer set_vae pattern", test_trainer_set_vae_integration),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"Test: {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n  All verifications passed! Safe to run on GPU.")
        sys.exit(0)


if __name__ == "__main__":
    main()
