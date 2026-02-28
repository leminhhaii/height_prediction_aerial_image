"""
Loss function registry for DSM2DTM.

Supports registering custom loss functions via decorator pattern.
"""

from typing import Dict, Type, Callable

import torch.nn as nn

# Global registry
_LOSS_REGISTRY: Dict[str, Type] = {}


def register_loss(name: str):
    """
    Decorator to register a loss function class.

    Usage:
        @register_loss("my_custom_loss")
        class MyCustomLoss(nn.Module):
            ...
    """
    def decorator(cls):
        _LOSS_REGISTRY[name] = cls
        return cls
    return decorator


def get_loss(loss_config) -> nn.Module:
    """
    Create a loss function from config.

    Args:
        loss_config: LossConfig dataclass or dict with 'type' key.

    Returns:
        Loss function module.
    """
    if hasattr(loss_config, "type"):
        loss_type = loss_config.type
    elif isinstance(loss_config, dict):
        loss_type = loss_config.get("type", "pixel")
    else:
        loss_type = str(loss_config)

    if loss_type not in _LOSS_REGISTRY:
        available = ", ".join(_LOSS_REGISTRY.keys())
        raise ValueError(
            f"Unknown loss type: '{loss_type}'. Available: {available}"
        )

    loss_cls = _LOSS_REGISTRY[loss_type]

    # Pass config kwargs to the loss class constructor
    if hasattr(loss_config, "__dataclass_fields__"):
        kwargs = {k: v for k, v in loss_config.__dict__.items() if k != "type"}
    elif isinstance(loss_config, dict):
        kwargs = {k: v for k, v in loss_config.items() if k != "type"}
    else:
        kwargs = {}

    return loss_cls(**kwargs)


def list_losses() -> list:
    """Return list of registered loss names."""
    return list(_LOSS_REGISTRY.keys())
