"""
Normalization strategies for DSM2DTM.

Two strategies are supported:
- LogGlobalNorm: Global log-normalization (used by pixel-loss approach)
- PercentileNorm: Per-image percentile normalization (used by noise-loss approach)

These are INCOMPATIBLE — a model trained with one normalization must use the same
normalization at inference time.
"""

import math
from abc import ABC, abstractmethod

import numpy as np


class NormalizationStrategy(ABC):
    """Base class for normalization strategies."""

    @abstractmethod
    def normalize(self, arr: np.ndarray, data_type: str = "dsm") -> np.ndarray:
        """Normalize an array to [-1, 1] range."""
        ...

    @abstractmethod
    def denormalize(self, arr_norm: np.ndarray, data_type: str = "dtm") -> np.ndarray:
        """Denormalize an array from [-1, 1] back to real values."""
        ...

    @abstractmethod
    def denormalize_tensor(self, tensor_norm, data_type: str = "dtm"):
        """Denormalize a PyTorch tensor from [-1, 1] back to real values."""
        ...


class LogGlobalNorm(NormalizationStrategy):
    """
    Global Log-Normalization.

    Applies: clip → ln(v+1) → scale to [-1, 1] using global min/max.
    Used by the pixel-loss training approach.
    """

    def __init__(
        self,
        dsm_global_min: float = 760.0,
        dsm_global_max: float = 1450.0,
        dtm_global_min: float = 0.0,
        dtm_global_max: float = 650.0,
    ):
        self.dsm_min = dsm_global_min
        self.dsm_max = dsm_global_max
        self.dtm_min = dtm_global_min
        self.dtm_max = dtm_global_max

        # Precompute log constants
        self.ln_dsm_min = math.log(self.dsm_min + 1)
        self.ln_dsm_max = math.log(self.dsm_max + 1)
        self.ln_dtm_min = math.log(self.dtm_min + 1)
        self.ln_dtm_max = math.log(self.dtm_max + 1)

    def _get_bounds(self, data_type: str):
        if data_type == "dsm":
            return self.dsm_min, self.dsm_max, self.ln_dsm_min, self.ln_dsm_max
        else:  # dtm / ndsm
            return self.dtm_min, self.dtm_max, self.ln_dtm_min, self.ln_dtm_max

    def normalize(self, arr: np.ndarray, data_type: str = "dsm") -> np.ndarray:
        g_min, g_max, ln_min, ln_max = self._get_bounds(data_type)

        arr = np.clip(arr, g_min, g_max)
        arr_log = np.log(arr + 1)

        denom = ln_max - ln_min
        if denom < 1e-6:
            return np.zeros_like(arr)

        arr_norm = 2.0 * (arr_log - ln_min) / denom - 1.0
        return arr_norm

    def denormalize(self, arr_norm: np.ndarray, data_type: str = "dtm") -> np.ndarray:
        _, _, ln_min, ln_max = self._get_bounds(data_type)
        range_ln = ln_max - ln_min

        v_01 = (arr_norm + 1.0) / 2.0
        v_log = v_01 * range_ln + ln_min
        v_real = np.exp(v_log) - 1.0
        return v_real

    def denormalize_tensor(self, tensor_norm, data_type: str = "dtm"):
        """Denormalize a PyTorch tensor."""
        import torch

        _, _, ln_min, ln_max = self._get_bounds(data_type)
        range_ln = ln_max - ln_min

        v_01 = (tensor_norm + 1.0) / 2.0
        v_log = v_01 * range_ln + ln_min
        v_real = torch.exp(v_log) - 1.0
        return v_real


class PercentileNorm(NormalizationStrategy):
    """
    Per-image Percentile Normalization.

    Clips to [P_low, P_high] percentile, then min-max normalizes to [-1, 1].
    Used by the noise-loss training approach.

    Note: This normalization is NOT invertible without knowing the original
    min/max values. For denormalization during inference, we store and return
    the per-image stats.
    """

    def __init__(self, p_low: float = 1.0, p_high: float = 99.0):
        self.p_low = p_low
        self.p_high = p_high
        # Store last normalization stats for denormalization
        self._last_min = None
        self._last_max = None

    def normalize(self, arr: np.ndarray, data_type: str = "dsm") -> np.ndarray:
        p_lo, p_hi = np.percentile(arr, (self.p_low, self.p_high))
        arr = np.clip(arr, p_lo, p_hi)

        mn, mx = arr.min(), arr.max()
        self._last_min = mn
        self._last_max = mx

        if mx - mn < 1e-6:
            return np.zeros_like(arr)

        a = (arr - mn) / (mx - mn)
        a = a * 2.0 - 1.0
        return a

    def denormalize(self, arr_norm: np.ndarray, data_type: str = "dtm") -> np.ndarray:
        """
        Approximate denormalization using stored stats.
        Only valid immediately after a corresponding normalize() call.
        """
        if self._last_min is None or self._last_max is None:
            # Fallback: just undo [-1,1] -> [0,1] scaling
            return (arr_norm + 1.0) / 2.0

        a_01 = (arr_norm + 1.0) / 2.0
        return a_01 * (self._last_max - self._last_min) + self._last_min

    def denormalize_tensor(self, tensor_norm, data_type: str = "dtm"):
        """Approximate denormalization for tensors."""
        if self._last_min is None or self._last_max is None:
            return (tensor_norm + 1.0) / 2.0

        a_01 = (tensor_norm + 1.0) / 2.0
        return a_01 * (self._last_max - self._last_min) + self._last_min


def get_normalizer(config) -> NormalizationStrategy:
    """
    Factory function to create a normalizer from config.

    Args:
        config: Can be a DataConfig, Config, or a dict with normalization params.

    Returns:
        NormalizationStrategy instance.
    """
    if hasattr(config, "data"):
        # Full Config object
        data_cfg = config.data
    elif hasattr(config, "normalization"):
        # DataConfig object
        data_cfg = config
    elif isinstance(config, dict):
        # Dict — wrap in a simple namespace
        class _NS:
            pass
        data_cfg = _NS()
        for k, v in config.items():
            setattr(data_cfg, k, v)
    else:
        raise ValueError(f"Unknown config type: {type(config)}")

    norm_type = getattr(data_cfg, "normalization", "log_global")

    if norm_type == "log_global":
        return LogGlobalNorm(
            dsm_global_min=getattr(data_cfg, "dsm_global_min", 760.0),
            dsm_global_max=getattr(data_cfg, "dsm_global_max", 1450.0),
            dtm_global_min=getattr(data_cfg, "dtm_global_min", 0.0),
            dtm_global_max=getattr(data_cfg, "dtm_global_max", 650.0),
        )
    elif norm_type == "percentile":
        return PercentileNorm(
            p_low=getattr(data_cfg, "percentile_low", 1.0),
            p_high=getattr(data_cfg, "percentile_high", 99.0),
        )
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}. Use 'log_global' or 'percentile'.")
