"""
Evaluation metrics for DSM2DTM.

Provides overall and per-elevation-bin metric computation.
"""

import math
from typing import Dict, List

import numpy as np


def calculate_rmse(pred_arr: np.ndarray, gt_arr: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error between prediction and ground truth.

    Handles shape mismatches and NoData filtering.
    """
    if pred_arr.shape != gt_arr.shape:
        h = min(pred_arr.shape[0], gt_arr.shape[0])
        w = min(pred_arr.shape[1], gt_arr.shape[1])
        pred_arr = pred_arr[:h, :w]
        gt_arr = gt_arr[:h, :w]

    valid_mask = (gt_arr > -10000) & (gt_arr < 10000)
    if not valid_mask.any():
        return float("nan")

    diff = pred_arr[valid_mask].astype(np.float64) - gt_arr[valid_mask].astype(np.float64)
    mse = np.mean(diff ** 2)
    return np.sqrt(mse)


def calculate_metrics(pred_arr: np.ndarray, gt_arr: np.ndarray) -> Dict[str, float]:
    """
    Calculate MAE, MSE, RMSE between prediction and ground truth.

    Returns:
        Dict with 'mae', 'mse', 'rmse' keys.
    """
    if pred_arr.shape != gt_arr.shape:
        h = min(pred_arr.shape[0], gt_arr.shape[0])
        w = min(pred_arr.shape[1], gt_arr.shape[1])
        pred_arr = pred_arr[:h, :w]
        gt_arr = gt_arr[:h, :w]

    valid_mask = (gt_arr > -10000) & (gt_arr < 10000)
    if not valid_mask.any():
        return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")}

    diff = pred_arr[valid_mask].astype(np.float64) - gt_arr[valid_mask].astype(np.float64)

    mae = np.mean(np.abs(diff))
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)

    return {"mae": mae, "mse": mse, "rmse": rmse}


def calculate_metrics_by_elevation(
    pred_arr: np.ndarray,
    gt_arr: np.ndarray,
    elevation_bins: List[float],
    unit: str = "ft",
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for different elevation ranges.

    Args:
        pred_arr: Predicted elevation array.
        gt_arr: Ground truth elevation array.
        elevation_bins: List of bin boundaries.
        unit: Unit label for bin names.

    Returns:
        Dict mapping bin names to metric dicts.
    """
    if pred_arr.shape != gt_arr.shape:
        h = min(pred_arr.shape[0], gt_arr.shape[0])
        w = min(pred_arr.shape[1], gt_arr.shape[1])
        pred_arr = pred_arr[:h, :w]
        gt_arr = gt_arr[:h, :w]

    results = {}

    for i in range(len(elevation_bins) - 1):
        low, high = elevation_bins[i], elevation_bins[i + 1]
        bin_name = f"{low}-{high}{unit}"

        mask = (gt_arr >= low) & (gt_arr < high) & (gt_arr > -10000) & (gt_arr < 10000)

        if not mask.any():
            results[bin_name] = {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan"), "count": 0}
            continue

        diff = pred_arr[mask].astype(np.float64) - gt_arr[mask].astype(np.float64)

        mae = np.mean(np.abs(diff))
        mse = np.mean(diff ** 2)
        rmse = np.sqrt(mse)
        count = int(np.sum(mask))

        results[bin_name] = {"mae": mae, "mse": mse, "rmse": rmse, "count": count}

    return results
