"""
Visualization utilities for DSM2DTM evaluation.

Generates error maps, analysis plots, and elevation-based visualizations.
"""

import os
import math
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..utils.geo import save_tif_with_metadata, save_png_uint16


def save_error_visualization(
    pred_arr: np.ndarray,
    gt_arr: np.ndarray,
    out_dir: str,
    metrics: Dict[str, float],
    unit: str = "ft",
    ref_image_path: str = None,
):
    """
    Save error visualization images:
    1. Absolute Error Heatmap
    2. Prediction vs Ground Truth Scatter
    3. Error Histogram
    4. Signed Error Map
    Also saves error.tif with geospatial metadata.
    """
    if pred_arr.shape != gt_arr.shape:
        h = min(pred_arr.shape[0], gt_arr.shape[0])
        w = min(pred_arr.shape[1], gt_arr.shape[1])
        pred_arr = pred_arr[:h, :w]
        gt_arr = gt_arr[:h, :w]

    valid_mask = (gt_arr > -10000) & (gt_arr < 10000) & (pred_arr > -10000) & (pred_arr < 10000)
    error = np.abs(pred_arr - gt_arr)

    # Save error TIF
    nodata_val = -10000.0
    error_tif_data = error.copy()
    error_tif_data[~valid_mask] = nodata_val

    error_tif = os.path.join(out_dir, "error.tif")
    save_tif_with_metadata(error_tif_data, error_tif, ref_image_path=ref_image_path, nodata=nodata_val)

    # Create 4-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Error heatmap
    ax1 = axes[0, 0]
    error_valid = error.copy()
    error_valid[~valid_mask] = 0
    vmax_val = max(20.0, np.percentile(error[valid_mask], 95)) if valid_mask.any() else 20.0
    im1 = ax1.imshow(error_valid, cmap="jet", vmin=0, vmax=vmax_val)
    ax1.set_title(f'Absolute Error |pred - gt| (MAE: {metrics["mae"]:.2f}{unit}, RMSE: {metrics["rmse"]:.2f}{unit})')
    plt.colorbar(im1, ax=ax1, label=f"Error ({unit})")

    # 2. Scatter plot
    ax2 = axes[0, 1]
    if valid_mask.any():
        gt_valid = gt_arr[valid_mask]
        pred_valid = pred_arr[valid_mask]

        n_points = len(gt_valid)
        if n_points > 10000:
            step = n_points // 10000
            gt_sample = gt_valid[::step]
            pred_sample = pred_valid[::step]
        else:
            gt_sample = gt_valid
            pred_sample = pred_valid

        ax2.scatter(gt_sample, pred_sample, alpha=0.5, s=3, c="blue", edgecolors="none")

        min_val = min(gt_valid.min(), pred_valid.min())
        max_val = max(gt_valid.max(), pred_valid.max())
        ax2.plot([min_val, max_val], [min_val, max_val], "r-", linewidth=2, label="Perfect prediction")
        ax2.set_xlim(min_val - 5, max_val + 5)
        ax2.set_ylim(min_val - 5, max_val + 5)

    ax2.set_xlabel(f"Ground Truth ({unit})")
    ax2.set_ylabel(f"Prediction ({unit})")
    ax2.set_title("Prediction vs Ground Truth")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal", adjustable="box")

    # 3. Error histogram
    ax3 = axes[1, 0]
    if valid_mask.any():
        error_flat = error[valid_mask]
        ax3.hist(error_flat, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax3.axvline(metrics["mae"], color="r", linestyle="--", linewidth=2, label=f'MAE: {metrics["mae"]:.2f}{unit}')
    ax3.axvline(metrics["rmse"], color="g", linestyle="--", linewidth=2, label=f'RMSE: {metrics["rmse"]:.2f}{unit}')
    ax3.set_xlabel(f"Absolute Error ({unit})")
    ax3.set_ylabel("Frequency (pixels)")
    ax3.set_title("Error Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Signed error
    ax4 = axes[1, 1]
    signed_error = pred_arr - gt_arr
    signed_error[~valid_mask] = 0
    if valid_mask.any():
        vmax_signed = max(
            abs(np.percentile(signed_error[valid_mask], 5)),
            abs(np.percentile(signed_error[valid_mask], 95)),
        )
    else:
        vmax_signed = 20.0
    im4 = ax4.imshow(signed_error, cmap="RdBu_r", vmin=-vmax_signed, vmax=vmax_signed)
    ax4.set_title("Signed Error (pred - gt)\n(Red=over-predict, Blue=under-predict)")
    plt.colorbar(im4, ax=ax4, label=f"Error ({unit})")

    plt.tight_layout()
    error_plot_path = os.path.join(out_dir, "error_analysis.png")
    plt.savefig(error_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Save simple error heatmap separately
    plt.figure(figsize=(10, 8))
    plt.imshow(error_valid, cmap="jet", vmin=0, vmax=vmax_val)
    plt.colorbar(label=f"Absolute Error ({unit})")
    plt.title(f'Error Map (RMSE: {metrics["rmse"]:.2f}{unit})')
    plt.savefig(os.path.join(out_dir, "error_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_elevation_analysis(
    all_elevation_metrics: List[Dict[str, Dict[str, float]]],
    output_dir: str,
    unit: str = "ft",
    logger=None,
):
    """
    Create plots and tables for elevation-based error analysis.

    Returns:
        Pandas DataFrame with aggregated elevation analysis.
    """
    import pandas as pd

    # Aggregate metrics across all images
    aggregated = {}
    for img_metrics in all_elevation_metrics:
        for bin_name, metrics in img_metrics.items():
            if bin_name not in aggregated:
                aggregated[bin_name] = {"rmse_list": [], "mae_list": [], "count": 0}
            if not math.isnan(metrics["rmse"]):
                aggregated[bin_name]["rmse_list"].append(metrics["rmse"])
                aggregated[bin_name]["mae_list"].append(metrics["mae"])
                aggregated[bin_name]["count"] += metrics["count"]

    elevation_data = []
    for bin_name in sorted(aggregated.keys(), key=lambda x: int(x.split("-")[0])):
        data = aggregated[bin_name]
        if len(data["rmse_list"]) > 0:
            avg_rmse = np.mean(data["rmse_list"])
            std_rmse = np.std(data["rmse_list"])
            avg_mae = np.mean(data["mae_list"])
            std_mae = np.std(data["mae_list"])
        else:
            avg_rmse = std_rmse = avg_mae = std_mae = float("nan")

        elevation_data.append({
            "Elevation Range": bin_name,
            f"Avg RMSE ({unit})": avg_rmse,
            f"Std RMSE ({unit})": std_rmse,
            f"Avg MAE ({unit})": avg_mae,
            f"Std MAE ({unit})": std_mae,
            "Total Pixels": data["count"],
        })

    df = pd.DataFrame(elevation_data)
    csv_path = os.path.join(output_dir, "elevation_analysis.csv")
    df.to_csv(csv_path, index=False)

    if logger:
        logger.info(f"Elevation analysis saved to {csv_path}")
        logger.info("\n=== Elevation-based Error Analysis ===")
        logger.info(df.to_string(index=False))

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    valid_data = df[~df[f"Avg RMSE ({unit})"].isna()]
    x_labels = valid_data["Elevation Range"].tolist()
    x_pos = np.arange(len(x_labels))

    ax1 = axes[0]
    ax1.bar(x_pos, valid_data[f"Avg RMSE ({unit})"], yerr=valid_data[f"Std RMSE ({unit})"],
            capsize=5, alpha=0.7, color="steelblue")
    ax1.set_xlabel("Elevation Range")
    ax1.set_ylabel(f"RMSE ({unit})")
    ax1.set_title("Average RMSE by Elevation Range")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = axes[1]
    ax2.bar(x_pos, valid_data[f"Avg MAE ({unit})"], yerr=valid_data[f"Std MAE ({unit})"],
            capsize=5, alpha=0.7, color="coral")
    ax2.set_xlabel("Elevation Range")
    ax2.set_ylabel(f"MAE ({unit})")
    ax2.set_title("Average MAE by Elevation Range")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "elevation_analysis_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    if logger:
        logger.info(f"Elevation analysis plot saved to {plot_path}")

    return df


def create_overall_visualization(
    results_list: List[Dict],
    output_dir: str,
    unit: str = "ft",
    logger=None,
):
    """Create overall visualization of evaluation results with 4 panels."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    names = [r["name"] for r in results_list]
    maes = [r["mae"] for r in results_list]
    rmses = [r["rmse"] for r in results_list]

    # 1. RMSE distribution
    ax1 = axes[0, 0]
    ax1.hist(rmses, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.axvline(np.mean(rmses), color="r", linestyle="--", label=f"Mean: {np.mean(rmses):.2f}{unit}")
    ax1.axvline(np.median(rmses), color="g", linestyle="--", label=f"Median: {np.median(rmses):.2f}{unit}")
    ax1.set_xlabel(f"RMSE ({unit})")
    ax1.set_ylabel("Frequency")
    ax1.set_title("RMSE Distribution Across Test Images")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. MAE vs RMSE scatter
    ax2 = axes[0, 1]
    ax2.scatter(maes, rmses, alpha=0.6, edgecolors="black", linewidth=0.5)
    ax2.set_xlabel(f"MAE ({unit})")
    ax2.set_ylabel(f"RMSE ({unit})")
    ax2.set_title("MAE vs RMSE per Image")
    ax2.grid(True, alpha=0.3)

    # 3. Top 20 worst cases
    ax3 = axes[1, 0]
    sorted_results = sorted(results_list, key=lambda x: x["rmse"], reverse=True)[:20]
    sorted_names = [r["name"] for r in sorted_results]
    sorted_rmses = [r["rmse"] for r in sorted_results]
    y_pos = np.arange(len(sorted_names))
    ax3.barh(y_pos, sorted_rmses, alpha=0.7, color="coral")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(sorted_names, fontsize=8)
    ax3.set_xlabel(f"RMSE ({unit})")
    ax3.set_title("Top 20 Images with Highest RMSE")
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis="x")

    # 4. Box plot
    ax4 = axes[1, 1]
    bp = ax4.boxplot([maes, rmses], labels=["MAE", "RMSE"], patch_artist=True)
    colors = ["lightblue", "lightcoral"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax4.set_ylabel(f"Error ({unit})")
    ax4.set_title("Distribution of MAE and RMSE")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "overall_evaluation.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    if logger:
        logger.info(f"Overall visualization saved to {plot_path}")
