"""
Report generation utilities for DSM2DTM evaluation.

Generates summary text files, JSON results, and worst-case analysis.
"""

import os
import json
import shutil
from typing import Dict, List

import numpy as np


def save_results_summary(
    results_list: List[Dict],
    output_dir: str,
    config_info: Dict = None,
    unit: str = "ft",
    logger=None,
):
    """
    Save overall evaluation results as text summary and JSON.

    Args:
        results_list: List of per-image result dicts.
        output_dir: Output directory.
        config_info: Optional dict with experiment config for logging.
        unit: Unit label.
        logger: Optional logger.
    """
    if not results_list:
        if logger:
            logger.warning("No valid results to summarize.")
        return

    avg_mae = np.mean([r["mae"] for r in results_list])
    avg_mse = np.mean([r["mse"] for r in results_list])
    avg_rmse = np.mean([r["rmse"] for r in results_list])
    std_rmse = np.std([r["rmse"] for r in results_list])

    if logger:
        logger.info(f"\n=== Overall Evaluation Results ===")
        logger.info(f"Processed: {len(results_list)} images")
        logger.info(f"Average MAE: {avg_mae:.4f} {unit}")
        logger.info(f"Average MSE: {avg_mse:.4f} {unit}²")
        logger.info(f"Average RMSE: {avg_rmse:.4f} {unit} (±{std_rmse:.4f})")

    # Text summary
    summary_path = os.path.join(output_dir, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== Overall Evaluation Results ===\n\n")
        if config_info:
            for k, v in config_info.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
        f.write(f"Processed: {len(results_list)} images\n\n")
        f.write(f"Average MAE: {avg_mae:.6f} {unit}\n")
        f.write(f"Average MSE: {avg_mse:.6f} {unit}²\n")
        f.write(f"Average RMSE: {avg_rmse:.6f} {unit} (±{std_rmse:.6f})\n\n")
        f.write("=== Individual Results ===\n\n")
        for r in sorted(results_list, key=lambda x: x["rmse"]):
            f.write(f"{r['name']}: MAE={r['mae']:.4f}{unit}, MSE={r['mse']:.4f}{unit}², RMSE={r['rmse']:.4f}{unit}\n")

    # JSON results
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json_results = {
            "summary": {
                "processed_images": len(results_list),
                "avg_mae": avg_mae,
                "avg_mse": avg_mse,
                "avg_rmse": avg_rmse,
                "std_rmse": std_rmse,
            },
            "individual_results": [
                {k: v for k, v in r.items() if k != "output_dir"} for r in results_list
            ],
        }
        json.dump(json_results, f, indent=2)

    if logger:
        logger.info(f"Results saved to {summary_path} and {json_path}")


def save_worst_cases(
    results_list: List[Dict],
    output_dir: str,
    n_worst: int = 10,
    unit: str = "ft",
    logger=None,
):
    """
    Copy the N worst-performing test cases to a separate folder.

    Args:
        results_list: List of per-image result dicts (must have 'output_dir' key).
        output_dir: Base output directory.
        n_worst: Number of worst cases to save.
        unit: Unit label.
        logger: Optional logger.
    """
    worst_dir = os.path.join(output_dir, "worst_cases")
    os.makedirs(worst_dir, exist_ok=True)

    sorted_results = sorted(results_list, key=lambda x: x["rmse"], reverse=True)
    worst_cases = sorted_results[:n_worst]

    if logger:
        logger.info(f"\n=== Top {n_worst} Worst Cases (Highest RMSE) ===")

    for i, case in enumerate(worst_cases):
        src_dir = case.get("output_dir")
        if src_dir and os.path.exists(src_dir):
            dst_dir = os.path.join(worst_dir, f"rank_{i + 1}_{case['name']}")
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)

        if logger:
            logger.info(
                f"  Rank {i + 1}: {case['name']} - "
                f"RMSE: {case['rmse']:.4f}{unit}, MAE: {case['mae']:.4f}{unit}"
            )

    # Save summary
    summary_path = os.path.join(worst_dir, "worst_cases_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== Worst Cases Summary ===\n\n")
        for i, case in enumerate(worst_cases):
            f.write(f"Rank {i + 1}: {case['name']}\n")
            f.write(f"  RMSE: {case['rmse']:.4f}{unit}\n")
            f.write(f"  MAE: {case['mae']:.4f}{unit}\n")
            f.write(f"  MSE: {case['mse']:.4f}{unit}²\n\n")

    if logger:
        logger.info(f"Worst cases saved to {worst_dir}")
