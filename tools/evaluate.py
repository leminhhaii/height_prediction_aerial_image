#!/usr/bin/env python
"""
DSM2DTM – Evaluation entry point.

Evaluates predictions against ground truth DTM tiles.

Usage:
    # Evaluate using split (auto-discovers predictions)
    python tools/evaluate.py --config configs/eval_default.yaml \
        --predictions_dir experiments/pixel_loss/inference

    # Evaluate with explicit GT directory
    python tools/evaluate.py --config configs/eval_default.yaml \
        --predictions_dir experiments/pixel_loss/inference \
        --gt_dir datasets/new_data/ndsm
"""

import sys
import os
import glob
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import rasterio

from src.utils.config import add_config_args, parse_config_args
from src.utils.logging_utils import setup_logger
from src.data.split import load_split
from src.data.preprocessing import extract_index
from src.evaluation.metrics import calculate_metrics, calculate_metrics_by_elevation
from src.evaluation.visualization import (
    save_error_visualization,
    plot_elevation_analysis,
    create_overall_visualization,
)
from src.evaluation.report import save_results_summary, save_worst_cases


def load_tif_as_array(path: str) -> np.ndarray:
    """Load a .tif file and return as float64 numpy array."""
    with rasterio.open(path) as ds:
        return ds.read(1).astype(np.float64)


def discover_pairs(predictions_dir: str, gt_dir: str, test_indices=None):
    """
    Discover prediction/GT pairs.

    Searches for pred_dtm.tif or *.tif inside per-image subdirs.
    """
    pairs = []

    if test_indices is not None:
        # test_indices entries are [dsm_name, dtm_name] pairs or plain indices
        for entry in test_indices:
            if isinstance(entry, (list, tuple)):
                dsm_name, dtm_name = entry[0], entry[1]
                name = os.path.splitext(dsm_name)[0]  # e.g. "dsm_174"
            else:
                dsm_name = f"dsm_{entry}.tif"
                dtm_name = dsm_name
                name = f"dsm_{entry}"

            pred_dir = os.path.join(predictions_dir, name)
            gt_path = os.path.join(gt_dir, dtm_name)

            if not os.path.exists(gt_path):
                continue

            # Look for prediction file
            pred_path = None
            for candidate in ["pred_dtm.tif", "pred_denorm.tif", "prediction.tif"]:
                p = os.path.join(pred_dir, candidate)
                if os.path.exists(p):
                    pred_path = p
                    break
            if pred_path is None:
                # Try any TIF in the subdir
                tifs = glob.glob(os.path.join(pred_dir, "*.tif"))
                if tifs:
                    pred_path = tifs[0]

            if pred_path:
                pairs.append({"name": name, "pred": pred_path, "gt": gt_path, "dir": pred_dir})
    else:
        # Auto-discover from prediction subdirs
        for subdir in sorted(os.listdir(predictions_dir)):
            pred_dir = os.path.join(predictions_dir, subdir)
            if not os.path.isdir(pred_dir):
                continue

            gt_path = os.path.join(gt_dir, f"{subdir}.tif")
            if not os.path.exists(gt_path):
                continue

            pred_path = None
            for candidate in ["pred_dtm.tif", "pred_denorm.tif", "prediction.tif"]:
                p = os.path.join(pred_dir, candidate)
                if os.path.exists(p):
                    pred_path = p
                    break
            if pred_path is None:
                tifs = glob.glob(os.path.join(pred_dir, "*.tif"))
                if tifs:
                    pred_path = tifs[0]

            if pred_path:
                pairs.append({"name": subdir, "pred": pred_path, "gt": gt_path, "dir": pred_dir})

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Evaluate DSM→DTM predictions against ground truth")
    add_config_args(parser)
    parser.add_argument("--predictions_dir", type=str, required=True,
                        help="Directory containing prediction subdirs")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="Ground truth DTM directory (overrides data.data_root/dtm_dir)")
    args = parser.parse_args()

    config = parse_config_args(args)

    os.makedirs(config.output.dir, exist_ok=True)
    logger = setup_logger(config.output.dir, name="evaluate")

    # Determine GT dir
    gt_dir = args.gt_dir
    if gt_dir is None:
        gt_dir = os.path.join(config.data.data_root, config.data.dtm_dir)
    logger.info(f"Predictions: {args.predictions_dir}")
    logger.info(f"Ground truth: {gt_dir}")

    # Get test indices from split
    test_indices = None
    if os.path.exists(config.data.split_json):
        split_data = load_split(config.data.split_json)
        test_indices = split_data.get("test", None)
        if test_indices:
            logger.info(f"Test set: {len(test_indices)} images from split")

    # Discover pairs
    pairs = discover_pairs(args.predictions_dir, gt_dir, test_indices)
    logger.info(f"Found {len(pairs)} prediction/GT pairs")

    if not pairs:
        logger.error("No valid pairs found. Check paths.")
        return

    # Evaluate each pair
    results_list = []
    all_elevation_metrics = []
    unit = config.output.unit
    elevation_bins = config.evaluation.elevation_bins

    test_results_dir = os.path.join(config.output.dir, "test_results")
    os.makedirs(test_results_dir, exist_ok=True)

    for pair in pairs:
        name = pair["name"]
        logger.info(f"Evaluating {name}...")

        pred_arr = load_tif_as_array(pair["pred"])
        gt_arr = load_tif_as_array(pair["gt"])

        # Overall metrics
        metrics = calculate_metrics(pred_arr, gt_arr)

        # Per-elevation metrics
        elev_metrics = calculate_metrics_by_elevation(pred_arr, gt_arr, elevation_bins, unit)

        # Save per-image results
        img_out_dir = os.path.join(test_results_dir, name)
        os.makedirs(img_out_dir, exist_ok=True)

        save_error_visualization(
            pred_arr, gt_arr, img_out_dir, metrics, unit,
            ref_image_path=pair["gt"],
        )

        result = {
            "name": name,
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "output_dir": img_out_dir,
        }
        results_list.append(result)
        all_elevation_metrics.append(elev_metrics)

        logger.info(f"  MAE={metrics['mae']:.4f}{unit}, RMSE={metrics['rmse']:.4f}{unit}")

    # Summary results
    save_results_summary(results_list, config.output.dir, unit=unit, logger=logger)
    save_worst_cases(results_list, config.output.dir, n_worst=config.evaluation.n_worst, unit=unit, logger=logger)

    # Elevation analysis
    if all_elevation_metrics:
        plot_elevation_analysis(all_elevation_metrics, config.output.dir, unit=unit, logger=logger)

    # Overall visualization
    create_overall_visualization(results_list, config.output.dir, unit=unit, logger=logger)

    logger.info(f"\nAll results saved to {config.output.dir}")


if __name__ == "__main__":
    main()
