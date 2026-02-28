#!/usr/bin/env python
"""
DSM2DTM – Inference entry point.

Usage:
    # Single image
    python tools/infer.py --config configs/infer_default.yaml --input path/to/dsm.tif

    # Directory of images
    python tools/infer.py --config configs/infer_default.yaml --input_dir path/to/dsm_folder

    # Test set from split
    python tools/infer.py --config configs/infer_default.yaml --split test
"""

import sys
import os
import glob
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.utils.config import add_config_args, parse_config_args
from src.utils.logging_utils import setup_logger
from src.data.normalization import get_normalizer
from src.data.split import load_split
from src.data.preprocessing import extract_index
from src.models.controlnet_loader import load_all_models
from src.models.pipeline import DSM2DTMPipeline


def main():
    parser = argparse.ArgumentParser(description="Run inference: DSM→DTM conversion")
    add_config_args(parser)
    parser.add_argument("--input", type=str, default=None, help="Path to a single DSM .tif file")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory of DSM .tif files")
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"],
                        help="Run inference on a split from split_dataset.json")
    args = parser.parse_args()

    config = parse_config_args(args)

    os.makedirs(config.output.dir, exist_ok=True)
    logger = setup_logger(config.output.dir, name="infer")
    logger.info(f"Output: {config.output.dir}")

    # Build input file list
    dsm_files = []
    if args.input:
        dsm_files = [args.input]
    elif args.input_dir:
        dsm_files = sorted(glob.glob(os.path.join(args.input_dir, "*.tif")))
    elif args.split:
        split_data = load_split(config.data.split_json)
        split_entries = split_data.get(args.split, [])
        dsm_dir = os.path.join(config.data.data_root, config.data.dsm_dir)
        for entry in split_entries:
            # entry is [dsm_name, dtm_name] pair
            dsm_name = entry[0] if isinstance(entry, (list, tuple)) else entry
            path = os.path.join(dsm_dir, dsm_name)
            if os.path.exists(path):
                dsm_files.append(path)
    else:
        parser.error("Provide one of --input, --input_dir, or --split")

    logger.info(f"Found {len(dsm_files)} DSM file(s)")

    # Load models
    normalizer = get_normalizer(config)
    models = load_all_models(
        base_model_id=config.model.base_model,
        controlnet_path=config.model.controlnet_path,
        vae_lora_path=config.model.vae_path,
        prediction_type=config.model.prediction_type,
        device=config.device,
    )

    pipeline = DSM2DTMPipeline.from_models_tuple(
        models=models,
        normalizer=normalizer,
        device=config.device,
        vae_scale_factor=config.vae_scale_factor,
        fp16=config.fp16,
    )

    # Run inference
    for dsm_path in dsm_files:
        name = os.path.splitext(os.path.basename(dsm_path))[0]
        out_dir = os.path.join(config.output.dir, name)
        os.makedirs(out_dir, exist_ok=True)

        logger.info(f"Processing {name}...")

        try:
            pipeline.predict_and_save(
                image_path=dsm_path,
                output_dir=out_dir,
                num_steps=config.inference.num_steps,
                guidance_scale=config.inference.guidance_scale,
                prompt=config.inference.prompt,
                seed=config.inference.seed,
            )
            logger.info(f"  → Saved to {out_dir}")
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
