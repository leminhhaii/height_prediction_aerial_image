#!/usr/bin/env python
"""
DSM2DTM – Data preparation tools.

Subcommands:
    split    – Create train/val/test split JSON
    crop     – Crop and split large tiles for VAE training
    stats    – Calculate global statistics

Usage:
    python tools/prepare_data.py split --data_root datasets/new_data --output datasets/split_dataset.json
    python tools/prepare_data.py crop --dsm_dir datasets/new_data/dsm --dtm_dir datasets/new_data/dtm --output_dir datasets/new_data_vae --crop_size 512
    python tools/prepare_data.py stats --data_root datasets/new_data
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def cmd_split(args):
    """Create dataset split."""
    from src.data.split import create_split

    split = create_split(
        data_root=args.data_root,
        dsm_dir=args.dsm_subdir,
        dtm_dir=args.dtm_subdir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        output_path=args.output,
        seed=args.seed,
    )
    for k, v in split.items():
        print(f"  {k}: {len(v)} images")


def cmd_crop(args):
    """Crop and split tiles for VAE training."""
    from src.data.crop import crop_and_split_for_vae

    crop_and_split_for_vae(
        dsm_dir=args.dsm_dir,
        dtm_dir=args.dtm_dir,
        output_dir=args.output_dir,
        crop_size=args.crop_size,
        split_json=args.split_json,
    )
    print("Done.")


def cmd_stats(args):
    """Calculate global statistics."""
    import glob
    import numpy as np
    import rasterio

    data_root = args.data_root
    dsm_dir = os.path.join(data_root, args.dsm_subdir)
    dtm_dir = os.path.join(data_root, args.dtm_subdir)

    for name, d in [("DSM", dsm_dir), ("DTM/nDSM", dtm_dir)]:
        files = sorted(glob.glob(os.path.join(d, "*.tif")))
        if not files:
            print(f"  {name}: no .tif files in {d}")
            continue

        all_min, all_max = float("inf"), float("-inf")
        for f in files:
            with rasterio.open(f) as ds:
                arr = ds.read(1).astype(np.float64)
                valid = arr[(arr > -10000) & (arr < 10000)]
                if len(valid) > 0:
                    all_min = min(all_min, valid.min())
                    all_max = max(all_max, valid.max())
        print(f"  {name}: min={all_min:.2f}, max={all_max:.2f}  ({len(files)} files in {d})")


def main():
    parser = argparse.ArgumentParser(description="DSM2DTM data preparation tools")
    sub = parser.add_subparsers(dest="command", required=True)

    # split
    p_split = sub.add_parser("split", help="Create train/val/test split")
    p_split.add_argument("--data_root", type=str, required=True)
    p_split.add_argument("--dsm_subdir", type=str, default="dsm")
    p_split.add_argument("--dtm_subdir", type=str, default="ndsm")
    p_split.add_argument("--output", type=str, default="datasets/split_dataset.json")
    p_split.add_argument("--train_ratio", type=float, default=0.8)
    p_split.add_argument("--val_ratio", type=float, default=0.1)
    p_split.add_argument("--seed", type=int, default=42)
    p_split.set_defaults(func=cmd_split)

    # crop
    p_crop = sub.add_parser("crop", help="Crop and split tiles for VAE training")
    p_crop.add_argument("--dsm_dir", type=str, required=True)
    p_crop.add_argument("--dtm_dir", type=str, required=True)
    p_crop.add_argument("--output_dir", type=str, required=True)
    p_crop.add_argument("--crop_size", type=int, default=512)
    p_crop.add_argument("--split_json", type=str, default=None)
    p_crop.set_defaults(func=cmd_crop)

    # stats
    p_stats = sub.add_parser("stats", help="Calculate global data statistics")
    p_stats.add_argument("--data_root", type=str, required=True)
    p_stats.add_argument("--dsm_subdir", type=str, default="dsm")
    p_stats.add_argument("--dtm_subdir", type=str, default="ndsm")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
