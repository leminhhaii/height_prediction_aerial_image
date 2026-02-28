"""
Dataset split creation utility.

Creates train/val/test splits from paired DSM/DTM files and saves to JSON.
"""

import os
import json
import random
from pathlib import Path
from typing import Optional

from .preprocessing import extract_index


def create_split(
    data_root: str,
    dsm_dir: str = "dsm",
    dtm_dir: str = "ndsm",
    output_path: str = "datasets/split_dataset.json",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    file_extension: str = "*.TIF",
) -> dict:
    """
    Scan DSM/DTM directories, match files by numeric index,
    and create a train/val/test split.

    Args:
        data_root: Root directory containing DSM and DTM subdirectories.
        dsm_dir: Name of the DSM subdirectory.
        dtm_dir: Name of the DTM subdirectory.
        output_path: Path to save the split JSON file.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        seed: Random seed for reproducibility.
        file_extension: Glob pattern for file extension.

    Returns:
        Dictionary with 'train', 'val', 'test' keys.
    """
    dsm_path = Path(data_root) / dsm_dir
    dtm_path = Path(data_root) / dtm_dir

    if not dsm_path.exists():
        raise FileNotFoundError(f"DSM directory not found: {dsm_path}")
    if not dtm_path.exists():
        raise FileNotFoundError(f"DTM directory not found: {dtm_path}")

    all_dsm_files = sorted([p.name for p in dsm_path.glob(file_extension)])
    all_dtm_files = sorted([p.name for p in dtm_path.glob(file_extension)])

    print(f"Found {len(all_dsm_files)} DSM files and {len(all_dtm_files)} DTM files")

    dsm_by_index = {extract_index(f): f for f in all_dsm_files if extract_index(f) is not None}
    dtm_by_index = {extract_index(f): f for f in all_dtm_files if extract_index(f) is not None}

    common_indices = sorted(set(dsm_by_index.keys()) & set(dtm_by_index.keys()))
    print(f"Found {len(common_indices)} matching DSM-DTM pairs")

    paired_files = [(dsm_by_index[idx], dtm_by_index[idx]) for idx in common_indices]

    random.seed(seed)
    random.shuffle(paired_files)

    num_total = len(paired_files)
    num_train = int(train_ratio * num_total)
    num_val = int(val_ratio * num_total)

    train_pairs = paired_files[:num_train]
    val_pairs = paired_files[num_train : num_train + num_val]
    test_pairs = paired_files[num_train + num_val :]

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    split_data = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
    }

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(split_data, f, indent=4)

    print(f"Split data saved to {output_path}")
    return split_data


def load_split(split_json_path: str) -> dict:
    """Load a split JSON file."""
    with open(split_json_path, "r") as f:
        return json.load(f)
