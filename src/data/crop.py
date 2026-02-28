"""
Data cropping and splitting utility for VAE fine-tuning.

Crops original TIF images into patches and splits into train/val/test.
"""

import os
import glob
import shutil
import random

from tqdm import tqdm

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def crop_and_split_for_vae(
    source_dirs: list,
    output_dir: str = "datasets/new_data_vae",
    crop_size: int = 512,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    temp_dir: str = "all_cropped_patches",
    cleanup_temp: bool = True,
):
    """
    Crop TIF files into patches and split into train/val/test for VAE training.

    Args:
        source_dirs: List of directories containing source TIF files.
        output_dir: Output directory for the split data.
        crop_size: Size of each crop patch.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        seed: Random seed.
        temp_dir: Temporary directory for cropped patches.
        cleanup_temp: Whether to clean up the temp directory after.
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for crop_and_split_for_vae")

    os.makedirs(temp_dir, exist_ok=True)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(output_dir, split, "all_data"), exist_ok=True)

    # Step 1: Find and crop files
    print("--- Step 1: Finding and cropping images ---")
    source_files = []
    for base_dir in source_dirs:
        source_files.extend(glob.glob(os.path.join(base_dir, "**", "*.TIF"), recursive=True))

    if not source_files:
        print(f"ERROR: No .TIF files found in {source_dirs}")
        return

    print(f"Found {len(source_files)} TIF source files. Cropping...")

    cropped_count = 0
    skipped_count = 0

    for file_path in tqdm(source_files, desc="Cropping"):
        try:
            with rasterio.open(file_path) as src:
                height = src.height
                width = src.width

                if height < crop_size or width < crop_size:
                    skipped_count += 1
                    continue

                base_name = os.path.splitext(os.path.basename(file_path))[0]

                # 4 corner crops
                windows = {
                    "TL": Window(0, 0, crop_size, crop_size),
                    "TR": Window(width - crop_size, 0, crop_size, crop_size),
                    "BL": Window(0, height - crop_size, crop_size, crop_size),
                    "BR": Window(width - crop_size, height - crop_size, crop_size, crop_size),
                }

                for corner_name, window in windows.items():
                    crop_data = src.read(window=window)
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "height": crop_size,
                        "width": crop_size,
                        "transform": src.window_transform(window),
                    })

                    out_filename = f"{base_name}_{corner_name}.TIF"
                    out_path = os.path.join(temp_dir, out_filename)

                    with rasterio.open(out_path, "w", **out_meta) as dest:
                        dest.write(crop_data)

                    cropped_count += 1

        except Exception as e:
            print(f"ERROR processing {file_path}: {e}")

    print(f"Cropping complete: {cropped_count} patches created, {skipped_count} files skipped.")

    # Step 2: Split into train/val/test
    print("\n--- Step 2: Splitting data ---")
    all_patches = glob.glob(os.path.join(temp_dir, "*.TIF"))

    if not all_patches:
        print("ERROR: No patches found after cropping.")
        return

    random.seed(seed)
    random.shuffle(all_patches)

    total = len(all_patches)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    splits = {
        "train": all_patches[:train_end],
        "val": all_patches[train_end:val_end],
        "test": all_patches[val_end:],
    }

    for split_name, files in splits.items():
        dest_path = os.path.join(output_dir, split_name, "all_data")
        for f in tqdm(files, desc=f"Copying to {split_name}"):
            shutil.copy(f, dest_path)
        print(f"  {split_name}: {len(files)} files")

    if cleanup_temp:
        shutil.rmtree(temp_dir)

    print(f"\nDone! VAE data ready at: {output_dir}")
