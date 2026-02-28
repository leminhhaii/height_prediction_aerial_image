"""
Data preprocessing utilities for DSM2DTM.

Consolidates load_image (6 copies), pad_image (4 copies),
pad_or_crop (4 copies), and extract_index (3 copies) into single implementations.
"""

import re
import numpy as np
from pathlib import Path
from PIL import Image


def load_image(path) -> np.ndarray:
    """
    Load a single-channel elevation image as float32 numpy array.

    Args:
        path: Path to the image file (TIF, PNG, etc.)

    Returns:
        2D numpy array of shape (H, W) with float32 dtype.
    """
    p = str(path)
    im = Image.open(p)
    if im.mode != "F":
        im = im.convert("F")
    arr = np.array(im, dtype=np.float32)
    return arr


def pad_or_crop(arr: np.ndarray, crop_size: int) -> np.ndarray:
    """
    Pad (edge-mode) if smaller than crop_size, then center-crop to crop_size.
    Used during training to ensure fixed-size inputs.

    Args:
        arr: 2D numpy array (H, W).
        crop_size: Target size for both dimensions.

    Returns:
        2D array of shape (crop_size, crop_size).
    """
    h, w = arr.shape
    cs = crop_size

    # Pad if needed
    if h < cs or w < cs:
        pad_h = max(0, cs - h)
        pad_w = max(0, cs - w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
        h, w = arr.shape

    # Center crop
    start_h = (h - cs) // 2
    start_w = (w - cs) // 2
    arr = arr[start_h : start_h + cs, start_w : start_w + cs]
    return arr


def pad_image(arr: np.ndarray, min_size: int) -> np.ndarray:
    """
    Pad image to at least min_size and ensure dimensions are divisible by 8
    (required for VAE encoding). Used during inference.

    Args:
        arr: 2D numpy array (H, W).
        min_size: Minimum size for each dimension.

    Returns:
        Padded 2D array with dimensions divisible by 8.
    """
    h, w = arr.shape

    # First, pad to min_size if needed (center-pad)
    if h < min_size or w < min_size:
        pad_h = max(0, min_size - h)
        pad_w = max(0, min_size - w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
        h, w = arr.shape

    # Ensure divisible by 8 for VAE
    new_h = ((h + 7) // 8) * 8
    new_w = ((w + 7) // 8) * 8

    if new_h != h or new_w != w:
        arr = np.pad(arr, ((0, new_h - h), (0, new_w - w)), mode="edge")

    return arr


def extract_index(filename: str) -> int:
    """
    Extract the numeric index from a filename like 'dsm_249.TIF' → 249.

    Args:
        filename: Filename string.

    Returns:
        Integer index, or None if no number found.
    """
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else None
