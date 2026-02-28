"""
Geospatial I/O utilities for DSM2DTM.

Handles TIF file saving/loading with rasterio metadata preservation.
"""

import os
import numpy as np
from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def save_tif_with_metadata(
    data: np.ndarray,
    output_path: str,
    ref_image_path: str = None,
    nodata: float = None,
) -> None:
    """
    Save a 2D numpy array as a GeoTIFF, preserving geospatial metadata from
    a reference image if available.

    Args:
        data: 2D numpy array (H, W) to save.
        output_path: Output .tif file path.
        ref_image_path: Optional reference image to copy geospatial metadata from.
        nodata: Optional NoData value to set.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if HAS_RASTERIO and ref_image_path and os.path.exists(ref_image_path):
        try:
            with rasterio.open(ref_image_path) as src:
                profile = src.profile

            h, w = data.shape
            profile.update(
                dtype=rasterio.float32,
                count=1,
                compress="lzw",
                height=h,
                width=w,
            )
            if nodata is not None:
                profile.update(nodata=nodata)

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(data.astype(np.float32), 1)
            return
        except Exception:
            pass

    # Fallback: save as PIL
    Image.fromarray(data.astype(np.float32)).save(output_path)


def load_tif_with_metadata(path: str):
    """
    Load a TIF file. Returns (data, profile) if rasterio is available,
    otherwise returns (data, None).

    Args:
        path: Path to the TIF file.

    Returns:
        Tuple of (numpy array [H, W], rasterio profile or None).
    """
    if HAS_RASTERIO:
        try:
            with rasterio.open(path) as src:
                data = src.read(1).astype(np.float32)
                profile = src.profile
            return data, profile
        except Exception:
            pass

    # Fallback
    im = Image.open(path)
    if im.mode != "F":
        im = im.convert("F")
    return np.array(im, dtype=np.float32), None


def save_png_uint16(data: np.ndarray, output_path: str) -> None:
    """
    Save a 2D array as a 16-bit PNG (for visualization).
    Auto-stretches to [0, 65535] range.
    """
    mn, mx = data.min(), data.max()
    if mx - mn > 1e-6:
        vis_norm = (data - mn) / (mx - mn)
    else:
        vis_norm = np.zeros_like(data)

    img_u16 = (vis_norm * 65535.0).round().clip(0, 65535).astype(np.uint16)
    Image.fromarray(img_u16, mode="I;16").save(output_path)
