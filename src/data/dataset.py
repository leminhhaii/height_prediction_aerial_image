"""
Dataset classes for DSM2DTM training.

Provides DSMPairDataset and PairedDSMDataset that work with any
NormalizationStrategy.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import load_image, pad_or_crop
from .normalization import NormalizationStrategy


class DSMPairDataset(Dataset):
    """
    Dataset for DSM-DTM pairs using a file list.

    Directory structure:
        data_root/dsm_dir/*.TIF
        data_root/dtm_dir/*.TIF

    Files are matched by name from the provided file_list.
    """

    def __init__(
        self,
        root: str,
        file_list: List[str],
        normalizer: NormalizationStrategy,
        dsm_dir: str = "dsm",
        dtm_dir: str = "ndsm",
        crop_size: int = 704,
    ):
        self.root = Path(root)
        self.dsm_dir = self.root / dsm_dir
        self.dtm_dir = self.root / dtm_dir
        self.crop_size = crop_size
        self.fnames = file_list
        self.normalizer = normalizer

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx: int):
        name = self.fnames[idx]
        dsm_p = self.dsm_dir / name
        dtm_p = self.dtm_dir / name

        dsm = load_image(dsm_p)
        dtm = load_image(dtm_p)

        dsm = pad_or_crop(dsm, self.crop_size)
        dtm = pad_or_crop(dtm, self.crop_size)

        dsm_n = self.normalizer.normalize(dsm, data_type="dsm")
        dtm_n = self.normalizer.normalize(dtm, data_type="dtm")

        # ControlNet expects 3-channel input for condition
        dsm_t = torch.from_numpy(dsm_n).unsqueeze(0).repeat(3, 1, 1)
        dtm_t = torch.from_numpy(dtm_n).unsqueeze(0)

        return {
            "dsm": dsm_t,
            "dtm": dtm_t,
            "fname": name,
        }


class PairedDSMDataset(Dataset):
    """
    Dataset for DSM-DTM pairs using paired (dsm_name, dtm_name) tuples.
    This allows DSM and DTM files to have different naming conventions.
    """

    def __init__(
        self,
        root: str,
        paired_files: List[Tuple[str, str]],
        normalizer: NormalizationStrategy,
        dsm_dir: str = "dsm",
        dtm_dir: str = "ndsm",
        crop_size: int = 704,
    ):
        self.root = Path(root)
        self.dsm_dir = self.root / dsm_dir
        self.dtm_dir = self.root / dtm_dir
        self.crop_size = crop_size
        self.paired_files = paired_files
        self.normalizer = normalizer

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx: int):
        dsm_name, dtm_name = self.paired_files[idx]
        dsm_p = self.dsm_dir / dsm_name
        dtm_p = self.dtm_dir / dtm_name

        dsm = load_image(dsm_p)
        dtm = load_image(dtm_p)

        dsm = pad_or_crop(dsm, self.crop_size)
        dtm = pad_or_crop(dtm, self.crop_size)

        dsm_n = self.normalizer.normalize(dsm, data_type="dsm")
        dtm_n = self.normalizer.normalize(dtm, data_type="dtm")

        dsm_t = torch.from_numpy(dsm_n).unsqueeze(0).repeat(3, 1, 1)
        dtm_t = torch.from_numpy(dtm_n).unsqueeze(0)

        return {
            "dsm": dsm_t,
            "dtm": dtm_t,
            "fname": f"{dsm_name}_{dtm_name}",
        }


def collate_fn(batch):
    """Custom collate function for DSM-DTM datasets."""
    dsm = torch.stack([b["dsm"] for b in batch])
    dtm = torch.stack([b["dtm"] for b in batch])
    fnames = [b["fname"] for b in batch]
    return {"dsm": dsm, "dtm": dtm, "fnames": fnames}
