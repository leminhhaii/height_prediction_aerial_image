#!/usr/bin/env python
"""
DSM2DTM – Training entry point.

Usage:
    python tools/train.py --config configs/train_pixel_loss.yaml
    python tools/train.py --config configs/train_noise_loss.yaml --lr 1e-5
    python tools/train.py --config configs/train_pixel_loss.yaml --set training.total_steps=5000
"""

import sys
import os
import argparse

# Allow running from repo root: python tools/train.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import add_config_args, parse_config_args
from src.training.trainer import ControlNetTrainer


def main():
    parser = argparse.ArgumentParser(description="Train ControlNet for DSM→DTM conversion")
    add_config_args(parser)
    args = parser.parse_args()

    config = parse_config_args(args)
    trainer = ControlNetTrainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
