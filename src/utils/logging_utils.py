"""
Logging utilities for DSM2DTM.

Consolidates the setup_logger function that was duplicated across 3 scripts.
"""

import os
import logging
import datetime


def setup_logger(
    output_dir: str,
    name: str = "dsm2dtm",
    prefix: str = "run",
) -> logging.Logger:
    """
    Setup a logger that writes to both console and a timestamped log file.

    Args:
        output_dir: Directory to save the log file.
        name: Logger name.
        prefix: Prefix for the log filename (e.g. "training", "evaluation").

    Returns:
        Configured logger instance.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(output_dir, f"{prefix}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate logs
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
