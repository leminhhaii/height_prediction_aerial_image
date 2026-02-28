"""
Configuration system for DSM2DTM.

Loads experiment configs from YAML files, with CLI override support.
"""

import os
import yaml
import argparse
from dataclasses import dataclass, field, fields, asdict
from typing import Optional, List, Any
from pathlib import Path

import torch


@dataclass
class ModelConfig:
    """Model-related configuration."""
    base_model: str = "runwayml/stable-diffusion-v1-5"
    vae_path: str = "models/vae_model_fintuned_lora_2"
    controlnet_path: Optional[str] = None  # None = init from UNet (training), path = load (inference)
    prediction_type: str = "sample"  # "sample" for pixel-loss, "epsilon" for noise-loss


@dataclass
class DataConfig:
    """Data-related configuration."""
    data_root: str = "datasets/new_data/"
    split_json: str = "datasets/split_dataset.json"
    dsm_dir: str = "dsm"
    dtm_dir: str = "ndsm"
    crop_size: int = 704

    # Normalization strategy: "log_global" or "percentile"
    normalization: str = "log_global"

    # Global stats for log_global normalization
    dsm_global_min: float = 760.0
    dsm_global_max: float = 1450.0
    dtm_global_min: float = 0.0
    dtm_global_max: float = 650.0

    # Percentile normalization params
    percentile_low: float = 1.0
    percentile_high: float = 99.0


@dataclass
class TrainingConfig:
    """Training-related configuration."""
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    total_steps: int = 10000
    warmup_steps: int = 500
    save_every: int = 2000
    val_every: int = 1000
    log_every: int = 100
    num_workers: int = 2


@dataclass
class LossConfig:
    """Loss function configuration."""
    type: str = "pixel"  # "pixel" or "noise"
    mae_weight: float = 1.0
    grad_weight: float = 0.5


@dataclass
class InferenceConfig:
    """Inference-related configuration."""
    num_steps: int = 50
    guidance_scale: float = 1.0
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = 42
    num_samples: int = 2

    # Training-time inference preview
    infer_every: int = 1000
    fixed_infer_pair_index: int = 0


@dataclass
class SchedulerConfig:
    """Noise scheduler configuration."""
    beta_start: float = 0.00085
    beta_end: float = 0.012
    num_train_timesteps: int = 1000


@dataclass
class OutputConfig:
    """Output-related configuration."""
    dir: str = "./experiments/default"
    unit: str = "ft"  # Unit label for metrics: "ft" or "m"


@dataclass
class EvaluationConfig:
    """Evaluation-specific configuration."""
    elevation_bins: List[float] = field(
        default_factory=lambda: [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350]
    )
    n_worst: int = 10


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Runtime
    fp16: bool = True
    seed: int = 42
    vae_scale_factor: float = 0.18215
    prompt_dropout_prob: float = 0.2

    # Device (set at runtime)
    _device: Optional[str] = None

    @property
    def device(self) -> torch.device:
        if self._device:
            return torch.device(self._device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @device.setter
    def device(self, value):
        if isinstance(value, torch.device):
            self._device = str(value)
        else:
            self._device = value


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively update base dict with override dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _dict_to_config(d: dict) -> Config:
    """Convert a nested dictionary to a Config dataclass."""
    cfg = Config()

    if "model" in d:
        cfg.model = ModelConfig(**{k: v for k, v in d["model"].items() if k in {f.name for f in fields(ModelConfig)}})
    if "data" in d:
        cfg.data = DataConfig(**{k: v for k, v in d["data"].items() if k in {f.name for f in fields(DataConfig)}})
    if "training" in d:
        cfg.training = TrainingConfig(**{k: v for k, v in d["training"].items() if k in {f.name for f in fields(TrainingConfig)}})
    if "loss" in d:
        cfg.loss = LossConfig(**{k: v for k, v in d["loss"].items() if k in {f.name for f in fields(LossConfig)}})
    if "inference" in d:
        cfg.inference = InferenceConfig(**{k: v for k, v in d["inference"].items() if k in {f.name for f in fields(InferenceConfig)}})
    if "scheduler" in d:
        cfg.scheduler = SchedulerConfig(**{k: v for k, v in d["scheduler"].items() if k in {f.name for f in fields(SchedulerConfig)}})
    if "output" in d:
        cfg.output = OutputConfig(**{k: v for k, v in d["output"].items() if k in {f.name for f in fields(OutputConfig)}})
    if "evaluation" in d:
        eval_d = d["evaluation"]
        cfg.evaluation = EvaluationConfig(**{k: v for k, v in eval_d.items() if k in {f.name for f in fields(EvaluationConfig)}})

    # Top-level scalars
    for key in ("fp16", "seed", "vae_scale_factor", "prompt_dropout_prob", "_device"):
        if key in d:
            setattr(cfg, key, d[key])

    return cfg


def load_config(yaml_path: str, cli_overrides: Optional[dict] = None) -> Config:
    """
    Load configuration from a YAML file, with optional CLI overrides.

    Args:
        yaml_path: Path to the YAML config file.
        cli_overrides: Optional dict of dot-separated key overrides.
            e.g. {"training.learning_rate": 1e-4, "loss.type": "noise"}
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Apply CLI overrides using dot-separated keys
    if cli_overrides:
        for dotted_key, value in cli_overrides.items():
            keys = dotted_key.split(".")
            d = raw
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value

    return _dict_to_config(raw)


def config_to_dict(cfg: Config) -> dict:
    """Convert Config to a serializable dict (for logging/saving)."""
    result = {}
    for f in fields(Config):
        val = getattr(cfg, f.name)
        if hasattr(val, "__dataclass_fields__"):
            result[f.name] = asdict(val)
        elif f.name == "_device":
            result["device"] = str(cfg.device)
        else:
            result[f.name] = val
    return result


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common config CLI arguments to an argparse parser."""
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    # Common overrides as flat CLI args
    parser.add_argument("--lr", type=float, default=None, help="Override training.learning_rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training.batch_size")
    parser.add_argument("--total_steps", type=int, default=None, help="Override training.total_steps")
    parser.add_argument("--loss_type", type=str, default=None, help="Override loss.type")
    parser.add_argument("--prediction_type", type=str, default=None, help="Override model.prediction_type")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output.dir")
    parser.add_argument("--controlnet_path", type=str, default=None, help="Override model.controlnet_path")
    parser.add_argument("--vae_path", type=str, default=None, help="Override model.vae_path")
    parser.add_argument("--steps", type=int, default=None, help="Override inference.num_steps")
    parser.add_argument("--guidance_scale", type=float, default=None, help="Override inference.guidance_scale")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--normalization", type=str, default=None, help="Override data.normalization")
    parser.add_argument("--split_json", type=str, default=None, help="Override data.split_json")

    # Generic dot-notation override (supports repeated --set flags)
    parser.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                        help="Override any config value using dot-notation, e.g. --set training.warmup_steps=100")

    return parser


def parse_config_args(args: argparse.Namespace) -> Config:
    """Parse CLI args into a Config object."""
    # Build overrides dict from flat CLI args
    overrides = {}

    _FLAT_MAP = {
        "lr": "training.learning_rate",
        "batch_size": "training.batch_size",
        "total_steps": "training.total_steps",
        "loss_type": "loss.type",
        "prediction_type": "model.prediction_type",
        "output_dir": "output.dir",
        "controlnet_path": "model.controlnet_path",
        "vae_path": "model.vae_path",
        "steps": "inference.num_steps",
        "guidance_scale": "inference.guidance_scale",
        "seed": "seed",
        "normalization": "data.normalization",
        "split_json": "data.split_json",
    }

    for cli_key, dotted_key in _FLAT_MAP.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            overrides[dotted_key] = val

    # Parse --set KEY=VALUE pairs
    for item in getattr(args, "set", []):
        if "=" not in item:
            raise ValueError(f"Invalid --set format: '{item}'. Expected KEY=VALUE")
        key, raw_val = item.split("=", 1)
        # Try to auto-convert types
        try:
            val = int(raw_val)
        except ValueError:
            try:
                val = float(raw_val)
            except ValueError:
                if raw_val.lower() in ("true", "false"):
                    val = raw_val.lower() == "true"
                else:
                    val = raw_val
        overrides[key] = val

    return load_config(args.config, cli_overrides=overrides if overrides else None)
