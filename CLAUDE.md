# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DSM2DTM** — Converts Digital Surface Models (DSM, terrain with buildings/trees) to Digital Terrain Models (DTM, bare earth elevation) using ControlNet on Stable Diffusion 1.5 with a LoRA-finetuned VAE for single-channel elevation data.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Data preparation
python tools/prepare_data.py split --data_root datasets/new_data --output datasets/split_dataset.json
python tools/prepare_data.py crop --dsm_dir datasets/new_data/dsm --dtm_dir datasets/new_data/dtm --output_dir datasets/new_data_vae
python tools/prepare_data.py stats --data_root datasets/new_data

# Train ControlNet
python tools/train.py --config configs/train_pixel_loss.yaml       # Latent-space pixel loss
python tools/train.py --config configs/train_pixel_decoded.yaml    # Pixel-space decoded loss (via VAE)
python tools/train.py --config configs/train_noise_loss.yaml       # Noise loss

# Verify losses (CPU, no GPU required)
python tools/verify_pixel_decoded_loss.py

# Override config values via CLI
python tools/train.py --config configs/train_pixel_loss.yaml --lr 1e-5 --batch_size 2
python tools/train.py --config configs/train_pixel_loss.yaml --set training.warmup_steps=100 loss.grad_weight=0.3

# Fine-tune VAE with LoRA
python tools/train_vae.py --data_dir datasets/new_data_vae --output_dir models/vae_lora_new --lora_rank 4 --epochs 50 --lr 1e-4

# Inference
python tools/infer.py --config configs/infer_default.yaml --input datasets/new_data/dsm/dsm_42.tif
python tools/infer.py --config configs/infer_default.yaml --input_dir datasets/new_data/dsm
python tools/infer.py --config configs/infer_default.yaml --split test --controlnet_path experiments/pixel_loss/best_model

# Evaluation
python tools/evaluate.py --config configs/eval_default.yaml --predictions_dir inference_output --gt_dir datasets/new_data/ndsm
```

## Architecture

### Code Layout

- **`tools/`** — Entry-point scripts (train, infer, evaluate, prepare_data, train_vae). All user-facing commands live here.
- **`src/`** — Core library imported by tools.
- **`configs/`** — YAML experiment configurations. All settings flow through these files.
- **`legacy/`** — Original monolithic scripts kept for reference only.
- **`datasets/`**, **`models/`**, **`experiments/`** — Data, pretrained weights, and training outputs (all git-ignored).

### Training Strategies

| | Pixel Loss | Pixel Decoded Loss | Noise Loss |
|---|---|---|---|
| prediction_type | `sample` | `sample` | `epsilon` |
| Loss | MAE + Sobel Gradient (latent space) | MAE + Sobel Gradient (pixel space via VAE) | MSE |
| Normalization | `log_global` | `log_global` | `percentile` |
| Config | `train_pixel_loss.yaml` | `train_pixel_decoded.yaml` | `train_noise_loss.yaml` |

**Pixel Decoded Loss** decodes predicted latents through the frozen VAE to pixel space before computing losses. Uses `vae.enable_gradient_checkpointing()` + `vae.enable_slicing()` to fit within 24GB VRAM. The `set_vae()` method is called automatically by the trainer via `hasattr` check.

**Normalization must match between training and inference.** The config system enforces this.

### Configuration System (`src/utils/config.py`)

Dataclass-based configs loaded from YAML. Supports:
- Flat CLI shortcuts: `--lr`, `--batch_size`
- Dot-notation overrides: `--set training.warmup_steps=100 loss.grad_weight=0.3`
- Key dataclasses: `ModelConfig`, `DataConfig`, `TrainingConfig`, `LossConfig`, `InferenceConfig`, `EvaluationConfig`

### Training Pipeline

Only **ControlNet** weights are trained. VAE, UNet, and Text Encoder are frozen.
- `src/training/trainer.py` — `ControlNetTrainer`: main training loop with gradient accumulation and fp16 mixed precision
- `src/training/validation.py` — Validation with RMSE in real elevation units for checkpointing
- Best model saved by lowest validation RMSE

### Loss Functions (`src/losses/`)

Uses a **registry pattern** — new losses are added via `@register_loss("name")` decorator, then referenced by name in YAML config. Import new losses in `src/losses/__init__.py`.
- **PixelLoss** (`pixel`): L1 on latents + L1 on Sobel gradients (preserves terrain edges)
- **PixelDecodedLoss** (`pixel_decoded`): Same as PixelLoss but computed in decoded pixel space via frozen VAE. Losses that need the VAE implement `set_vae(vae, scale_factor)` — the trainer calls this automatically.
- **NoiseLoss** (`noise`): Standard MSE on predicted noise

### Data Pipeline (`src/data/`)

- **Normalization strategies** (`normalization.py`): `LogGlobalNorm` (global log min/max → [-1,1]) and `PercentileNorm` (per-image 1st/99th percentile → [-1,1])
- **Datasets** (`dataset.py`): `PairedDSMDataset` and `DSMPairDataset` match DSM/DTM by filename index
- DSM inputs are broadcast to 3 channels for ControlNet; DTM targets remain single-channel

### VAE Modification (`src/models/vae_modifier.py`)

SD1.5 VAE adapted for single-channel elevation: `conv_in` (3→1) and `conv_out` (3→1) by weight averaging, then LoRA fine-tuned.

### Inference (`src/models/pipeline.py`)

`DSM2DTMPipeline` — loads ControlNet + base model, runs DDPM sampling, denormalizes, and saves as GeoTIFF preserving rasterio metadata (CRS, transforms).

### Evaluation (`src/evaluation/`)

- `metrics.py`: RMSE, MAE, MSE in real elevation units (feet)
- `visualization.py`: Error maps and elevation-stratified plots
- `report.py`: Summary generation with worst-case analysis
