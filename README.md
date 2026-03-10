# DSM2DTM — ControlNet-based Digital Surface Model to Digital Terrain Model Conversion

Convert DSM (Digital Surface Model) tiles to DTM (Digital Terrain Model) using ControlNet on Stable Diffusion 1.5 with a LoRA-finetuned VAE for single-channel elevation data.

## Project Structure

```
dsm2dtm/
├── configs/                        # YAML experiment configurations
│   ├── train_pixel_loss.yaml       # Pixel-loss training (MAE + Gradient)
│   ├── train_noise_loss.yaml       # Noise-loss training (MSE)
│   ├── infer_default.yaml          # Inference defaults
│   └── eval_default.yaml           # Evaluation defaults
├── tools/                          # Entry-point scripts
│   ├── train.py                    # ControlNet training
│   ├── infer.py                    # Run inference on DSM tiles
│   ├── evaluate.py                 # Evaluate predictions vs GT
│   ├── prepare_data.py             # Data prep (split, crop, stats)
│   └── train_vae.py                # VAE LoRA fine-tuning
├── src/                            # Core library
│   ├── utils/                      # Config, logging, prompt encoding, GeoTIFF I/O
│   ├── data/                       # Normalization, datasets, splits, preprocessing
│   ├── models/                     # VAE modifier, ControlNet loader, inference pipeline
│   ├── losses/                     # Loss registry + pixel/noise losses
│   ├── training/                   # Trainer, validation
│   └── evaluation/                 # Metrics, visualization, reporting
├── legacy/                         # Original monolithic scripts (reference)
├── datasets/                       # Data directory
├── models/                         # Pretrained model weights
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

```bash
# Create train/val/test split (80/10/10)
python tools/prepare_data.py split --data_root datasets/new_data --output datasets/split_dataset.json

# Crop tiles for VAE training (512×512)
python tools/prepare_data.py crop --dsm_dir datasets/new_data/dsm --dtm_dir datasets/new_data/dtm --output_dir datasets/new_data_vae

# Check global statistics
python tools/prepare_data.py stats --data_root datasets/new_data
```

### 3. Train ControlNet

```bash
# Pixel-loss approach (prediction_type=sample, MAE+Gradient loss)
python tools/train.py --config configs/train_pixel_loss.yaml

# Noise-loss approach (prediction_type=epsilon, MSE loss)
python tools/train.py --config configs/train_noise_loss.yaml

# Override any config value via CLI
python tools/train.py --config configs/train_pixel_loss.yaml --lr 1e-5 --total_steps 20000
python tools/train.py --config configs/train_pixel_loss.yaml --set training.warmup_steps=100 loss.grad_weight=0.3
```

### 4. Run inference

```bash
# On a single image
python tools/infer.py --config configs/infer_default.yaml --input datasets/new_data/dsm/dsm_42.tif

# On a directory
python tools/infer.py --config configs/infer_default.yaml --input_dir datasets/new_data/dsm

# On the test split
python tools/infer.py --config configs/infer_default.yaml --split test --controlnet_path experiments/pixel_loss/best_model
```

### 5. Evaluate

```bash
python tools/evaluate.py --config configs/eval_default.yaml     --predictions_dir inference_output     --gt_dir datasets/new_data/ndsm
```

## Two Training Strategies

| | Pixel Loss | Noise Loss |
|---|---|---|
| **prediction_type** | `sample` | `epsilon` |
| **Loss** | MAE + Sobel Gradient | MSE |
| **Normalization** | `log_global` | `percentile` |
| **Config** | `configs/train_pixel_loss.yaml` | `configs/train_noise_loss.yaml` |

> **Important**: Normalization must match between training and inference. The config system enforces this automatically.

## Configuration System

All settings are in YAML files under `configs/`. Override any value:

```bash
# Flat CLI shortcuts
python tools/train.py --config configs/train_pixel_loss.yaml --lr 1e-5 --batch_size 2

# Dot-notation override for any nested field
python tools/train.py --config configs/train_pixel_loss.yaml --set training.gradient_accumulation_steps=4

# Multiple overrides
python tools/train.py --config configs/train_pixel_loss.yaml \
    --set loss.mae_weight=0.8 loss.grad_weight=0.7 training.val_every=500
```

## Adding a Custom Loss

1. Create `src/losses/my_loss.py`:

```python
import torch
from .registry import register_loss

@register_loss("my_custom")
class MyCustomLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target, **kwargs):
        loss = ...  # your loss
        return {"loss": loss}
```

2. Import it in `src/losses/__init__.py` and use it in YAML:

```yaml
loss:
  type: "my_custom"
  alpha: 2.0
```

## Key Technical Details

### VAE Modification
The standard SD1.5 VAE (3-channel RGB) is modified for single-channel elevation data:
- Encoder `conv_in`: 3→1 channel (weights averaged)
- Decoder `conv_out`: 3→1 channel (weights averaged)
- LoRA adapter fine-tuned on elevation-domain crops

### Normalization Strategies
- **log_global**: Log transform + global min/max scaling to [-1, 1]. DSM range [760, 1450], DTM range [0, 650].
- **percentile**: Per-image percentile-based normalization (1st/99th percentile).

### Pixel Loss (prediction_type=sample)
$L = \lambda_{mae} \cdot L_{mae} + \lambda_{grad} \cdot L_{grad}$
- MAE: L1 between predicted and target latents
- Gradient: L1 between Sobel gradients (preserves edges/terrain features)

### Noise Loss (prediction_type=epsilon)
Standard MSE between predicted noise and actual noise.

## VAE Fine-tuning

```bash
python tools/train_vae.py \
    --data_dir datasets/new_data_vae \
    --output_dir models/vae_lora_new \
    --lora_rank 4 --epochs 50 --lr 1e-4
```

## Legacy Scripts

Original monolithic scripts are preserved in `legacy/` for reference. The new `src/` + `tools/` structure replaces them with shared, configurable modules.
