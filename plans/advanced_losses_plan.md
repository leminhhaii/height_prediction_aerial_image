# Plan: Advanced Loss Functions for DSM2DTM ControlNet Training

Saved for later implementation. See the research document: `Enhancing nDSM Prediction with Advanced Losses.md`

## Context

The current training pipeline has two loss strategies — `PixelLoss` (MAE + Sobel gradient on latents, prediction_type="sample") and `NoiseLoss` (MSE on noise, prediction_type="epsilon"). The research document identifies two critical failure modes:

1. **Ground-class imbalance**: Ground pixels (80-95% of area) dominate gradient descent, producing deceptively low RMSE but visually poor elevated structures.
2. **Blurred height discontinuities**: Standard losses don't enforce sharp object boundaries or distinguish which side of an edge is object vs ground.

**Goal**: Implement 6 new loss modules + a composite wrapper with curriculum scheduling, each as a standalone experiment. Existing `PixelLoss` and `NoiseLoss` remain completely unchanged.

---

## Architecture: How New Losses Integrate

```
                    YAML config (loss.type)
                           │
                    ┌──────┴──────┐
                    │  get_loss() │  ← existing registry, no changes
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           │               │               │
     existing types    new standalone    composite
     "pixel","noise"   "bmse","focal"   "composite"
     (UNCHANGED)       "dbr","dir_grad"  ├── sub-loss 1
                       "laplacian"       ├── sub-loss 2
                       "surface_normal"  └── curriculum scheduler
```

**Key design**: All new losses operate on **latent-space tensors** `[B, 4, H/8, W/8]` (same as existing PixelLoss). For BMSE/DBR, per-pixel weight maps are computed from `dtm_target` in pixel space then downsampled to latent resolution. This avoids the ~40% overhead of VAE decoding every step.

---

## Files to Create (8 new files)

### 1. `src/losses/gradient_utils.py` — Shared gradient operators

Extract `sobel_gradients()` from `pixel_loss.py` into a shared module. Add Laplacian kernel. Both `pixel_loss.py` and new losses will import from here.

- `sobel_gradients(x) -> (grad_x, grad_y)` — existing code, moved here
- `laplacian(x) -> lap` — discrete Laplacian via `[[0,1,0],[1,-4,1],[0,1,0]]` kernel
- `gradient_loss(pred, target)` — existing code, stays in `pixel_loss.py` (it's PixelLoss-specific)

### 2. `src/losses/bmse_loss.py` — Balanced MSE Loss (`@register_loss("bmse")`)

Addresses ground-class imbalance by weighting errors inversely proportional to height-bin frequency.

```
forward(pred_latents, target_latents, dtm_target=None, **kwargs):
  1. Compute histogram of dtm_target values → bin frequencies
  2. Weight = 1 / (frequency + margin), normalize to mean=1, clamp [0.1, 10]
  3. Assign weight per pixel based on its height bin
  4. Downsample weight map: F.adaptive_avg_pool2d to latent resolution
  5. Broadcast across 4 latent channels
  6. Return: mean(weight_map * (pred_latents - target_latents)^2)
```

Config params: `num_bins=64`, `sigma=1.0`, `bmse_margin=1.0`

### 3. `src/losses/focal_regression_loss.py` — Focal Regression (`@register_loss("focal_regression")`)

Dynamically scales loss based on prediction difficulty — down-weights easy ground, up-weights hard building errors.

```
forward(pred_latents, target_latents, **kwargs):
  1. error = |pred - target|
  2. focal_weight = (error / (max_error + eps))^gamma + beta
  3. Return: mean(focal_weight * error)
```

Config params: `focal_gamma=2.0`, `focal_beta=1.0`

### 4. `src/losses/dbr_loss.py` — Density-Based Relevance (`@register_loss("dbr")`)

KDE-based inverse density weighting with sqrt compression.

```
forward(pred_latents, target_latents, dtm_target=None, **kwargs):
  1. Compute KDE density from dtm_target pixel values
  2. relevance = 1 / sqrt(density + eps), normalize to mean=1
  3. Downsample to latent resolution, broadcast
  4. Return: mean(relevance * |pred - target|)
```

Config params: `num_bins=64`, `kde_bandwidth=1.0`

### 5. `src/losses/directional_gradient_loss.py` — Direction-Aligned Gradient (`@register_loss("directional_gradient")`)

Replaces isotropic gradient magnitude with sign-preserving directional loss. Penalizes sign mismatches (predicting upslope where ground truth is downslope) more heavily.

```
forward(pred_latents, target_latents, **kwargs):
  1. Compute Sobel ∂x, ∂y for pred and target (via gradient_utils)
  2. Base L1: |∂x_pred - ∂x_target|, |∂y_pred - ∂y_target|
  3. Sign mismatch mask: where pred and target gradients have opposite signs
  4. Amplify loss at mismatched locations by sign_penalty multiplier
  5. Return: mean(weighted_∂x) + mean(weighted_∂y)
```

Config params: `sign_penalty=2.0`

### 6. `src/losses/laplacian_loss.py` — Laplacian Loss (`@register_loss("laplacian")`)

Second-order derivative loss. L1 norm enforces sparsity — consolidates edges into sharp single-pixel boundaries.

```
forward(pred_latents, target_latents, **kwargs):
  1. Compute ∇²pred and ∇²target via Laplacian kernel (from gradient_utils)
  2. Return: L1(∇²pred, ∇²target)  [or L2, configurable]
```

Config params: `laplacian_norm="l1"`

### 7. `src/losses/surface_normal_loss.py` — Surface Normal Loss (`@register_loss("surface_normal")`)

Enforces 3D geometric consistency via cosine distance between predicted and GT surface normals.

```
forward(pred_latents, target_latents, **kwargs):
  1. Compute Sobel ∂x, ∂y for pred and target
  2. Normal vector = normalize(-∂x, -∂y, 1) per pixel
  3. Cosine similarity between pred and target normals
  4. Return: mean(1 - cosine_sim)
```

Config params: none needed (operates on latents by default)

### 8. `src/losses/composite_loss.py` — Composite Wrapper + Curriculum (`@register_loss("composite")`)

Orchestrates multiple sub-losses with optional phased curriculum scheduling.

```python
__init__(components=[...], curriculum={...}, **kwargs):
  - Instantiate each sub-loss via get_loss() recursively
  - Parse curriculum phases

forward(**kwargs):
  - Read global_step from kwargs
  - For each sub-loss:
    - Compute effective weight (base_weight * curriculum_ramp)
    - Skip if weight ≈ 0 (phase not yet active)
    - Call sub-loss forward, accumulate weighted total
  - Return: {"loss": total, ...sub_metrics}
```

Curriculum annealing: when a sub-loss activates at step S, its weight ramps linearly from 0 to configured weight over `anneal_steps`.

---

## Files to Modify (3 existing files)

### 9. `src/losses/__init__.py` — Add imports for all new modules

```python
# Add these imports (existing imports stay):
from .bmse_loss import BalancedMSELoss
from .focal_regression_loss import FocalRegressionLoss
from .dbr_loss import DBRLoss
from .directional_gradient_loss import DirectionalGradientLoss
from .laplacian_loss import LaplacianLoss
from .surface_normal_loss import SurfaceNormalLoss
from .composite_loss import CompositeLoss
```

### 10. `src/utils/config.py` — Extend LossConfig (lines 64-69)

Add new fields with defaults so existing configs keep working:

```python
@dataclass
class LossConfig:
    type: str = "pixel"
    # Existing (unchanged)
    mae_weight: float = 1.0
    grad_weight: float = 0.5
    # NEW: BMSE params
    num_bins: int = 64
    sigma: float = 1.0
    bmse_margin: float = 1.0
    # NEW: Focal regression
    focal_gamma: float = 2.0
    focal_beta: float = 1.0
    # NEW: DBR
    kde_bandwidth: float = 1.0
    # NEW: Directional gradient
    sign_penalty: float = 2.0
    # NEW: Laplacian
    laplacian_norm: str = "l1"
    # NEW: Composite
    components: Optional[list] = None
    curriculum: Optional[dict] = None
```

All new fields have defaults → existing YAML configs parse identically.

### 11. `src/training/trainer.py` — Pass extra kwargs to loss (lines 241-253)

**Only change**: add `dtm_target`, `timestep`, and `global_step` to loss function calls.

Before (line 242-253):
```python
if cfg.model.prediction_type == "sample":
    loss_dict = self.loss_fn(pred_latents=model_output, target_latents=latents)
else:
    loss_dict = self.loss_fn(noise_pred=model_output, noise=noise)
```

After:
```python
loss_kwargs = dict(dtm_target=dtm_target, timestep=t, global_step=self.global_step)
if cfg.model.prediction_type == "sample":
    loss_dict = self.loss_fn(pred_latents=model_output, target_latents=latents, **loss_kwargs)
else:
    loss_dict = self.loss_fn(noise_pred=model_output, noise=noise, **loss_kwargs)
```

Existing `PixelLoss` and `NoiseLoss` already accept `**kwargs` — they silently ignore the extra arguments. **Zero impact on existing behavior.**

Also update `pixel_loss.py` import to use `gradient_utils.sobel_gradients` (keep `gradient_loss` in pixel_loss.py since it's PixelLoss-specific).

---

## New YAML Config Files (5 experiments)

### `configs/train_bmse.yaml`
BMSE only — tests imbalance correction in isolation.
- `loss.type: "bmse"`, `prediction_type: "sample"`, normalization: `log_global`
- Output: `./experiments/bmse_loss`

### `configs/train_focal.yaml`
Focal regression only — tests dynamic difficulty weighting.
- `loss.type: "focal_regression"`, `prediction_type: "sample"`
- Output: `./experiments/focal_loss`

### `configs/train_directional_gradient.yaml`
Directional gradient only — tests sign-preserving edge awareness.
- `loss.type: "directional_gradient"`, `prediction_type: "sample"`
- Output: `./experiments/directional_gradient`

### `configs/train_composite_basic.yaml`
BMSE + Directional Gradient — combined imbalance + edge experiment.
- `loss.type: "composite"`, 2 components, no curriculum
- Output: `./experiments/composite_basic`

### `configs/train_composite_curriculum.yaml`
Full 3-phase curriculum with all losses:
- Phase 1 (0-1000): BMSE only (safe zero-conv warmup)
- Phase 2 (1000-4000): + Directional Gradient
- Phase 3 (4000+): + Laplacian + Surface Normal
- `anneal_steps: 500` for smooth transitions
- Output: `./experiments/composite_curriculum`

---

## Implementation Order

1. **Foundation**: `gradient_utils.py` + update `pixel_loss.py` import + extend `LossConfig` + trainer kwargs
2. **Imbalance losses**: `bmse_loss.py`, `focal_regression_loss.py`, `dbr_loss.py`
3. **Geometric losses**: `directional_gradient_loss.py`, `laplacian_loss.py`, `surface_normal_loss.py`
4. **Orchestration**: `composite_loss.py` + `__init__.py` imports
5. **Configs**: All 5 YAML experiment files

---

## Verification

1. **Backward compat**: Run `python tools/train.py --config configs/train_pixel_loss.yaml` with `--total_steps 10` — must work identically
2. **Individual loss test**: Run each new loss type standalone: `python tools/train.py --config configs/train_bmse.yaml --total_steps 10`
3. **Composite test**: Run `python tools/train.py --config configs/train_composite_curriculum.yaml --total_steps 10` — check that only BMSE fires at step 0
4. **Check logs**: Verify per-component loss metrics appear in training logs (e.g., `loss_bmse`, `loss_dir_grad`, `weight_bmse`, etc.)
