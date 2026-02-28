#!/usr/bin/env python
"""
DSM2DTM – VAE LoRA Fine-tuning Script.

Converts the key VAE fine-tuning notebooks to a runnable script.

Usage:
    python tools/train_vae.py \
        --base_model runwayml/stable-diffusion-v1-5 \
        --data_dir datasets/new_data_vae \
        --output_dir models/vae_lora_new \
        --lora_rank 4 \
        --epochs 50 \
        --lr 1e-4 \
        --batch_size 8
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model

from diffusers import AutoencoderKL

from src.models.vae_modifier import modify_vae_for_single_channel


class SingleChannelTIFDataset(Dataset):
    """Dataset for VAE training: reads single-channel .tif crops."""

    def __init__(self, root_dir: str):
        import glob
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.tif")))
        if not self.files:
            raise FileNotFoundError(f"No .tif files found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import rasterio
        import numpy as np
        with rasterio.open(self.files[idx]) as ds:
            arr = ds.read(1).astype(np.float32)
        # Normalize to roughly [-1, 1]
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        arr = np.clip(arr, -3, 3) / 3.0
        return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VAE with LoRA for single-channel elevation data")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--data_dir", type=str, required=True, help="Dir with train/ and val/ subdirs of .tif crops")
    parser.add_argument("--output_dir", type=str, default="models/vae_lora_new")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading VAE from {args.base_model}...")
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae")
    modify_vae_for_single_channel(vae, device)

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "conv1", "conv2"],
        lora_dropout=0.0,
    )
    vae = get_peft_model(vae, lora_config)
    vae.print_trainable_parameters()
    vae = vae.to(device)

    # Datasets
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    train_ds = SingleChannelTIFDataset(train_dir)
    val_ds = SingleChannelTIFDataset(val_dir) if os.path.exists(val_dir) else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if val_ds else None

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds) if val_ds else 0} samples")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, vae.parameters()),
        lr=args.lr,
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(args.epochs):
        # Train
        vae.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            batch = batch.to(device)

            posterior = vae.encode(batch).latent_dist
            z = posterior.sample()
            recon = vae.decode(z).sample

            loss = F.mse_loss(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # Validate
        if val_loader:
            vae.eval()
            val_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    posterior = vae.encode(batch).latent_dist
                    z = posterior.sample()
                    recon = vae.decode(z).sample
                    val_total += F.mse_loss(recon, batch).item()
            avg_val_loss = val_total / len(val_loader)
            history["val_loss"].append(avg_val_loss)

            print(f"  Epoch {epoch + 1}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_dir = os.path.join(args.output_dir, "best_model")
                os.makedirs(best_dir, exist_ok=True)
                vae.save_pretrained(best_dir)
                print(f"  → New best model (val_loss={avg_val_loss:.6f})")
        else:
            print(f"  Epoch {epoch + 1}: train_loss={avg_train_loss:.6f}")

        # Checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            vae.save_pretrained(ckpt_dir)
            print(f"  Checkpoint saved: {ckpt_dir}")

    # Save final
    final_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    vae.save_pretrained(final_dir)

    # Save history
    import json
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Models saved to {args.output_dir}")


if __name__ == "__main__":
    main()
