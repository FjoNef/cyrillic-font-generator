"""
train.py — GAN training loop for the Cyrillic font generator.

Usage
-----
    python train/train.py --config configs/train_config.yaml
    python train/train.py --config configs/train_config.yaml --resume models/checkpoints/epoch_50.pth

Losses
------
  Generator loss:
    L_GAN  = BCE(D(G(z)), real)          — fool the discriminator
    L_L1   = L1(G(z), y) * lambda_l1    — pixel-level reconstruction
    L_G    = L_GAN + L_L1

  Discriminator loss:
    L_D    = 0.5 * [BCE(D(y), real) + BCE(D(G(z)), fake)]

Checkpoints
-----------
  Saved every `checkpoint_interval` epochs to `models/checkpoints/epoch_N.pth`.
  Each checkpoint contains: generator, discriminator, optimizers, epoch, config.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import yaml

# Local imports (run from repo root: python src/model/train/train.py)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import CyrillicFontDataset
from train.model import StyleEncoder, UNetGenerator, PatchDiscriminator


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _gan_loss(pred: torch.Tensor, target_is_real: bool, criterion: nn.BCEWithLogitsLoss) -> torch.Tensor:
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return criterion(pred, target)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(
    epoch: int,
    style_encoder: StyleEncoder,
    generator: UNetGenerator,
    discriminator: PatchDiscriminator,
    opt_g: torch.optim.Adam,
    opt_d: torch.optim.Adam,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"epoch_{epoch:04d}.pth"
    torch.save(
        {
            "epoch": epoch,
            "style_encoder_state": style_encoder.state_dict(),
            "generator_state": generator.state_dict(),
            "discriminator_state": discriminator.state_dict(),
            "opt_g_state": opt_g.state_dict(),
            "opt_d_state": opt_d.state_dict(),
        },
        path,
    )
    print(f"  💾 Checkpoint saved → {path}")


def load_checkpoint(
    path: str,
    style_encoder: StyleEncoder,
    generator: UNetGenerator,
    discriminator: PatchDiscriminator,
    opt_g: torch.optim.Adam,
    opt_d: torch.optim.Adam,
    device: torch.device,
) -> int:
    ckpt = torch.load(path, map_location=device)
    style_encoder.load_state_dict(ckpt["style_encoder_state"])
    generator.load_state_dict(ckpt["generator_state"])
    discriminator.load_state_dict(ckpt["discriminator_state"])
    opt_g.load_state_dict(ckpt["opt_g_state"])
    opt_d.load_state_dict(ckpt["opt_d_state"])
    print(f"  ✅ Resumed from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config_path: str, resume: str | None = None) -> None:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # --- Dataset ---
    dataset = CyrillicFontDataset(
        fonts_dir=data_cfg["fonts_dir"],
        style_chars=data_cfg["style_latin_chars"],
        image_size=data_cfg["image_size"],
    )
    val_size = max(1, int(0.05 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=device.type == "cuda",
    )
    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val samples")

    # --- Models ---
    style_encoder = StyleEncoder(
        style_dim=model_cfg["style_embedding_dim"]
    ).to(device)
    generator = UNetGenerator(
        style_dim=model_cfg["style_embedding_dim"],
        char_emb_dim=model_cfg["char_embedding_dim"],
        base_filters=model_cfg["unet_base_filters"],
    ).to(device)
    discriminator = PatchDiscriminator(
        ndf=model_cfg["patch_discriminator_ndf"]
    ).to(device)

    # --- Optimisers ---
    opt_g = torch.optim.Adam(
        list(style_encoder.parameters()) + list(generator.parameters()),
        lr=train_cfg["lr_generator"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=train_cfg["lr_discriminator"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
    )

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    lambda_l1 = train_cfg["lambda_l1"]

    # --- Resume ---
    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(
            resume, style_encoder, generator, discriminator, opt_g, opt_d, device
        )

    checkpoint_dir = Path(out_cfg["checkpoint_dir"])
    checkpoint_interval = train_cfg["checkpoint_interval"]
    log_interval = train_cfg["log_interval"]
    total_epochs = train_cfg["epochs"]

    # --- Training loop ---
    for epoch in range(start_epoch + 1, total_epochs + 1):
        style_encoder.train()
        generator.train()
        discriminator.train()

        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        epoch_loss_l1 = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:04d}/{total_epochs}", leave=False)
        for batch_idx, (style_glyphs, target_glyph, char_index) in enumerate(pbar):
            # style_glyphs : [B, N, 1, H, W]
            # target_glyph : [B, 1, H, W]
            # char_index   : [B]  int64
            style_glyphs = style_glyphs.to(device)
            target_glyph = target_glyph.to(device)
            char_index   = char_index.to(device)

            # Pick first style glyph as discriminator conditioning image.
            cond_glyph = style_glyphs[:, 0]   # [B, 1, H, W]

            # -------- Encode style --------
            style_emb = style_encoder(style_glyphs)   # [B, style_dim]

            # -------- Train Discriminator --------
            opt_d.zero_grad()
            fake_glyph = generator(style_emb.detach(), char_index).detach()

            real_logits = discriminator(target_glyph, cond_glyph)
            fake_logits = discriminator(fake_glyph, cond_glyph)

            loss_d_real = _gan_loss(real_logits, True, criterion_gan)
            loss_d_fake = _gan_loss(fake_logits, False, criterion_gan)
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            opt_d.step()

            # -------- Train Generator --------
            opt_g.zero_grad()
            fake_glyph = generator(style_emb, char_index)
            fake_logits_g = discriminator(fake_glyph, cond_glyph)

            loss_gan = _gan_loss(fake_logits_g, True, criterion_gan)
            loss_l1  = criterion_l1(fake_glyph, target_glyph) * lambda_l1
            loss_g   = loss_gan + loss_l1
            loss_g.backward()
            opt_g.step()

            epoch_loss_g  += loss_g.item()
            epoch_loss_d  += loss_d.item()
            epoch_loss_l1 += loss_l1.item()

            if batch_idx % log_interval == 0:
                pbar.set_postfix(
                    G=f"{loss_g.item():.4f}",
                    D=f"{loss_d.item():.4f}",
                    L1=f"{loss_l1.item():.4f}",
                )

        n_batches = len(train_loader)
        print(
            f"Epoch {epoch:04d} | "
            f"G={epoch_loss_g/n_batches:.4f}  "
            f"D={epoch_loss_d/n_batches:.4f}  "
            f"L1={epoch_loss_l1/n_batches:.4f}"
        )

        if epoch % checkpoint_interval == 0:
            save_checkpoint(
                epoch, style_encoder, generator, discriminator,
                opt_g, opt_d, checkpoint_dir
            )

    # Final checkpoint.
    save_checkpoint(
        total_epochs, style_encoder, generator, discriminator,
        opt_g, opt_d, checkpoint_dir
    )
    print(f"\n✅ Training complete. Final checkpoint in {checkpoint_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Cyrillic font generator GAN.")
    parser.add_argument(
        "--config",
        default="configs/train_config.yaml",
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint .pth file to resume from.",
    )
    args = parser.parse_args()
    train(args.config, args.resume)
