"""
train.py — GAN training loop for the Cyrillic font generator.

Usage
-----
    # Train with real fonts from config
    python train/train.py --config configs/train_config.yaml
    
    # Resume from checkpoint
    python train/train.py --config configs/train_config.yaml --resume models/checkpoints/epoch_50.pth
    
    # Train with synthetic data (no fonts required)
    python train/train.py --synthetic --batch_size 16 --num_epochs 50
    
    # Override config parameters
    python train/train.py --config configs/train_config.yaml --batch_size 64 --num_epochs 100

Flags
-----
    --config PATH           Path to training config YAML (optional with --synthetic)
    --resume PATH           Path to checkpoint .pth file to resume training
    --synthetic             Use synthetic random data instead of real fonts
    --batch_size N          Override batch_size from config
    --num_epochs N          Override epochs from config

Losses
------
  Generator loss:
    L_GAN  = BCE(D(G(z)), real)                          — fool the discriminator
    L_L1   = L1(G(z), y) * lambda_l1                    — pixel-level reconstruction
    L_FM   = Σ L1(D_feat(G(z)), D_feat(y)) * lambda_fm  — discriminator feature matching
    L_G    = L_GAN + L_L1 + L_FM

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
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.amp import GradScaler, autocast as _autocast_fn  # PyTorch ≥ 2.0

    def autocast(enabled: bool = True):  # type: ignore[misc]
        """Thin wrapper so call sites use autocast(enabled=...) regardless of torch version."""
        return _autocast_fn("cuda", enabled=enabled)
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast  # type: ignore[assignment]
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import yaml

# Local imports (run from repo root: python src/model/train/train.py)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import CyrillicFontDataset, SyntheticFontDataset
from train.model import StyleEncoder, UNetGenerator, PatchDiscriminator


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
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
    print(f"  [SAVED] Checkpoint saved -> {path}")


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
    print(f"  [OK] Resumed from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    config_path: str | None = None,
    resume: str | None = None,
    use_synthetic: bool = False,
    batch_size_override: int | None = None,
    num_epochs_override: int | None = None,
) -> None:
    # Load config or use defaults for synthetic mode.
    if config_path:
        cfg = load_config(config_path)
    elif use_synthetic:
        # Sensible defaults for synthetic mode (no config required).
        cfg = {
            "data": {
                "fonts_dir": "",  # Not used in synthetic mode.
                "style_latin_chars": ["A", "B", "C", "D", "E", "H", "I", "O", "R", "X"],
                "image_size": 128,
            },
            "model": {
                "style_embedding_dim": 256,
                "char_embedding_dim": 64,
                "unet_base_filters": 32,
                "patch_discriminator_ndf": 64,
            },
            "training": {
                "batch_size": 32,
                "epochs": 50,
                "lr_generator": 0.0002,
                "lr_discriminator": 0.0002,
                "beta1": 0.5,
                "beta2": 0.999,
                "lambda_l1": 10,
                "lambda_fm": 10,
                "checkpoint_interval": 10,
                "log_interval": 100,
            },
            "output": {
                "checkpoint_dir": "../../models/checkpoints/",
            },
        }
    else:
        raise ValueError("Either --config must be provided or --synthetic must be set.")
    
    # Apply CLI overrides to config.
    if batch_size_override is not None:
        cfg["training"]["batch_size"] = batch_size_override
    if num_epochs_override is not None:
        cfg["training"]["epochs"] = num_epochs_override
    
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training device: {device}")
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    CUDA version: {torch.version.cuda}")
        print(f"    cuDNN version: {torch.backends.cudnn.version()}")
        # Lets cuDNN auto-tune the fastest convolution algorithms for fixed input sizes.
        torch.backends.cudnn.benchmark = True
        print("    cuDNN benchmark: enabled")

    # --- Dataset ---
    if use_synthetic:
        print("Using synthetic dataset (random noise tensors)")
        dataset = SyntheticFontDataset(
            num_samples=1000,
            num_style_glyphs=len(data_cfg["style_latin_chars"]),
            image_size=data_cfg["image_size"],
        )
    else:
        dataset = CyrillicFontDataset(
            fonts_dir=data_cfg["fonts_dir"],
            style_chars=data_cfg["style_latin_chars"],
            image_size=data_cfg["image_size"],
        )
    val_size = max(1, int(0.05 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    use_pin_memory = device.type == "cuda"
    num_workers = min(4, os.cpu_count() or 1)
    # batch_size tuning: default is 32. With 8GB VRAM (RTX 3070Ti Laptop) and
    # style_glyphs shape [B, 10, 1, 128, 128] (~20 MB at FP32 for B=32), you can
    # experiment with batch_size 16–64. Increase for higher GPU utilisation;
    # decrease if you hit OOM errors. Override via --batch_size CLI flag.
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0),  # requires num_workers > 0
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
    
    print(f"[+] Models loaded on {device}")

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
    lambda_fm = train_cfg.get("lambda_fm", 10.0)

    # Automatic Mixed Precision scalers — one per optimizer to handle separate
    # backward passes for generator and discriminator independently.
    # GradScaler is a no-op when CUDA is unavailable (CPU training stays FP32).
    use_amp = device.type == "cuda"
    scaler_g = GradScaler(enabled=use_amp)
    scaler_d = GradScaler(enabled=use_amp)
    if use_amp:
        print("[*] AMP (FP16 mixed precision) enabled")

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
        epoch_loss_fm = 0.0
        epoch_start = time.time()

        # Log GPU memory at start of epoch
        if device.type == "cuda":
            torch.cuda.synchronize()
            mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"   GPU memory: {mem_allocated:.2f}GB allocated / {mem_reserved:.2f}GB reserved")

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

            # -------- Train Discriminator --------
            opt_d.zero_grad()
            with autocast(enabled=use_amp):
                # Encode style — reused for both D and G steps.
                style_emb = style_encoder(style_glyphs)   # [B, style_dim]
                fake_glyph = generator(style_emb.detach(), char_index, cond_glyph).detach()

                real_logits = discriminator(target_glyph, cond_glyph)
                fake_logits = discriminator(fake_glyph, cond_glyph)

                loss_d_real = _gan_loss(real_logits, True, criterion_gan)
                loss_d_fake = _gan_loss(fake_logits, False, criterion_gan)
                loss_d = 0.5 * (loss_d_real + loss_d_fake)

            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # -------- Train Generator --------
            opt_g.zero_grad()
            with autocast(enabled=use_amp):
                style_emb = style_encoder(style_glyphs)   # [B, style_dim]
                fake_glyph = generator(style_emb, char_index, cond_glyph)
                fake_logits_g, fake_features = discriminator.forward_with_features(fake_glyph, cond_glyph)
                _, real_features = discriminator.forward_with_features(target_glyph, cond_glyph)

                loss_gan = _gan_loss(fake_logits_g, True, criterion_gan)
                loss_l1  = criterion_l1(fake_glyph, target_glyph) * lambda_l1
                loss_fm  = sum(
                    F.l1_loss(ff, rf.detach())
                    for ff, rf in zip(fake_features, real_features)
                ) * lambda_fm
                loss_g   = loss_gan + loss_l1 + loss_fm

            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            epoch_loss_g  += loss_g.item()
            epoch_loss_d  += loss_d.item()
            epoch_loss_l1 += loss_l1.item()
            epoch_loss_fm += loss_fm.item()

            if batch_idx % log_interval == 0:
                pbar.set_postfix(
                    G=f"{loss_g.item():.4f}",
                    D=f"{loss_d.item():.4f}",
                    L1=f"{loss_l1.item():.4f}",
                    FM=f"{loss_fm.item():.4f}",
                )

        epoch_duration = time.time() - epoch_start
        n_batches = len(train_loader)
        print(
            f"Epoch {epoch:04d} | "
            f"G={epoch_loss_g/n_batches:.4f}  "
            f"D={epoch_loss_d/n_batches:.4f}  "
            f"L1={epoch_loss_l1/n_batches:.4f}  "
            f"FM={epoch_loss_fm/n_batches:.4f}  "
            f"[{epoch_duration:.1f}s]"
        )
        print(f"Epoch {epoch} completed in {epoch_duration:.1f}s")

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
    print(f"\n[DONE] Training complete. Final checkpoint in {checkpoint_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Cyrillic font generator GAN.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to training config YAML (optional with --synthetic).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint .pth file to resume from.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic random data instead of real fonts (no fonts_dir required).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override number of epochs from config.",
    )
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        resume=args.resume,
        use_synthetic=args.synthetic,
        batch_size_override=args.batch_size,
        num_epochs_override=args.num_epochs,
    )
