"""
Benchmark torch.compile impact on training speed.

Runs 3 epochs of training with synthetic data (1000 samples) twice:
1. Without torch.compile (baseline)
2. With torch.compile enabled

Reports median epoch time for each configuration.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add src/model to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.amp import GradScaler as _GradScaler_fn, autocast as _autocast_fn
    def autocast(enabled: bool = True):
        return _autocast_fn("cuda", enabled=enabled)
    def GradScaler(enabled: bool = True):
        return _GradScaler_fn("cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from data.dataset import SyntheticFontDataset
from train.model import StyleEncoder, UNetGenerator, PatchDiscriminator


def _gan_loss(pred: torch.Tensor, target_is_real: bool, criterion: nn.BCEWithLogitsLoss) -> torch.Tensor:
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return criterion(pred, target)


def benchmark_training(use_compile: bool = False, num_epochs: int = 3) -> list[float]:
    """Run training for num_epochs and return list of epoch times."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    dataset = SyntheticFontDataset(num_samples=1000, num_style_glyphs=10, image_size=128)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    
    # Models
    style_encoder = StyleEncoder(style_dim=256).to(device)
    generator = UNetGenerator(style_dim=256, char_emb_dim=64, base_filters=32).to(device)
    discriminator = PatchDiscriminator(ndf=64).to(device)
    
    if use_compile:
        generator = torch.compile(generator)
        discriminator = torch.compile(discriminator)
    
    # Optimizers
    opt_g = torch.optim.Adam(
        list(style_encoder.parameters()) + list(generator.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    
    use_amp = device.type == "cuda"
    scaler_g = GradScaler(enabled=use_amp)
    scaler_d = GradScaler(enabled=use_amp)
    
    torch.backends.cudnn.benchmark = True
    
    epoch_times = []
    
    for epoch in range(1, num_epochs + 1):
        style_encoder.train()
        generator.train()
        discriminator.train()
        
        epoch_start = time.time()
        
        for style_glyphs, target_glyph, char_index in loader:
            style_glyphs = style_glyphs.to(device)
            target_glyph = target_glyph.to(device)
            char_index = char_index.to(device)
            cond_glyph = style_glyphs[:, 0]
            
            # Train Discriminator
            opt_d.zero_grad()
            with autocast(enabled=use_amp):
                style_emb = style_encoder(style_glyphs)
                fake_glyph = generator(style_emb.detach(), char_index, cond_glyph).detach()
                real_logits = discriminator(target_glyph, cond_glyph)
                fake_logits = discriminator(fake_glyph, cond_glyph)
                loss_d = 0.5 * (_gan_loss(real_logits, True, criterion_gan) + 
                               _gan_loss(fake_logits, False, criterion_gan))
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()
            
            # Train Generator
            opt_g.zero_grad()
            with autocast(enabled=use_amp):
                style_emb = style_encoder(style_glyphs)
                fake_glyph = generator(style_emb, char_index, cond_glyph)
                fake_logits_g, fake_features = discriminator.forward_with_features(fake_glyph, cond_glyph)
                _, real_features = discriminator.forward_with_features(target_glyph, cond_glyph)
                
                loss_gan = _gan_loss(fake_logits_g, True, criterion_gan)
                loss_l1 = criterion_l1(fake_glyph, target_glyph) * 10
                loss_fm = sum(
                    torch.nn.functional.l1_loss(ff, rf.detach())
                    for ff, rf in zip(fake_features, real_features)
                ) * 10
                loss_g = loss_gan + loss_l1 + loss_fm
            
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f"  Epoch {epoch}/{num_epochs}: {epoch_time:.2f}s")
    
    return epoch_times


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Benchmark requires GPU.")
        return
    
    print("=" * 60)
    print("torch.compile Benchmark")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: 1000 synthetic samples, batch_size=64")
    print(f"Config: base_filters=32, AMP enabled, cudnn.benchmark=True")
    print()
    
    # Baseline: without torch.compile
    print("[1/2] Baseline (no torch.compile)")
    print("-" * 60)
    baseline_times = benchmark_training(use_compile=False, num_epochs=3)
    baseline_median = sorted(baseline_times)[1]  # median of 3
    print(f"✓ Baseline median: {baseline_median:.2f}s/epoch")
    print()
    
    # With torch.compile
    print("[2/2] With torch.compile")
    print("-" * 60)
    compile_times = benchmark_training(use_compile=True, num_epochs=3)
    compile_median = sorted(compile_times)[1]  # median of 3
    print(f"✓ Compiled median: {compile_median:.2f}s/epoch")
    print()
    
    # Summary
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Baseline:      {baseline_median:.2f}s/epoch")
    print(f"torch.compile: {compile_median:.2f}s/epoch")
    speedup = baseline_median / compile_median
    if speedup > 1.0:
        print(f"Speedup:       {speedup:.2f}× faster")
        pct = (speedup - 1.0) * 100
        print(f"               ({pct:.1f}% improvement)")
    else:
        slowdown = compile_median / baseline_median
        print(f"Slowdown:      {slowdown:.2f}× slower")
        pct = (slowdown - 1.0) * 100
        print(f"               ({pct:.1f}% regression)")
    print()
    
    # Recommendation
    if speedup >= 1.10:
        print("✓ Recommendation: Enable use_compile in train_config.yaml (>10% speedup)")
    elif speedup >= 1.0:
        print("⚠ Recommendation: Marginal benefit (<10% speedup); keep disabled by default")
    else:
        print("✗ Recommendation: Keep torch.compile disabled (slower than baseline)")


if __name__ == "__main__":
    main()
