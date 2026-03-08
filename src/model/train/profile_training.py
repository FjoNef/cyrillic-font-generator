"""
profile_training.py — Training speed profiling & optimization experiments.

Runs 2 synthetic epochs with detailed per-phase timing to identify bottlenecks,
then experiments with batch size, num_workers, prefetch_factor, and torch.compile.

Usage
-----
    cd src/model
    python train/profile_training.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.amp import GradScaler, autocast as _autocast_fn

    def autocast(enabled: bool = True):
        return _autocast_fn("cuda", enabled=enabled)

except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # type: ignore[assignment]

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import SyntheticFontDataset
from train.model import StyleEncoder, UNetGenerator, PatchDiscriminator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_SAMPLES = 1000     # Synthetic dataset size (matches TRAINING.md benchmark)
IMAGE_SIZE = 128
STYLE_DIM = 256
CHAR_EMB_DIM = 64
BASE_FILTERS = 32
NDF = 64
LR = 0.0002
LAMBDA_L1 = 10.0
LAMBDA_FM = 10.0
NUM_EPOCHS_PROFILE = 2   # Warm-up + measure epoch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync():
    """Synchronize CUDA for accurate wall-time measurement."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_mem_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / (1024 ** 3)
    return 0.0


def _gpu_reserved_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved(0) / (1024 ** 3)
    return 0.0


class PhaseTimer:
    """Accumulates wall-time for named phases across batches."""

    def __init__(self):
        self._acc: dict[str, float] = {}
        self._t0: float | None = None
        self._phase: str | None = None

    def start(self, phase: str):
        _sync()
        self._t0 = time.perf_counter()
        self._phase = phase

    def stop(self):
        _sync()
        dt = time.perf_counter() - self._t0
        self._acc[self._phase] = self._acc.get(self._phase, 0.0) + dt
        self._t0 = None
        self._phase = None

    def report(self) -> dict[str, float]:
        return dict(self._acc)


def _build_models(device: torch.device):
    se = StyleEncoder(style_dim=STYLE_DIM).to(device)
    gen = UNetGenerator(style_dim=STYLE_DIM, char_emb_dim=CHAR_EMB_DIM, base_filters=BASE_FILTERS).to(device)
    disc = PatchDiscriminator(ndf=NDF).to(device)
    return se, gen, disc


def _build_loader(batch_size: int, num_workers: int, prefetch_factor: int | None = None) -> DataLoader:
    ds = SyntheticFontDataset(num_samples=NUM_SAMPLES, num_style_glyphs=10, image_size=IMAGE_SIZE)
    kwargs: dict = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(ds, **kwargs)


def _gan_loss(pred: torch.Tensor, real: bool, crit: nn.BCEWithLogitsLoss) -> torch.Tensor:
    target = torch.ones_like(pred) if real else torch.zeros_like(pred)
    return crit(pred, target)


# ---------------------------------------------------------------------------
# Core: run one epoch and return (epoch_time_s, phase_times_dict)
# ---------------------------------------------------------------------------

def run_epoch(
    device: torch.device,
    loader: DataLoader,
    style_encoder: StyleEncoder,
    generator: UNetGenerator,
    discriminator: PatchDiscriminator,
    opt_g: torch.optim.Adam,
    opt_d: torch.optim.Adam,
    scaler_g: GradScaler,
    scaler_d: GradScaler,
    use_amp: bool,
    timer: PhaseTimer,
) -> float:
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    style_encoder.train()
    generator.train()
    discriminator.train()

    _sync()
    epoch_start = time.perf_counter()

    for batch_idx, (style_glyphs, target_glyph, char_index) in enumerate(loader):
        # --- Data transfer ---
        timer.start("data_transfer")
        style_glyphs = style_glyphs.to(device, non_blocking=True)
        target_glyph = target_glyph.to(device, non_blocking=True)
        char_index = char_index.to(device, non_blocking=True)
        cond_glyph = style_glyphs[:, 0]
        timer.stop()

        # --- D step ---
        timer.start("D_forward")
        opt_d.zero_grad()
        with autocast(enabled=use_amp):
            style_emb = style_encoder(style_glyphs)
            fake_glyph = generator(style_emb.detach(), char_index, cond_glyph).detach()
            real_logits = discriminator(target_glyph, cond_glyph)
            fake_logits = discriminator(fake_glyph, cond_glyph)
            loss_d_real = _gan_loss(real_logits, True, criterion_gan)
            loss_d_fake = _gan_loss(fake_logits, False, criterion_gan)
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
        timer.stop()

        timer.start("D_backward")
        scaler_d.scale(loss_d).backward()
        scaler_d.step(opt_d)
        scaler_d.update()
        timer.stop()

        # --- G step ---
        timer.start("G_forward")
        opt_g.zero_grad()
        with autocast(enabled=use_amp):
            style_emb = style_encoder(style_glyphs)
            fake_glyph = generator(style_emb, char_index, cond_glyph)
            fake_logits_g, fake_features = discriminator.forward_with_features(fake_glyph, cond_glyph)
            _, real_features = discriminator.forward_with_features(target_glyph, cond_glyph)
            loss_gan = _gan_loss(fake_logits_g, True, criterion_gan)
            loss_l1 = criterion_l1(fake_glyph, target_glyph) * LAMBDA_L1
            loss_fm = sum(
                F.l1_loss(ff, rf.detach()) for ff, rf in zip(fake_features, real_features)
            ) * LAMBDA_FM
            loss_g = loss_gan + loss_l1 + loss_fm
        timer.stop()

        timer.start("G_backward")
        scaler_g.scale(loss_g).backward()
        scaler_g.step(opt_g)
        scaler_g.update()
        timer.stop()

    _sync()
    return time.perf_counter() - epoch_start


# ---------------------------------------------------------------------------
# Run 2 epochs with given settings, return avg of epoch 2 (skip warm-up)
# ---------------------------------------------------------------------------

def benchmark(
    label: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None = None,
    use_compile: bool = False,
) -> tuple[float, dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    torch.backends.cudnn.benchmark = True

    loader = _build_loader(batch_size, num_workers, prefetch_factor)
    se, gen, disc = _build_models(device)

    if use_compile:
        try:
            gen = torch.compile(gen, mode="reduce-overhead")
            se = torch.compile(se, mode="reduce-overhead")
            disc = torch.compile(disc, mode="reduce-overhead")
            print(f"  [compile] torch.compile applied (reduce-overhead)")
        except Exception as e:
            print(f"  [compile] FAILED — {e}. Falling back to eager mode.")
            use_compile = False

    opt_g = torch.optim.Adam(
        list(se.parameters()) + list(gen.parameters()), lr=LR, betas=(0.5, 0.999)
    )
    opt_d = torch.optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))
    scaler_g = GradScaler(enabled=use_amp)
    scaler_d = GradScaler(enabled=use_amp)

    timer = PhaseTimer()
    times = []

    for epoch in range(1, NUM_EPOCHS_PROFILE + 1):
        t = run_epoch(device, loader, se, gen, disc, opt_g, opt_d, scaler_g, scaler_d, use_amp, timer)
        times.append(t)
        print(f"  Epoch {epoch}: {t:.2f}s  |  "
              f"VRAM alloc={torch.cuda.memory_allocated(0)/1024**3:.2f}GB  "
              f"reserved={torch.cuda.memory_reserved(0)/1024**3:.2f}GB")

    # AMP health check
    if use_amp:
        print(f"  AMP scaler_g scale: {scaler_g.get_scale():.1f}  "
              f"scaler_d scale: {scaler_d.get_scale():.1f}  "
              f"(should be > 1 if AMP is active)")

    # Phase breakdown (over both epochs combined)
    phase_report = timer.report()
    total_phase = sum(phase_report.values())
    print(f"  Phase breakdown (both epochs combined, {total_phase:.1f}s total):")
    for phase, t_acc in sorted(phase_report.items(), key=lambda x: -x[1]):
        pct = 100 * t_acc / total_phase if total_phase > 0 else 0
        print(f"    {phase:20s} {t_acc:6.2f}s  ({pct:.1f}%)")

    # Use epoch 2 (warm-up already done in epoch 1)
    measured = times[1] if len(times) >= 2 else times[0]
    print(f"  >> Measured epoch time (epoch 2): {measured:.2f}s\n")

    del se, gen, disc, opt_g, opt_d
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return measured, phase_report


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("CYRILLIC FONT GENERATOR — TRAINING SPEED PROFILING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:    {props.name}")
        print(f"VRAM:   {props.total_memory / 1024**3:.1f} GB")
        print(f"SM:     {props.multi_processor_count} SMs")
    print(f"PyTorch {torch.__version__}")
    print(f"CPU cores: {os.cpu_count()}")
    print()

    results: list[tuple[str, float]] = []

    # -----------------------------------------------------------------------
    # BASELINE: batch_size=32, num_workers=4 (current config)
    # -----------------------------------------------------------------------
    print("─" * 70)
    print("BASELINE: batch_size=32, num_workers=4, prefetch=2 (current config)")
    print("─" * 70)
    t, phases = benchmark("baseline", batch_size=32, num_workers=4, prefetch_factor=2)
    results.append(("Baseline (B=32, w=4, pf=2)", t))

    # -----------------------------------------------------------------------
    # Strategy 1: num_workers sweep
    # -----------------------------------------------------------------------
    for nw in [0, 2, 6, 8]:
        print("─" * 70)
        print(f"Strategy 1 — num_workers={nw}, batch_size=32")
        print("─" * 70)
        t, _ = benchmark(f"w{nw}", batch_size=32, num_workers=nw,
                         prefetch_factor=(2 if nw > 0 else None))
        results.append((f"num_workers={nw} (B=32)", t))

    # -----------------------------------------------------------------------
    # Strategy 2: batch_size sweep (keep num_workers at best so far)
    # -----------------------------------------------------------------------
    for bs in [64, 128]:
        print("─" * 70)
        print(f"Strategy 2 — batch_size={bs}, num_workers=4")
        print("─" * 70)
        try:
            t, _ = benchmark(f"bs{bs}", batch_size=bs, num_workers=4, prefetch_factor=2)
            results.append((f"batch_size={bs} (w=4)", t))
        except RuntimeError as e:
            print(f"  OOM at batch_size={bs}: {e}")
            results.append((f"batch_size={bs} (w=4)", float("inf")))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Strategy 3: prefetch_factor=4 vs 2
    # -----------------------------------------------------------------------
    print("─" * 70)
    print("Strategy 3 — prefetch_factor=4, batch_size=32, num_workers=4")
    print("─" * 70)
    t, _ = benchmark("pf4", batch_size=32, num_workers=4, prefetch_factor=4)
    results.append(("prefetch_factor=4 (B=32, w=4)", t))

    # -----------------------------------------------------------------------
    # Strategy 4: torch.compile
    # -----------------------------------------------------------------------
    print("─" * 70)
    print("Strategy 4 — torch.compile(mode='reduce-overhead'), batch_size=32, num_workers=4")
    print("─" * 70)
    t, _ = benchmark("compile", batch_size=32, num_workers=4, prefetch_factor=2, use_compile=True)
    results.append(("torch.compile reduce-overhead (B=32, w=4)", t))

    # -----------------------------------------------------------------------
    # Best combination: pick best num_workers + batch_size combination
    # Find best workers from strategy 1 results
    # -----------------------------------------------------------------------
    worker_results = [(label, t) for label, t in results if label.startswith("num_workers=")]
    best_workers_label, best_workers_t = min(worker_results, key=lambda x: x[1])
    best_nw = int(best_workers_label.split("=")[1].split(" ")[0])

    bs_results = [(label, t) for label, t in results if label.startswith("batch_size=")]
    best_bs = 32  # fallback
    if bs_results:
        valid_bs = [(label, t) for label, t in bs_results if t < float("inf")]
        if valid_bs:
            best_bs_label, _ = min(valid_bs, key=lambda x: x[1])
            best_bs = int(best_bs_label.split("=")[1].split(" ")[0])

    print("─" * 70)
    print(f"BEST COMBO: batch_size={best_bs}, num_workers={best_nw}, prefetch_factor=4")
    print("─" * 70)
    t, _ = benchmark("best_combo", batch_size=best_bs, num_workers=best_nw, prefetch_factor=4)
    results.append((f"Best combo (B={best_bs}, w={best_nw}, pf=4)", t))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    baseline_t = results[0][1]
    print(f"{'Config':<45}  {'Epoch 2 (s)':>11}  {'vs baseline':>12}")
    print("─" * 72)
    for label, t in results:
        if t < float("inf"):
            speedup = baseline_t / t
            flag = " ✓ UNDER 60s" if t < 60 else ""
            print(f"{label:<45}  {t:>11.2f}  {speedup:>10.2f}x{flag}")
        else:
            print(f"{label:<45}  {'OOM':>11}  {'—':>12}")

    best_label, best_t = min(((l, t) for l, t in results if t < float("inf")), key=lambda x: x[1])
    print(f"\nBest config: {best_label}  →  {best_t:.2f}s/epoch")
    if best_t < 60:
        print("✓ 60s TARGET ACHIEVED")
    else:
        print(f"✗ 60s target not reached — closest: {best_t:.2f}s (gap: {best_t - 60:.1f}s)")

    print("\nDone.")
