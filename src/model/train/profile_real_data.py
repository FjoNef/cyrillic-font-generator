"""
profile_real_data.py — Profile real font data loading and rendering speed.

Measures how long per-epoch GPU compute vs data loading takes with real fonts.
Uses a subset of fonts to keep the run short.

Usage
-----
    cd src/model
    python train/profile_real_data.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import CyrillicFontDataset, CYRILLIC_CHARS

FONTS_DIR = str(Path(__file__).resolve().parents[3] / "data" / "fonts")
BATCH_SIZE = 32
NUM_WORKERS_LIST = [0, 4]
FONT_SUBSET_SIZE = 10   # Use 10 fonts → 10 × 66 = 660 samples
PREFETCH = 2

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _render_and_load(num_workers: int, prefetch: int | None = None):
    """Measure DataLoader throughput for real font data."""
    dataset = CyrillicFontDataset(
        fonts_dir=FONTS_DIR,
        style_chars=["A", "B", "C", "D", "E", "H", "I", "O", "R", "X"],
        image_size=128,
    )
    # Use first N fonts' worth of samples
    samples_per_font = len(CYRILLIC_CHARS)  # 66
    subset_indices = list(range(min(FONT_SUBSET_SIZE * samples_per_font, len(dataset))))
    subset = Subset(dataset, subset_indices)

    kwargs: dict = dict(
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available()),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0 and prefetch is not None:
        kwargs["prefetch_factor"] = prefetch

    loader = DataLoader(subset, **kwargs)

    num_batches = len(loader)
    total_samples = len(subset)

    print(f"  Dataset: {total_samples} samples ({FONT_SUBSET_SIZE} fonts × {samples_per_font} chars), "
          f"{num_batches} batches at B={BATCH_SIZE}")

    # Warm-up iteration
    for _ in loader:
        break

    # Timed iteration (data loading only — no GPU work)
    _sync()
    t0 = time.perf_counter()
    for i, (style_glyphs, target_glyph, char_index) in enumerate(loader):
        if torch.cuda.is_available():
            style_glyphs = style_glyphs.to("cuda", non_blocking=True)
            target_glyph = target_glyph.to("cuda", non_blocking=True)
            _ = char_index.to("cuda", non_blocking=True)
    _sync()
    elapsed = time.perf_counter() - t0

    ms_per_batch = 1000 * elapsed / num_batches
    ms_per_sample = 1000 * elapsed / total_samples
    extrapolated_full_epoch = elapsed * (1974 * samples_per_font / total_samples)

    print(f"  Elapsed: {elapsed:.2f}s  |  {ms_per_batch:.1f}ms/batch  |  {ms_per_sample:.2f}ms/sample")
    print(f"  Extrapolated full-dataset epoch (1974 fonts × 66 chars = 130,284 samples):")
    print(f"    Data-loading-only: {extrapolated_full_epoch:.0f}s ({extrapolated_full_epoch/60:.1f} min)")
    return elapsed, ms_per_batch


if __name__ == "__main__":
    print("=" * 70)
    print("REAL DATA LOADING PROFILE")
    print("=" * 70)
    print(f"Font subset: {FONT_SUBSET_SIZE} fonts")
    print()

    for nw in NUM_WORKERS_LIST:
        print(f"num_workers={nw}, prefetch_factor={PREFETCH if nw > 0 else 'n/a'}")
        t, ms = _render_and_load(nw, PREFETCH if nw > 0 else None)
        print()

    print("Note: On-the-fly rendering (PIL ImageFont.truetype) per sample renders")
    print(f"  10 style glyphs + 1 target glyph = 11 ImageFont calls per sample.")
    print("  For 130,284 samples that is ~1.43M font render calls per epoch.")
    print("  A cached .pt dataset would eliminate this entirely.")
    print()
    print("Done.")
