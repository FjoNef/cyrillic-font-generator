"""
build_cache.py — Pre-render all font glyphs and save as .pt cache files.

Each font produces one <font_id>.pt file in the output directory containing
style glyphs and all target Cyrillic glyphs as float32 tensors, eliminating
on-the-fly PIL rendering during training.

Usage
-----
    cd src/model
    python data/build_cache.py --fonts_dir ../../data/fonts --output ../../data/fonts_cache

    # Limit to first N fonts (useful for a quick test)
    python data/build_cache.py --fonts_dir ../../data/fonts --output ../../data/fonts_cache --max_fonts 100

Cache size estimates
--------------------
    Each .pt file: (10 + 66) × 1 × 128 × 128 × 4 bytes ≈ 5.0 MB (float32)
    1974 fonts × 5 MB ≈ 9.9 GB total (float32)
    Use --uint8 to store uint8 → reduces to ≈ 2.5 GB (converted to float32 at load time)

After building, set fonts_cache_dir in configs/train_config.yaml:
    data:
      fonts_cache_dir: "../../data/fonts_cache"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import (
    CyrillicFontDataset,
    DEFAULT_STYLE_CHARS,
    CYRILLIC_CHARS,
    _font_has_coverage,
    _render_glyph,
    _to_tensor,
)


def build_cache(
    fonts_dir: str,
    output_dir: str,
    style_chars: list[str] = DEFAULT_STYLE_CHARS,
    cyrillic_chars: list[str] = CYRILLIC_CHARS,
    image_size: int = 128,
    max_fonts: int | None = None,
    use_uint8: bool = False,
) -> None:
    fonts_path = Path(fonts_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    required_chars = style_chars + cyrillic_chars
    font_files = sorted(
        str(p)
        for p in fonts_path.rglob("*.?tf")
        if _font_has_coverage(str(p), required_chars)
    )

    if max_fonts is not None:
        font_files = font_files[:max_fonts]

    print(f"Found {len(font_files)} eligible fonts → caching to {out_path}")
    skipped = 0
    t0 = time.perf_counter()

    for fp in tqdm(font_files, unit="font"):
        font_name = Path(fp).stem
        cache_file = out_path / f"{font_name}.pt"

        if cache_file.exists():
            skipped += 1
            continue

        try:
            # Render all style glyphs: [N, 1, H, W]
            style_tensors = torch.stack(
                [_to_tensor(_render_glyph(fp, ch, image_size)) for ch in style_chars], dim=0
            )
            # Render all Cyrillic target glyphs: [66, 1, H, W]
            target_tensors = torch.stack(
                [_to_tensor(_render_glyph(fp, ch, image_size)) for ch in cyrillic_chars], dim=0
            )

            payload: dict
            if use_uint8:
                # Convert float32 [-1,1] → uint8 [0,255] for ~4× size reduction.
                # CachedFontDataset will convert back at __getitem__.
                def _to_uint8(t: torch.Tensor) -> torch.Tensor:
                    return ((t + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)

                payload = {
                    "style_glyphs": _to_uint8(style_tensors),
                    "target_glyphs": _to_uint8(target_tensors),
                    "uint8": True,
                }
            else:
                payload = {
                    "style_glyphs": style_tensors,
                    "target_glyphs": target_tensors,
                    "uint8": False,
                }

            torch.save(payload, cache_file)

        except Exception as e:  # noqa: BLE001
            print(f"\n  [WARN] Skipping {fp}: {e}")

    elapsed = time.perf_counter() - t0
    cached_count = len(font_files) - skipped
    print(
        f"\nDone in {elapsed:.1f}s — {cached_count} files written, {skipped} skipped (already existed)."
    )
    total_mb = sum(f.stat().st_size for f in out_path.glob("*.pt")) / (1024 ** 2)
    print(f"Cache size: {total_mb:.1f} MB ({total_mb / 1024:.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build font glyph cache for fast training.")
    parser.add_argument("--fonts_dir", default="../../data/fonts", help="Directory with .ttf/.otf files.")
    parser.add_argument("--output", default="../../data/fonts_cache", help="Output cache directory.")
    parser.add_argument("--max_fonts", type=int, default=None, help="Limit to first N fonts.")
    parser.add_argument("--uint8", action="store_true", help="Store uint8 (4× smaller, slower load).")
    parser.add_argument("--image_size", type=int, default=128)
    args = parser.parse_args()

    build_cache(
        fonts_dir=args.fonts_dir,
        output_dir=args.output,
        image_size=args.image_size,
        max_fonts=args.max_fonts,
        use_uint8=args.uint8,
    )
