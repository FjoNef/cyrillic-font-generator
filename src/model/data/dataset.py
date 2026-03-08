"""
dataset.py — PyTorch Dataset for paired Latin/Cyrillic glyph training data.

Data format
-----------
Each training sample is derived from a single font file that contains both
Latin and Cyrillic glyphs. For every target Cyrillic character we:

  1. Render N=10 Latin reference glyphs from the same font as grayscale
     128×128 PIL images → style_glyphs  (tensor: [N, 1, 128, 128])
  2. Render the target Cyrillic glyph from the same font → target_glyph
     (tensor: [1, 128, 128])
  3. Record the character index (0–65) → char_index  (int64 scalar)

All pixel values are normalised to [-1, 1] (mean 0.5, std 0.5).

Usage
-----
    dataset = CyrillicFontDataset(fonts_dir="data/fonts", config=cfg)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    style_glyphs, target_glyph, char_index = next(iter(loader))
    # style_glyphs : [B, 10, 1, 128, 128]  float32
    # target_glyph : [B, 1, 128, 128]       float32
    # char_index   : [B]                    int64
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from fontTools.ttLib import TTFont
except ImportError as e:
    raise ImportError("fonttools is required. Run: pip install fonttools") from e

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Russian Cyrillic charset — 33 uppercase + 33 lowercase = 66 total characters.
# Index 0 = А (uppercase А), index 32 = Я (uppercase Я)
# Index 33 = а (lowercase а), index 65 = я (lowercase я)
CYRILLIC_UPPERCASE = list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
CYRILLIC_LOWERCASE = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
CYRILLIC_CHARS: List[str] = CYRILLIC_UPPERCASE + CYRILLIC_LOWERCASE
CHAR_TO_INDEX = {ch: idx for idx, ch in enumerate(CYRILLIC_CHARS)}

# Default Latin reference characters (10 chosen for structural diversity).
# These MUST match the tensor contract: A, B, C, D, E, H, I, O, R, X (uppercase only).
DEFAULT_STYLE_CHARS: List[str] = ["A", "B", "C", "D", "E", "H", "I", "O", "R", "X"]

IMAGE_SIZE = 128
NORMALIZE = transforms.Normalize(mean=[0.5], std=[0.5])  # maps [0,1] → [-1,1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _font_has_coverage(font_path: str, chars: List[str]) -> bool:
    """Return True if the font contains glyphs for every character in *chars*."""
    try:
        tt = TTFont(font_path, lazy=True)
        try:
            cmap = tt.getBestCmap()
            if cmap is None:
                return False
            return all(ord(ch) in cmap for ch in chars)
        finally:
            tt.close()
    except Exception:
        return False


def _render_glyph(font_path: str, char: str, size: int = IMAGE_SIZE) -> Image.Image:
    """
    Render a single glyph from *font_path* as a grayscale PIL image of shape
    (size, size).  The glyph is centred and scaled to fill ~80 % of the canvas.

    Returns a white-on-black image (background=0, glyph=255).
    """
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    try:
        pil_font = ImageFont.truetype(font_path, size=int(size * 0.8))
    except OSError:
        pil_font = ImageFont.load_default()

    # Measure and centre the glyph.
    bbox = draw.textbbox((0, 0), char, font=pil_font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - w) // 2 - bbox[0]
    y = (size - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=255, font=pil_font)
    return img


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a grayscale PIL image to a normalised float32 tensor [1, H, W]."""
    arr = np.array(img, dtype=np.float32) / 255.0          # [H, W] in [0, 1]
    tensor = torch.from_numpy(arr).unsqueeze(0)             # [1, H, W]
    return NORMALIZE(tensor)                                # [-1, 1]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CyrillicFontDataset(Dataset):
    """
    Paired (Latin style glyphs, Cyrillic target glyph) dataset.

    Parameters
    ----------
    fonts_dir : str | Path
        Directory containing .ttf / .otf font files.
    style_chars : list[str], optional
        Latin characters to render as style reference.  Defaults to
        DEFAULT_STYLE_CHARS (10 characters).
    image_size : int
        Glyph render resolution in pixels (square).  Default: 128.
    cyrillic_chars : list[str], optional
        Target Cyrillic characters.  Defaults to CYRILLIC_CHARS (66 chars).
    """

    def __init__(
        self,
        fonts_dir: str | Path,
        style_chars: List[str] = DEFAULT_STYLE_CHARS,
        image_size: int = IMAGE_SIZE,
        cyrillic_chars: List[str] = CYRILLIC_CHARS,
    ) -> None:
        self.fonts_dir = Path(fonts_dir)
        self.style_chars = style_chars
        self.image_size = image_size
        self.cyrillic_chars = cyrillic_chars

        required_chars = style_chars + cyrillic_chars
        self.font_paths: List[str] = [
            str(p)
            for p in self.fonts_dir.rglob("*.?tf")   # .ttf and .otf
            if _font_has_coverage(str(p), required_chars)
        ]

        if not self.font_paths:
            raise RuntimeError(
                f"No eligible fonts found in {fonts_dir}. "
                "Run `python src/model/data/download_fonts.py` first."
            )

        # Each (font, cyrillic_char) pair is one training sample.
        self._samples: List[Tuple[str, str]] = [
            (fp, ch) for fp in self.font_paths for ch in self.cyrillic_chars
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        style_glyphs : torch.Tensor  [N, 1, H, W]  float32  values in [-1, 1]
            N Latin reference glyphs from the source font.
        target_glyph : torch.Tensor  [1, H, W]     float32  values in [-1, 1]
            Ground-truth Cyrillic glyph from the same font.
        char_index   : torch.Tensor  scalar         int64
            Index of the target character (0–65).
        """
        font_path, cyrillic_char = self._samples[idx]

        # Render N Latin style reference glyphs.
        style_imgs = [
            _to_tensor(_render_glyph(font_path, ch, self.image_size))
            for ch in self.style_chars
        ]
        style_glyphs = torch.stack(style_imgs, dim=0)  # [N, 1, H, W]

        # Render target Cyrillic glyph.
        target_glyph = _to_tensor(
            _render_glyph(font_path, cyrillic_char, self.image_size)
        )  # [1, H, W]

        char_index = torch.tensor(CHAR_TO_INDEX[cyrillic_char], dtype=torch.int64)

        return style_glyphs, target_glyph, char_index


# ---------------------------------------------------------------------------
# Synthetic Dataset (for testing without real fonts)
# ---------------------------------------------------------------------------

class SyntheticFontDataset(Dataset):
    """
    Synthetic dataset that generates random tensors without requiring any fonts.
    
    Used for quick training tests, pipeline validation, and CI environments
    where font files are not available.
    
    Parameters
    ----------
    num_samples : int
        Total number of synthetic samples.
    num_style_glyphs : int
        Number of Latin reference glyphs (N). Default: 10.
    image_size : int
        Glyph render resolution in pixels (square). Default: 128.
    num_chars : int
        Number of target characters. Default: 66 (Russian Cyrillic).
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_style_glyphs: int = 10,
        image_size: int = IMAGE_SIZE,
        num_chars: int = 66,
    ) -> None:
        self.num_samples = num_samples
        self.num_style_glyphs = num_style_glyphs
        self.image_size = image_size
        self.num_chars = num_chars

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        style_glyphs : torch.Tensor  [N, 1, H, W]  float32  values in [-1, 1]
            Random noise tensors simulating N Latin reference glyphs.
        target_glyph : torch.Tensor  [1, H, W]     float32  values in [-1, 1]
            Random noise tensor simulating a Cyrillic glyph.
        char_index   : torch.Tensor  scalar         int64
            Random character index (0 to num_chars-1).
        """
        # Generate random noise in [-1, 1] to match training data normalization.
        style_glyphs = torch.randn(
            self.num_style_glyphs, 1, self.image_size, self.image_size
        ).clamp(-1, 1)
        
        target_glyph = torch.randn(1, self.image_size, self.image_size).clamp(-1, 1)
        
        char_index = torch.tensor(idx % self.num_chars, dtype=torch.int64)
        
        return style_glyphs, target_glyph, char_index
