"""
Dataset loader for Cyrillic font generation training.
Supports Google Fonts with paired Latin+Cyrillic coverage.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# Latin style reference characters (10 chars)
LATIN_CHARS = ['A', 'B', 'C', 'D', 'E', 'H', 'I', 'O', 'R', 'X']

# Russian Cyrillic characters (66 total)
CYRILLIC_CHARS = [
    # Uppercase (0-32)
    'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н',
    'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь',
    'Э', 'Ю', 'Я',
    # Lowercase (33-65)
    'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н',
    'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь',
    'э', 'ю', 'я',
]


class FontDataset(Dataset):
    """
    Dataset that renders glyphs from TTF/OTF font files.
    Each sample: (style_glyphs, target_glyph, char_index)
    """
    
    def __init__(
        self,
        font_paths: List[str],
        image_size: int = 128,
        augment: bool = True,
    ):
        """
        Args:
            font_paths: List of paths to .ttf/.otf font files with Latin+Cyrillic
            image_size: Glyph rasterization size (default 128x128)
            augment: Apply random transformations (rotation, scale, etc.)
        """
        self.font_paths = font_paths
        self.image_size = image_size
        self.augment = augment
        
        # Build dataset: (font_path, char_index) pairs
        self.samples = []
        for font_path in font_paths:
            for char_idx in range(len(CYRILLIC_CHARS)):
                self.samples.append((font_path, char_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        font_path, char_idx = self.samples[idx]
        
        # Load font
        font_size = 96  # Base size
        if self.augment:
            font_size += random.randint(-12, 12)  # Random size variation
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            # Fallback to default font if load fails
            print(f"Warning: Failed to load {font_path}: {e}")
            font = ImageFont.load_default()
        
        # Render Latin style glyphs (10 chars)
        style_glyphs = []
        for char in LATIN_CHARS:
            img = self._render_glyph(char, font)
            if self.augment:
                img = self._augment_glyph(img)
            style_glyphs.append(img)
        style_glyphs = np.stack(style_glyphs, axis=0)  # [10, 128, 128]
        
        # Render target Cyrillic glyph
        target_char = CYRILLIC_CHARS[char_idx]
        target_glyph = self._render_glyph(target_char, font)
        if self.augment:
            target_glyph = self._augment_glyph(target_glyph)
        
        # Convert to tensors: normalize to [-1, 1] where +1=black, -1=white
        style_glyphs = torch.from_numpy(style_glyphs).float()
        style_glyphs = style_glyphs.unsqueeze(1)  # [10, 1, 128, 128]
        style_glyphs = (style_glyphs / 127.5) - 1.0  # [0,255] -> [-1,1]
        
        target_glyph = torch.from_numpy(target_glyph).float()
        target_glyph = target_glyph.unsqueeze(0)  # [1, 128, 128]
        target_glyph = (target_glyph / 127.5) - 1.0
        
        char_index = torch.tensor(char_idx, dtype=torch.int64)
        
        return style_glyphs, target_glyph, char_index
    
    def _render_glyph(self, char: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
        """
        Render a single character to 128x128 grayscale image.
        Returns numpy array [128, 128] with values in [0, 255].
        """
        img = Image.new('L', (self.image_size, self.image_size), color=255)  # White bg
        draw = ImageDraw.Draw(img)
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center the character
        x = (self.image_size - text_width) // 2 - bbox[0]
        y = (self.image_size - text_height) // 2 - bbox[1]
        
        draw.text((x, y), char, font=font, fill=0)  # Black ink
        
        return np.array(img, dtype=np.float32)
    
    def _augment_glyph(self, img: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to a glyph image.
        """
        pil_img = Image.fromarray(img.astype(np.uint8), mode='L')
        
        # Random rotation (-5 to +5 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            pil_img = pil_img.rotate(angle, fillcolor=255)
        
        # Random scale (0.9 to 1.1)
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            new_size = int(self.image_size * scale)
            pil_img = pil_img.resize((new_size, new_size), Image.Resampling.LANCZOS)
            # Crop or pad to original size
            if new_size > self.image_size:
                left = (new_size - self.image_size) // 2
                pil_img = pil_img.crop((left, left, left + self.image_size, left + self.image_size))
            else:
                new_img = Image.new('L', (self.image_size, self.image_size), color=255)
                offset = (self.image_size - new_size) // 2
                new_img.paste(pil_img, (offset, offset))
                pil_img = new_img
        
        return np.array(pil_img, dtype=np.float32)


class SyntheticFontDataset(Dataset):
    """
    Synthetic dataset generator for testing/debugging when no real fonts available.
    Generates simple geometric shapes as glyphs.
    """
    
    def __init__(self, num_samples: int = 1000, image_size: int = 128):
        self.num_samples = num_samples
        self.image_size = image_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random synthetic glyphs
        style_glyphs = torch.randn(10, 1, self.image_size, self.image_size) * 0.5
        target_glyph = torch.randn(1, self.image_size, self.image_size) * 0.5
        char_index = torch.tensor(idx % 66, dtype=torch.int64)
        
        # Clip to [-1, 1]
        style_glyphs = torch.clamp(style_glyphs, -1, 1)
        target_glyph = torch.clamp(target_glyph, -1, 1)
        
        return style_glyphs, target_glyph, char_index


def collect_font_files(data_dir: str) -> List[str]:
    """
    Recursively collect all .ttf and .otf files in a directory.
    """
    data_path = Path(data_dir)
    font_paths = []
    
    for ext in ['*.ttf', '*.otf']:
        font_paths.extend([str(p) for p in data_path.rglob(ext)])
    
    print(f"Found {len(font_paths)} font files in {data_dir}")
    return font_paths


if __name__ == '__main__':
    # Test dataset loading
    print("Testing synthetic dataset...")
    synthetic_ds = SyntheticFontDataset(num_samples=10)
    style, target, idx = synthetic_ds[0]
    print(f"Style glyphs: {style.shape}, Target: {target.shape}, Index: {idx}")
    
    # Test real font dataset if fonts available
    font_dir = '../../data/fonts'
    if os.path.exists(font_dir):
        print(f"\nTesting real font dataset from {font_dir}...")
        font_paths = collect_font_files(font_dir)
        if font_paths:
            font_ds = FontDataset(font_paths[:5], augment=True)
            style, target, idx = font_ds[0]
            print(f"Style glyphs: {style.shape}, Target: {target.shape}, Index: {idx}")
