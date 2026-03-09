"""
create_mini_model.py — Create a miniaturized real-architecture ONNX model for E2E testing.

This creates a model with the EXACT same architecture as the production model but with
4x fewer filters (nf=8 instead of nf=32), resulting in ~1/16 the parameters (~1.3M vs 21.6M).

Key differences from smoke_model.py:
- Real UNet architecture with StyleEncoder + UNetGenerator
- Random weights (not trained) — just needs to produce non-constant output
- Much smaller capacity (nf=8) → ~2-5MB vs 50MB

Expected size: ~5MB FP32 → ~2.5MB INT8 (ConvTranspose stays FP32)

Usage:
    python src/model/export/create_mini_model.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Ensure src/model is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.model import _conv_block, _deconv_block


# ---------------------------------------------------------------------------
# Mini models (same architecture, 4x fewer filters)
# ---------------------------------------------------------------------------

class MiniStyleEncoder(nn.Module):
    """Miniaturized StyleEncoder: nf=8 instead of 64."""

    def __init__(self, in_channels: int = 1, style_dim: int = 64) -> None:
        super().__init__()
        # Shared CNN backbone
        self.encoder = nn.Sequential(
            _conv_block(in_channels, 16,  norm=False),   # [B, 16,  64, 64]
            _conv_block(16,  32),                         # [B, 32,  32, 32]
            _conv_block(32,  64),                         # [B, 64,  16, 16]
            _conv_block(64,  128),                        # [B, 128,  8,  8]
            _conv_block(128, 128),                        # [B, 128,  4,  4]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)               # [B, 128, 1, 1]
        self.fc = nn.Linear(128, style_dim)               # [B, 64]

    def forward(self, style_glyphs: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = style_glyphs.shape
        x = style_glyphs.view(B * N, C, H, W)
        x = self.encoder(x)
        x = self.pool(x).view(B * N, -1)
        x = self.fc(x)
        x = x.view(B, N, -1).mean(dim=1)
        return x


class MiniUNetGenerator(nn.Module):
    """Miniaturized UNetGenerator: nf=8 instead of nf=32 (64)."""

    NUM_CHARS = 66

    def __init__(
        self,
        style_dim: int = 64,
        char_emb_dim: int = 16,
        base_filters: int = 8,
    ) -> None:
        super().__init__()
        nf = base_filters
        self.char_embedding = nn.Embedding(self.NUM_CHARS, char_emb_dim)

        # Project (style_dim + char_emb_dim) → 128 spatial bottleneck (was 512)
        self.cond_proj = nn.Sequential(
            nn.Linear(style_dim + char_emb_dim, 128),
            nn.ReLU(inplace=True),
        )

        # --- Encoder ---
        self.enc1 = _conv_block(1,      nf,       norm=False)  # [nf,   64, 64]
        self.enc2 = _conv_block(nf,     nf * 2)               # [nf*2, 32, 32]
        self.enc3 = _conv_block(nf * 2, nf * 4)               # [nf*4, 16, 16]
        self.enc4 = _conv_block(nf * 4, nf * 8)               # [nf*8,  8,  8]
        self.enc5 = _conv_block(nf * 8, nf * 8)               # [nf*8,  4,  4]
        self.enc6 = _conv_block(nf * 8, nf * 8)               # [nf*8,  2,  2]
        self.enc7 = _conv_block(nf * 8, nf * 8, norm=False)   # [nf*8,  1,  1]

        # --- Decoder ---
        # Bottleneck receives enc7 (nf*8=64) + projected conditioning (128) = 192
        self.dec1 = _deconv_block(nf * 8 + 128, nf * 8, dropout=True)   # [nf*8, 2, 2]
        self.dec2 = _deconv_block(nf * 8 * 2,   nf * 8, dropout=True)   # [nf*8, 4, 4]
        self.dec3 = _deconv_block(nf * 8 * 2,   nf * 8, dropout=True)   # [nf*8, 8, 8]
        self.dec4 = _deconv_block(nf * 8 * 2,   nf * 8)                  # [nf*8,16,16]
        self.dec5 = _deconv_block(nf * 8 + nf * 4,   nf * 4)             # [nf*4,32,32]
        self.dec6 = _deconv_block(nf * 4 + nf * 2,   nf * 2)             # [nf*2,64,64]
        self.dec7 = _deconv_block(nf * 2 + nf,       nf)                 # [nf, 128,128]

        self.final = nn.Sequential(
            nn.Conv2d(nf, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        style_emb: torch.Tensor,
        char_index: torch.Tensor,
        style_glyph_0: torch.Tensor,
    ) -> torch.Tensor:
        B = style_emb.shape[0]

        char_emb = self.char_embedding(char_index)
        cond = torch.cat([style_emb, char_emb], dim=1)
        cond_feat = self.cond_proj(cond)
        cond_spatial = cond_feat.view(B, 128, 1, 1)

        x = style_glyph_0
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        d = torch.cat([e7, cond_spatial], dim=1)
        d = self.dec1(d)
        d = self.dec2(torch.cat([d, e6], dim=1))
        d = self.dec3(torch.cat([d, e5], dim=1))
        d = self.dec4(torch.cat([d, e4], dim=1))
        d = self.dec5(torch.cat([d, e3], dim=1))
        d = self.dec6(torch.cat([d, e2], dim=1))
        d = self.dec7(torch.cat([d, e1], dim=1))

        glyph = self.final(d)
        return glyph


class MiniFontGeneratorONNX(nn.Module):
    """Combined wrapper for ONNX export."""

    def __init__(self, style_encoder: MiniStyleEncoder, generator: MiniUNetGenerator) -> None:
        super().__init__()
        self.style_encoder = style_encoder
        self.generator = generator

    def forward(
        self,
        style_glyphs: torch.Tensor,   # [B, 10, 1, 128, 128]
        char_index: torch.Tensor,     # [B]
    ) -> torch.Tensor:
        style_emb = self.style_encoder(style_glyphs)
        style_glyph_0 = style_glyphs[:, 0]
        return self.generator(style_emb, char_index, style_glyph_0)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def strip_initializer_value_info(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove value_info entries for initializers to avoid quantization conflicts."""
    init_names = {init.name for init in model.graph.initializer}
    stale = [vi for vi in model.graph.value_info if vi.name in init_names]
    for vi in stale:
        model.graph.value_info.remove(vi)
    return model


def create_mini_model(output_path: str = "models/v1/mini_generator.onnx") -> None:
    """Create and export the mini model."""
    import os
    import shutil

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create models with random weights
    style_encoder = MiniStyleEncoder(in_channels=1, style_dim=64)
    generator = MiniUNetGenerator(style_dim=64, char_emb_dim=16, base_filters=8)
    
    style_encoder.eval()
    generator.eval()
    
    model = MiniFontGeneratorONNX(style_encoder, generator)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Dummy inputs
    B, N, H, W = 1, 10, 128, 128
    dummy_style_glyphs = torch.randn(B, N, 1, H, W)
    dummy_char_index = torch.zeros(B, dtype=torch.int64)

    # Export to FP32 first
    temp_fp32_path = output_path.with_suffix(".fp32.onnx")
    print(f"Exporting FP32 model to {temp_fp32_path}")
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_style_glyphs, dummy_char_index),
            str(temp_fp32_path),
            opset_version=18,
            input_names=["style_glyphs", "char_index"],
            output_names=["generated_glyph"],
            dynamic_axes={
                "style_glyphs":    {0: "batch"},
                "char_index":      {0: "batch"},
                "generated_glyph": {0: "batch"},
            },
            do_constant_folding=True,
            export_params=True,
        )

    # Validate FP32 model
    print("Validating FP32 ONNX model...")
    onnx_model = onnx.load(str(temp_fp32_path))
    onnx.checker.check_model(onnx_model)
    fp32_size_mb = temp_fp32_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ FP32 model valid: {fp32_size_mb:.2f} MB")

    # Try FP16 quantization (better than INT8 for small models)
    quantized = False

    try:
        print("Attempting FP16 conversion...")
        from onnxconverter_common import float16
        import warnings
        
        fp32_model = onnx.load(str(temp_fp32_path), load_external_data=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fp16_model = float16.convert_float_to_float16(fp32_model, keep_io_types=True)
        onnx.save(fp16_model, str(output_path), save_as_external_data=False)
        
        fp16_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ FP16 conversion succeeded: {fp16_size_mb:.2f} MB")
        quantized = True

    except Exception as e:
        print(f"  ⚠️  FP16 conversion failed: {e.__class__.__name__}: {e}")
        print("  → Using FP32 model instead")

    if not quantized:
        fp32_model = onnx.load(str(temp_fp32_path), load_external_data=True)
        onnx.save(fp32_model, str(output_path), save_as_external_data=False)
        print(f"  ✓ FP32 model saved: {fp32_size_mb:.2f} MB")

    # Clean up temp files
    temp_fp32_path.unlink(missing_ok=True)

    # Validate with onnxruntime
    print("\nValidating with onnxruntime...")
    import onnxruntime as ort
    import numpy as np

    try:
        sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        outputs = sess.run(
            None,
            {
                "style_glyphs": dummy_style_glyphs.numpy(),
                "char_index":   dummy_char_index.numpy(),
            },
        )
        glyph_out = outputs[0]
        
        print(f"  ✓ Output shape: {glyph_out.shape}")
        print(f"  ✓ Output dtype: {glyph_out.dtype}")
        print(f"  ✓ Value range: [{glyph_out.min():.4f}, {glyph_out.max():.4f}]")
        
        # Check that output is not constant
        std = glyph_out.std()
        print(f"  ✓ Output std dev: {std:.4f}")
        if std < 0.01:
            print("  ⚠️  WARNING: Output appears to be nearly constant!")
        
    except Exception as e:
        print(f"  ⚠️  onnxruntime validation failed: {e.__class__.__name__}: {e}")

    final_size_mb = output_path.stat().st_size / (1024 * 1024)
    final_size_kb = output_path.stat().st_size / 1024
    print(f"\n✅ Mini model created: {output_path}")
    print(f"   Size: {final_size_mb:.2f} MB ({final_size_kb:.1f} KB)")
    print(f"   Parameters: {total_params:,}")
    

if __name__ == "__main__":
    create_mini_model()
