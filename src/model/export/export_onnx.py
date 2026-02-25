"""
export_onnx.py — Export the trained StyleEncoder + UNetGenerator as a single
                 ONNX graph and validate it with onnxruntime.

Usage
-----
    python export/export_onnx.py \
        --checkpoint models/checkpoints/epoch_0200.pth \
        --output     models/v1/generator.onnx

The exported model can be delivered to the browser as-is, or compressed further
with gzip/brotli (expected: ~12–18 MB after float16 quantization).

ONNX inputs
-----------
  style_glyphs   : float32  [B, 10, 1, 128, 128]
  char_index     : int64    [B]

ONNX output
-----------
  generated_glyph : float32  [B, 1, 128, 128]   values in [-1, 1]

See export/inference_contract.md for full client-side integration details.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Ensure src/model is on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.model import StyleEncoder, UNetGenerator


# ---------------------------------------------------------------------------
# Combined inference wrapper
# ---------------------------------------------------------------------------

class FontGeneratorONNX(torch.nn.Module):
    """
    Wraps StyleEncoder + UNetGenerator into a single forward pass suitable
    for ONNX export.

    ONNX trace inputs:
        style_glyphs : [B, N, 1, H, W]  float32
        char_index   : [B]               int64
    ONNX output:
        generated_glyph : [B, 1, H, W]  float32
    """

    def __init__(self, style_encoder: StyleEncoder, generator: UNetGenerator) -> None:
        super().__init__()
        self.style_encoder = style_encoder
        self.generator = generator

    def forward(
        self,
        style_glyphs: torch.Tensor,   # [B, N, 1, 128, 128]
        char_index: torch.Tensor,     # [B]
    ) -> torch.Tensor:
        style_emb = self.style_encoder(style_glyphs)         # [B, 256]
        return self.generator(style_emb, char_index)          # [B, 1, 128, 128]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(checkpoint_path: str, output_path: str) -> None:
    device = torch.device("cpu")   # Export on CPU for maximum compatibility.

    # --- Load checkpoint ---
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    style_encoder = StyleEncoder(style_dim=256)
    generator = UNetGenerator(style_dim=256, char_emb_dim=64, base_filters=64)

    style_encoder.load_state_dict(ckpt["style_encoder_state"])
    generator.load_state_dict(ckpt["generator_state"])

    style_encoder.eval()
    generator.eval()

    model = FontGeneratorONNX(style_encoder, generator)
    model.eval()

    # --- Dummy inputs for tracing ---
    B, N, H, W = 1, 10, 128, 128
    dummy_style_glyphs = torch.randn(B, N, 1, H, W)
    dummy_char_index   = torch.zeros(B, dtype=torch.int64)

    # --- ONNX export ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_fp32_path = output_path.with_suffix(".fp32.onnx")

    print(f"Exporting to ONNX (fp32): {temp_fp32_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_style_glyphs, dummy_char_index),
            str(temp_fp32_path),
            opset_version=17,
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

    # --- Validate fp32 model ---
    print("Validating fp32 ONNX model…")
    onnx_model = onnx.load(str(temp_fp32_path))
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model is valid.")

    # --- Float16 / dynamic quantization ---
    # Apply dynamic INT8 quantization to weight tensors to reduce file size.
    # This targets the largest linear and conv operators.
    print(f"Applying dynamic INT8 quantization → {output_path}")
    quantize_dynamic(
        str(temp_fp32_path),
        str(output_path),
        weight_type=QuantType.QUInt8,
    )

    # Clean up fp32 intermediate.
    temp_fp32_path.unlink(missing_ok=True)

    # --- Validate quantized model with onnxruntime ---
    print("Validating quantized ONNX model with onnxruntime…")
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    outputs = sess.run(
        None,
        {
            "style_glyphs": dummy_style_glyphs.numpy(),
            "char_index":   dummy_char_index.numpy(),
        },
    )
    glyph_out = outputs[0]
    assert glyph_out.shape == (B, 1, H, W), f"Unexpected output shape: {glyph_out.shape}"
    assert glyph_out.dtype == np.float32, f"Unexpected dtype: {glyph_out.dtype}"
    print(f"  ✓ Output shape: {glyph_out.shape}  dtype: {glyph_out.dtype}")
    print(f"  ✓ Value range: [{glyph_out.min():.3f}, {glyph_out.max():.3f}] (expect [-1, 1])")

    size_mb = output_path.stat().st_size / 1e6
    print(f"\n✅ Exported: {output_path}  ({size_mb:.1f} MB)")
    if size_mb > 20:
        print(f"  ⚠️  Model exceeds 20 MB target ({size_mb:.1f} MB). Consider reducing base_filters.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained model to ONNX.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pth checkpoint file.",
    )
    parser.add_argument(
        "--output",
        default="models/v1/generator.onnx",
        help="Output ONNX file path.",
    )
    args = parser.parse_args()
    export(args.checkpoint, args.output)
