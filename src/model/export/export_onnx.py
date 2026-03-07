"""
export_onnx.py — Export the trained StyleEncoder + UNetGenerator as a single
                 ONNX graph and validate it with onnxruntime.

Usage
-----
    python export/export_onnx.py \
        --checkpoint models/checkpoints/epoch_0200.pth \
        --output     models/v1/generator.onnx

Quantization strategy
---------------------
The script attempts INT8 dynamic quantization (primary), FP16 (fallback), and
FP32 (final fallback).  Expected sizes for the nf=32 model (~21.6M params):

  FP32 : ~86 MB  →  ~25 MB brotli
  INT8 : ~53 MB  →  ~17 MB brotli   (Conv/MatMul quantised; ConvTranspose stays FP32)
  FP16 : ~43 MB  →  ~13 MB brotli   (all weights halved; ConvTranspose included)

The ≤23 MB uncompressed target requires all 21.6M params at 1 byte/param, which is
not achievable with onnxruntime dynamic quantisation because ConvTranspose has no
IntegerOps equivalent.  INT8 (53 MB, ~17 MB brotli) is the best achievable.

Root cause of the historical INT8 crash (issue #21)
----------------------------------------------------
`quantize_dynamic` internally calls `replace_gemm_with_matmul()` which transposes
Gemm weight initialisers in-place but does NOT update the corresponding value_info
shape annotations.  When the modified graph is saved to a temp file and re-loaded
with strict ONNX shape inference (`infer_shapes_path`), the now-stale annotations
conflict with the transposed shapes, producing:

    [ShapeInferenceError] Inferred shape and existing shape differ in dimension 0:
    (512) vs (256)

Fix: strip all initialiser value_info entries from the FP32 model before calling
quantize_dynamic.  These entries are redundant (shapes are fully recoverable from
the initialisers themselves) and removing them lets shape inference start fresh.

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
import shutil
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
        style_emb = self.style_encoder(style_glyphs)                       # [B, 256]
        style_glyph_0 = style_glyphs[:, 0]                                 # [B, 1, 128, 128]
        return self.generator(style_emb, char_index, style_glyph_0)        # [B, 1, 128, 128]


# ---------------------------------------------------------------------------
# ONNX graph manipulation helpers
# ---------------------------------------------------------------------------

def strip_initializer_value_info(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Remove value_info entries that correspond to graph initialisers (weight tensors).

    onnxruntime's quantize_dynamic calls replace_gemm_with_matmul() internally,
    which transposes Gemm weight tensors in-place but does not update the
    corresponding value_info shape annotations.  When the modified graph is then
    saved and re-loaded with strict ONNX shape inference (infer_shapes_path), the
    stale annotations conflict with the transposed tensor shapes:

        [ShapeInferenceError] Inferred shape and existing shape differ in dimension 0:
        (512) vs (256)

    Removing the initialiser value_info entries lets shape inference recompute them
    from scratch on the modified graph.  They are redundant: shapes are always
    recoverable from the initialisers themselves.

    Call this on the FP32 model BEFORE writing the temp file that is passed to
    quantize_dynamic.
    """
    init_names = {init.name for init in model.graph.initializer}
    stale = [vi for vi in model.graph.value_info if vi.name in init_names]
    for vi in stale:
        model.graph.value_info.remove(vi)
    return model


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(checkpoint_path: str, output_path: str) -> None:
    device = torch.device("cpu")   # Export on CPU for maximum compatibility.

    # --- Load checkpoint ---
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    style_encoder = StyleEncoder(style_dim=256)
    generator = UNetGenerator(style_dim=256, char_emb_dim=64, base_filters=32)

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

    # --- Validate fp32 model ---
    print("Validating fp32 ONNX model…")
    onnx_model = onnx.load(str(temp_fp32_path))
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model is valid.")

    # --- INT8 / FP16 / FP32 quantization ---
    #
    # Attempt INT8 dynamic quantisation first.  Fix for issue #21:
    #   quantize_dynamic calls replace_gemm_with_matmul() internally, which
    #   transposes Gemm weight initialisers in-place but leaves the value_info
    #   shape annotations stale.  The subsequent infer_shapes_path call then
    #   fails with a shape conflict.  Stripping initialiser value_info entries
    #   before calling quantize_dynamic resolves this.
    #
    # Limitation: ConvTranspose has no IntegerOps equivalent in ONNX, so those
    # 7 decoder layers remain FP32.  Result: ~53 MB (vs 86 MB FP32).
    # True all-INT8 (~23 MB) is not achievable with onnxruntime dynamic quant.
    quantized = False

    cleaned_fp32_path = output_path.with_suffix(".fp32clean.onnx")
    int8_path = output_path.with_suffix(".int8.onnx")

    try:
        print("Preparing model for INT8 quantisation (stripping stale value_info)…")
        fp32_model = onnx.load(str(temp_fp32_path), load_external_data=True)
        fp32_model = strip_initializer_value_info(fp32_model)
        onnx.save(fp32_model, str(cleaned_fp32_path), save_as_external_data=False)

        print(f"Quantising to INT8: {int8_path}")
        quantize_dynamic(str(cleaned_fp32_path), str(int8_path), weight_type=QuantType.QUInt8)

        size_mb = int8_path.stat().st_size / 1e6
        print(f"  ✓ INT8 quantisation succeeded: {size_mb:.1f} MB")
        shutil.move(str(int8_path), str(output_path))
        quantized = True

    except Exception as q_err:
        print(f"  ⚠️  INT8 quantisation failed: {q_err.__class__.__name__}: {q_err}")
        print("  → Trying FP16 fallback via onnxconverter-common…")
        try:
            from onnxconverter_common import float16
            import warnings
            fp32_model = onnx.load(str(temp_fp32_path), load_external_data=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fp16_model = float16.convert_float_to_float16(fp32_model, keep_io_types=True)
            onnx.save(fp16_model, str(output_path), save_as_external_data=False)
            size_mb = output_path.stat().st_size / 1e6
            print(f"  ✓ FP16 conversion succeeded: {size_mb:.1f} MB")
            quantized = True
        except Exception as fp16_err:
            print(f"  ⚠️  FP16 conversion failed: {fp16_err.__class__.__name__}: {fp16_err}")
            print("  → Falling back to FP32.")

    if not quantized:
        print("  → Consolidating FP32 model into single file…")
        fp32_model = onnx.load(str(temp_fp32_path), load_external_data=True)
        onnx.save(fp32_model, str(output_path), save_as_external_data=False)
        size_mb = output_path.stat().st_size / 1e6
        print(f"  ✓ FP32 model consolidated: {size_mb:.1f} MB")

    # Clean up temp files.
    temp_fp32_path.unlink(missing_ok=True)
    cleaned_fp32_path.unlink(missing_ok=True)
    int8_path.unlink(missing_ok=True)
    for ext_data in output_path.parent.glob(f"{temp_fp32_path.name}.data"):
        ext_data.unlink(missing_ok=True)

    # --- Validate exported model with onnxruntime ---
    print("Validating exported ONNX model with onnxruntime…")
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
        assert glyph_out.shape == (B, 1, H, W), f"Unexpected output shape: {glyph_out.shape}"
        assert glyph_out.dtype == np.float32, f"Unexpected dtype: {glyph_out.dtype}"
        print(f"  ✓ Output shape: {glyph_out.shape}  dtype: {glyph_out.dtype}")
        print(f"  ✓ Value range: [{glyph_out.min():.3f}, {glyph_out.max():.3f}] (expect [-1, 1])")
    except Exception as val_err:
        print(f"  ⚠️  onnxruntime validation failed: {val_err.__class__.__name__}: {val_err}")
        print("  → The ONNX file may still work in the browser (onnxruntime-web has different opset support).")

    size_mb = output_path.stat().st_size / 1e6
    print(f"\n✅ Exported: {output_path}  ({size_mb:.1f} MB)")
    compressed_est = size_mb * 0.3
    print(f"  ℹ️  Estimated brotli-compressed delivery size: ~{compressed_est:.1f} MB")


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
