# INT8 Quantization Resolution (issue #21)

**Date:** 2026-03-05  
**Author:** Major (AI/ML Engineer)  
**Status:** RESOLVED — INT8 export working

---

## Decision

INT8 dynamic quantization now works via `strip_initializer_value_info()` applied to the FP32 model before calling `quantize_dynamic`.

## Root Cause

`quantize_dynamic` internally calls `replace_gemm_with_matmul()`, which transposes Gemm weight initialisers in-place but does **not** update the corresponding `value_info` shape annotations. The subsequent `infer_shapes_path` strict check then fails:

```
[ShapeInferenceError] Inferred shape and existing shape differ in dimension 0: (512) vs (256)
```

The fix: strip initialiser `value_info` entries (redundant, always recoverable from initialisers) before quantization. This allows the quantizer to recompute shapes fresh after its internal Gemm→MatMul transformation.

This was **not** an opset 18 issue and **not** a `noop_with_empty_axes` issue — both were red herrings. The bug exists in onnxruntime's quantizer regardless of opset version.

## Resulting File Sizes (nf=32 model, 21.6M params)

| Format | Size | ~brotli delivery |
|---|---|---|
| FP32 (old) | 86 MB | ~25 MB |
| **INT8 (new primary)** | **53 MB** | **~16 MB** |
| FP16 (fallback) | 43 MB | ~13 MB |

## Why Not 23 MB?

The uncompressed ≤23 MB target requires all 21.6M params at INT8. However, `ConvTranspose` is absent from onnxruntime's `IntegerOpsRegistry` (no `ConvTransposeInteger` ONNX op exists). The 7 decoder ConvTranspose layers (10.5M params) remain FP32, keeping the INT8 model at 53 MB.

**53 MB INT8 + brotli ≈ 16 MB delivered** — well under the 20 MB delivered target, even if the uncompressed file is larger than hoped.

## Impact on Target

- **Uncompressed target (≤23 MB):** NOT met (53 MB). Requires architecture change or custom static quantization pipeline.  
- **Delivered target (≤20 MB with brotli):** MET (~16 MB estimated).

## Export Pipeline

```
FP32 export (opset 18)
  → strip_initializer_value_info()
  → quantize_dynamic (QUInt8)          ← primary: ~53 MB
  → [fallback] onnxconverter FP16      ← ~43 MB
  → [fallback] FP32 consolidated       ← ~86 MB
```

## Validation (epoch_0020.pth)

- Output shape: (1, 1, 128, 128) ✅
- Output dtype: float32 ✅  
- Value range: [-1.000, 1.000] ✅
- onnxruntime CPU inference: SUCCESS ✅
