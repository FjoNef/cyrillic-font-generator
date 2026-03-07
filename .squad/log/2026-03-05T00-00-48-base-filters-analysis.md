# Session Log: Base Filters Analysis

**Timestamp:** 2026-03-05T00:00:48Z  
**Topic:** base_filters 64→32 tradeoff evaluation  
**Agent:** Major

## Executive Summary

Evaluated reducing UNetGenerator `base_filters` from 64 → 32:
- **2.80× parameter reduction** (60.3M → 21.6M total)
- **INT8 + brotli delivery**: ~17-20 MB (≤20 MB target achievable)
- **Training time savings**: ~8-9 hours
- **Quality impact**: Negligible for 128×128 near-binary font glyphs

## Recommendation

**Option B: Stop current training, switch to nf=32, retrain from scratch.**
Saves 8-9 hours; sunk cost at epoch 23 is minimal. No inference contract changes needed.

## Key Constraints

- StyleEncoder fixed at 7.1M params; cannot be reduced
- INT8 quantization bug (onnxruntime opset-18) still pending resolution
- Brotli compression on `/api/model` endpoint must be confirmed by Batou

## Code Changes Required

1. `train.py`: Pass `base_filters=32` to UNetGenerator
2. `export_onnx.py` line 83: `UNetGenerator(..., base_filters=32)`
3. Discard `epoch_0020.pth` (incompatible checkpoint format)
