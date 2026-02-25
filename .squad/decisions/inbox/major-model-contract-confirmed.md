# Model Tensor Contract Confirmation

**Date:** 2026-02-25  
**Author:** Major (AI/ML Engineer)  
**Status:** LOCKED — Do NOT change  
**PR:** #8

## Confirmation

The training pipeline (PR #8) **exactly implements** the tensor contract defined in `.squad/decisions.md` and expected by the frontend (PR #4).

## Contract Specification

### Inputs

1. **`style_glyphs`**: `[B, 10, 1, 128, 128]` float32
   - **Semantics:** 10 Latin reference characters rendered at 128×128 grayscale
   - **Characters:** A, B, C, D, E, H, I, O, R, X (in this order)
   - **Normalization:** [-1, 1] where **+1.0 = black ink (foreground), -1.0 = white (background)**
   - **Preprocessing:** User font → render glyphs → normalize → stack to [10,1,128,128]

2. **`char_index`**: `[B]` int64
   - **Semantics:** Which of the 66 Cyrillic characters to generate (0-indexed)
   - **Mapping:**
     - 0–32: uppercase А–Я (with Ё at index 6)
     - 33–65: lowercase а–я (with ё at index 39)
   - **Total:** 66 characters (33 uppercase + 33 lowercase)

### Output

1. **`generated_glyph`**: `[B, 1, 128, 128]` float32
   - **Semantics:** Generated Cyrillic glyph at 128×128 grayscale
   - **Range:** [-1, 1] where **+1.0 = black ink (foreground), -1.0 = white (background)**
   - **Postprocessing:** Frontend converts to ImageData via `((1 - value) / 2) * 255`

## Implementation Verification

✅ **Model architecture:** `FontGeneratorGAN` accepts `(style_glyphs, char_index)` and returns `generated_glyph`  
✅ **Dataset:** `FontDataset` produces samples with correct shapes and normalization  
✅ **ONNX export:** `export.py` creates ONNX model with input/output names and shapes as specified  
✅ **Quantization:** INT8 quantization applied, preserves float32 activations (no range issues)

## Why This Contract is LOCKED

The frontend (PR #4) has already implemented:
1. **Style glyph extraction:** 10 Latin chars rendered at 128×128, normalized to [-1,1]
2. **Character index mapping:** Russian Cyrillic charset (66 chars) with fixed indices
3. **Output postprocessing:** Converts [-1,1] model output to RGBA ImageData
4. **Web Worker protocol:** Passes `style_glyphs` and `char_index` to ONNX Runtime Web

**Changing this contract would break the frontend.** Any model trained for this project **MUST** conform to these exact shapes and semantics.

## Training Compliance

The training pipeline ensures compliance:
- **Dataset normalization:** `(pixel / 127.5) - 1.0` maps [0,255] → [-1,1]
- **Model output activation:** `tanh()` in final layer outputs [-1,1]
- **Character indexing:** Dataset uses indices 0-65 matching frontend mapping
- **Style glyphs:** Dataset renders the same 10 Latin chars in the same order

## ONNX Export Compliance

The export script (`export.py`) ensures:
- **Input names:** Exactly `style_glyphs` and `char_index`
- **Output name:** Exactly `generated_glyph`
- **Shapes:** Dynamic batch dimension, fixed spatial dimensions [B,10,1,128,128] / [B] / [B,1,128,128]
- **Types:** float32 for images, int64 for indices
- **Opset 17:** Compatible with ONNX Runtime Web 1.15+

## Validation

Before merging any trained model:
1. Check ONNX inputs/outputs with `onnx.load()` and inspect graph
2. Test inference in browser with sample style glyphs
3. Verify output range is [-1,1] (not [0,1] or other)
4. Confirm character indices map correctly (0-32 uppercase, 33-65 lowercase)
5. Validate color convention: +1.0 must render as black, -1.0 as white

## Related Files

- **Frontend contract:** `src/frontend/src/inference/worker/inferenceWorker.ts` lines 89-100
- **Training dataset:** `models/train/dataset.py` (normalization, character mapping)
- **Model architecture:** `models/train/model.py` (FontGeneratorGAN forward method)
- **Export script:** `models/train/export.py` (ONNX input/output definition)

## Summary

✅ **Contract is LOCKED**  
✅ **Training pipeline complies**  
✅ **ONNX export enforces contract**  
✅ **Frontend integration ready**

No further changes to this contract are permitted without explicit team discussion and frontend updates.