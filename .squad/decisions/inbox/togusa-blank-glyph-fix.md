# Blank Cyrillic Glyph — Frontend Audit & Smoke Test Fix

**Author:** Togusa (Frontend Dev)  
**Date:** 2026-03-09  
**Status:** Frontend fix applied — retraining required for full resolution  

---

## Root Cause Confirmed

The blank glyph bug is **model-level**, not frontend-level.

The `epoch_0200` ONNX model was trained with the **old, broken `UNetGenerator` architecture** that fed `torch.zeros(B,1,128,128)` through the U-Net encoder. All six skip connections were constant-zero regardless of input font. The model learned to minimise L1 loss by predicting near-all-background (-1.0), which postprocesses to 255 (white) on every pixel → blank canvas.

Cross-referenced with Major's inbox finding (`major-blank-glyph-finding.md`): non-blank check expected to fail with < 1 % of pixels above 0.0 on `epoch_0200`.

---

## Frontend Code Audit — Everything is Correct

A full audit of the inference pipeline found **no frontend code bugs**:

| Component | Formula / Logic | Status |
|-----------|----------------|--------|
| `FontLoader.extractStyleGlyphs()` | `1 - brightness * 2` (white bg→-1, black ink→+1) | ✅ Correct |
| `OnnxInference.generateGlyph()` | `((1 - outputData[i]) / 2) * 255` (+1→0 black, -1→255 white) | ✅ Correct |
| `App.tsx` postprocessing | Same formula, `alpha = 255` hardcoded | ✅ Correct |
| `inferenceWorker.ts` | `new Float32Array(outputData)` copy before return | ✅ Correct |
| `ModelLoader.ts` | `new Float32Array(msg.output)` defensive copy on receive | ✅ Correct |

No code changed between the model export (`ceab05d`) and the blank-output report (`git diff` was empty on all frontend inference files).

---

## Why the Smoke Test Gave a False Pass

The style conditioning smoke test checked **relative** style response (MAD between `fill(+1.0)` and `fill(-1.0)` inputs). MAD = 0.281 confirmed the model responds differently to different style extremes.

However, the model can produce slightly different near-all-background outputs and still pass a MAD > 0.01 threshold. Neither output needs to be non-blank for the relative test to pass.

**The test lacked an absolute pixel content assertion.**

---

## Fix Applied (Frontend)

Updated `src/frontend/e2e/style-conditioning-real.spec.ts`:

1. **Style conditioning test** — added `minA`, `maxA`, `minB`, `maxB` to returned data and added:
   ```javascript
   expect(result.maxA).toBeGreaterThan(-0.5); // not all-background
   expect(result.maxB).toBeGreaterThan(-0.5);
   ```

2. **New test**: `'non-blank: neutral style input must produce visible glyph pixels'`
   - Uses `fill(0.0)` (neutral midtone) style input and `char_index=0` (А)
   - Asserts `max > -0.5` (ink pixels present) and `std > 0.05` (structural variation)
   - This test **will fail** on the `epoch_0200` model and **pass** on the correctly retrained model

Total tests: was 20, now 21 (Chromium E2E). Unit tests unchanged (108/108 green).

---

## Action Required: Model Retraining

**For Major:** The architectural fix (`UNetGenerator.forward()` now passes `style_glyph_0` instead of `torch.zeros`) is already committed. The next step is to:

1. Retrain from scratch using the fixed architecture
2. Export new ONNX model
3. Run `python export/check_model.py models/v1/generator.onnx` to verify the **non-blank check passes**
4. Re-run `npx playwright test style-conditioning-real.spec.ts` to verify all 5 real-model tests pass (including the new non-blank test)

---

## Decision

**The smoke test must always include an absolute-value non-blank assertion alongside relative MAD.**  
A MAD-only test is insufficient to detect mode-collapsed models that output all-background.
