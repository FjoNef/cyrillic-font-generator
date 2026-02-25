# Saito PR #8 Verdict — CHANGES REQUESTED

**PR:** #8 feat/major-model-training → dev  
**Author:** Major  
**Reviewer:** Saito (QA)  
**Date:** 2026-02-25  
**Status:** ⚠️ CHANGES REQUESTED

## Blocking Issue

### Style Character Mismatch in Documentation

**Problem:** Conflicting style character specifications in `.squad/decisions.md`:

- **Line 184** (Major's ML engineering decisions):  
  `A, B, H, O, g, n, o, p, s, x` — mixed case, lowercase letters

- **Line 196** (Togusa's Frontend scaffold decisions):  
  `A,B,C,D,E,H,I,O,R,X` — uppercase only

**Current state:**
- Training code (`models/train/dataset.py:18`): `['A', 'B', 'C', 'D', 'E', 'H', 'I', 'O', 'R', 'X']` ✅
- Frontend code (`src/frontend/src/font/FontLoader.ts:5`): `['A', 'B', 'C', 'D', 'E', 'H', 'I', 'O', 'R', 'X']` ✅
- Documentation line 184: `A, B, H, O, g, n, o, p, s, x` ❌

**Impact:** Training code and frontend are aligned, but documentation is outdated and misleading.

**Required fix:**
1. Update `.squad/decisions.md` line 184 to correct contract: `A, B, C, D, E, H, I, O, R, X` (uppercase only)
2. Add comment in `dataset.py` explaining why uppercase-only (e.g., "Uses uppercase Latin letters for visual consistency and structural diversity")

---

## Non-Blocking Observations

### ✅ Acceptance Criteria — All Met

From issue #6:

- [x] Style encoder accepts 10 Latin glyphs at 128×128 grayscale float32
- [x] char_index input is int64 (66 values: 0–65)
- [x] Output is [B,1,128,128] float32 in [-1.0, 1.0] where +1.0 = black ink
- [x] ONNX export produces correct input/output names and shapes (`style_glyphs`, `char_index`, `generated_glyph`)
- [x] Tensor contract matches `inferenceWorker.ts` expectations

### ✅ Model Architecture

- StyleEncoder: Shared CNN + mean-pooling for permutation invariance — **CORRECT**
- UNetGenerator: Character embedding + style injection at bottleneck + skip connections — **CORRECT**
- PatchDiscriminator: 70×70 PatchGAN conditioned on style glyph — **CORRECT**
- Loss: Adversarial + L1 (lambda=100.0) — **STANDARD pix2pix approach**

### ✅ ONNX Export

- Opset 17 — **CORRECT**
- INT8 dynamic quantization — **GOOD for browser**
- Input/output names match frontend contract — **VERIFIED**
- Targets `models/v1/generator.onnx` — **CORRECT path**

### ✅ Training Pipeline

- Adam optimizers (lr=0.0002, betas=(0.5, 0.999)) — **Standard GAN training**
- Checkpoint every 10 epochs — **REASONABLE**
- Sample outputs every 5 epochs — **GOOD for monitoring**
- TensorBoard logging — **GOOD**

### ✅ Dataset

- Supports Google Fonts TTF/OTF files — **CORRECT**
- Synthetic fallback for testing — **GOOD**
- Augmentation (rotation ±5°, scale 0.9-1.1×) — **REASONABLE**
- Normalization to [-1, 1] with +1=black — **MATCHES model convention**

### ✅ Documentation

- `models/train/README.md` — comprehensive, clear instructions
- `models/v1/README.md` — explains model placement and contract
- Both READMEs reference the tensor contract correctly

---

## Summary

**Verdict:** CHANGES REQUESTED

The training pipeline implementation is **technically sound** and ready for training. All acceptance criteria are met. However, there is a **documentation inconsistency** that must be resolved before merge to prevent future confusion.

Once the style character documentation is corrected in `.squad/decisions.md` line 184, this PR is ready to merge.

---

**Next steps:**
1. Major: Update `.squad/decisions.md` line 184
2. Major: Add explanatory comment to `dataset.py`
3. Saito: Re-review and approve
4. Merge to dev
