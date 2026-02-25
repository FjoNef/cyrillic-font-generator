# Major — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **My focus:** AI model design, training, ONNX export, client-side inference strategy.

## Learnings
<!-- Append new entries below -->

### 2026-02-25: Style Character Contract Bug Fix — Training Pipeline

**Issue:** train.py script failed on synthetic data due to style character mismatch.

**Root cause:**
- `src/model/data/dataset.py` line 58: `DEFAULT_STYLE_CHARS` was set to `["A", "B", "H", "O", "g", "n", "o", "p", "s", "x"]` (old list with lowercase)
- `src/model/configs/train_config.yaml` line 12: `style_latin_chars` had the same incorrect list
- **Locked tensor contract** (decisions.md lines 211-212): Requires uppercase-only `["A", "B", "C", "D", "E", "H", "I", "O", "R", "X"]`

**Impact:**
- Training would use wrong Latin reference glyphs
- Generated models would be incompatible with frontend (PR #4)
- Frontend expects the exact 10 uppercase chars: A, B, C, D, E, H, I, O, R, X
- Violation of LOCKED tensor contract established in PR #8 and confirmed by Togusa in PR #4

**Fix:**
- Updated `dataset.py` line 59 to correct uppercase-only list
- Updated `train_config.yaml` line 13 to match
- Added explicit contract comments to both files
- Changes: 2 files, 2 lines, surgical fix

**Verification:**
- Reviewed tensor contract in decisions.md (multiple entries confirm uppercase-only)
- Confirmed frontend implementation in PR #4 uses uppercase Latin extraction
- Model contract explicitly locked — no changes permitted

**Key learning:** When a tensor contract is explicitly LOCKED in team decisions, any deviation in training data breaks the integration. Style character selection must be validated against the contract before training begins.

### 2026-02-25T160138: cGAN Training Pipeline Delivered (Issue #6, PR #8)

**Status:** COMPLETE — Ready for QA review  
**Deliverables:**
- `models/train/model.py` — FontGeneratorGAN (StyleEncoder + UNetGenerator + PatchDiscriminator)
- `models/train/train.py` — Training loop with adversarial + L1 loss
- `models/train/dataset.py` — FontDataset with normalization, character indexing, style glyph extraction
- `models/train/export.py` — ONNX export (opset 17, INT8 quantization, dynamic batch)
- `models/train/requirements.txt` & README.md

**Key decisions finalized:**
1. StyleEncoder: shared-weight CNN + mean-pooling (permutation-invariant style representation)
2. UNetGenerator: blank canvas input, character embedding + style vector at bottleneck, skip connections
3. PatchDiscriminator: 70×70 receptive field for per-patch realism
4. Loss: Adversarial (BCE) + L1 reconstruction (lambda=100)
5. ONNX contract: opset 17, dynamic batch, INT8 weight quantization, float32 activations preserved

**Tensor contract LOCKED:**
- Input: `style_glyphs` [B,10,1,128,128] float32 + `char_index` [B] int64
- Output: `generated_glyph` [B,1,128,128] float32 in [-1,1] range
- Semantic convention: +1.0 = black ink (foreground), -1.0 = white (background)
- Character mapping: 0-32 uppercase А–Я, 33-65 lowercase а–я (66 total Cyrillic chars)
- No changes permitted — frontend (PR #4) already implemented against this contract

**Next actions:**
- Acquire Google Fonts training data (OFL-licensed, Latin+Cyrillic pairs)
- Train for ~200 epochs on GPU (~4-8 hours)
- Export to `models/v1/generator.onnx`
- Batou's `/api/model` endpoint will serve it to Togusa's inference pipeline

### 2026-02-25: ML Engineering Specification

**Model architecture decisions:**
- StyleEncoder uses shared-weight CNN + mean-pool over N reference glyphs.
  Mean-pooling makes the style representation permutation-invariant (order of reference glyphs doesn't matter) and allows variable N at runtime without model changes.
- UNetGenerator injects conditioning at the bottleneck (1×1 spatial) via concatenation, not AdaIN.
  This is simpler to export to ONNX and avoids dynamic shape issues with batch normalisation variants.
- Generator uses a blank canvas input (all zeros) rather than a skeleton template.
  The model must learn to generate from scratch; a skeleton approach would require a separate glyph rendering step in the browser.
- PatchGAN discriminator (70×70 patches) with conditioning on one style glyph.
  Standard pix2pix choice; penalises per-patch realism rather than global image statistics.

**Data pipeline decisions:**
- 10 Latin reference characters chosen for maximum structural diversity: A, B, H, O, g, n, o, p, s, x.
  These cover: enclosed counters (O, o, B), diagonals (A, x), ascenders (H, B), descenders (g, p, y).
- Character index scheme: 0–32 = uppercase А–Я, 33–65 = lowercase а–я. Total 66 chars.
  Ё/ё included (indices 6 and 39).
- Google Fonts GitHub archive fallback for users without an API key. API path is faster.

**ONNX export decisions:**
- Combined StyleEncoder + UNetGenerator into single ONNX graph (FontGeneratorONNX wrapper).
  Single graph simplifies browser loading — one `InferenceSession.create()` call.
- Opset 17 — latest stable at time of writing, well-supported by ONNX Runtime Web.
- Dynamic INT8 weight quantization via `onnxruntime.quantization.quantize_dynamic`.
  Reduces model size by ~50–60% vs fp32. Inference activations remain float32.
- Float16 weight quantization considered but INT8 dynamic has better ONNX Runtime Web support.

**Inference contract decisions:**
- `style_glyphs` shape: `[B, 10, 1, 128, 128]` float32. Fixed N=10 for export simplicity.
- `char_index` shape: `[B]` int64. Simple integer, not one-hot, to avoid large sparse tensors.
- Output: `[B, 1, 128, 128]` float32 in [-1, 1]. Postprocessing is trivial: (x+1)/2*255.
- Recommended ONNX Runtime Web backends: WebGL first, WASM fallback.
- Expected per-glyph time: ~15–30ms WebGL, ~80–150ms WASM 4-thread.
- Inference should run in a Web Worker to avoid UI blocking.

### 2026-02-25: Model output color convention (PR #4 bug fix)

**Model output tensor convention:**
- Output range: [-1, 1] where **+1.0 = black ink (foreground), -1.0 = white background**.
- This is the tanh activation output from the generator; typical for GANs.

**Frontend color mapping bug:**
- **Incorrect formula:** `((output[px] + 1) / 2) * 255`  
  This mapped +1 → 255 (white) and -1 → 0 (black) — **inverted colors**.
- **Correct formula:** `((1 - output[px]) / 2) * 255`  
  This maps +1 → 0 (black ink) and -1 → 255 (white background).

**Affected files (fixed in commit c2adee9):**
- `src/frontend/src/App.tsx` line 67
- `src/frontend/src/inference/OnnxInference.ts` line 90

**Context:** Saito (QA) caught this in PR #4 review as a blocking bug. The issue was visible as inverted glyph rendering — black became white, white became black.
