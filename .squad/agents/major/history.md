# Major — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **My focus:** AI model design, training, ONNX export, client-side inference strategy.

## Learnings
<!-- Append new entries below -->

### 2026-02-25: Training Data Setup — Google Fonts OFL Download

**Task:** Set up real font training data and verify pipeline readiness for full training run.

**Implementation:**
- Fixed `src/model/data/download_fonts.py` fontTools import case sensitivity (fontTools vs fonttools on Python 3.14)
- Added TTFont.close() call in coverage check to prevent Windows file locking issues during unlink()
- Downloaded 718 OFL-licensed Google Fonts with full Latin+Cyrillic (Russian) coverage
- Source: GitHub archive google/fonts (1.6 GB download, 3750 OFL fonts extracted, 718 passed coverage validation)
- Coverage validation: Each font must contain all 10 style chars (A,B,C,D,E,H,I,O,R,X) + 66 Cyrillic chars (А-Я, а-я with Ё/ё)
- Fixed `train_config.yaml` path resolution: changed `fonts_dir: "../../data/fonts"` to `data/fonts` (must be relative to repo root, not config file)
- Fixed output paths: `models/` instead of `../../models/` for same reason

**Dataset statistics:**
- 718 font files across diverse families (Roboto, Alegreya, AdventPro, etc.)
- 718 fonts × 66 Cyrillic chars = 47,388 total samples
- Train/val split: 45,207 train (95%) / 2,379 val (5%)
- Batch size 32 → 1,413 batches per epoch

**Validation results:**
- 1-epoch validation run confirmed pipeline works end-to-end with real data
- Training on CPU: ~6 seconds per batch
- Losses initialized correctly: D≈0.74, G≈114, L1≈112.5 (typical for untrained GAN)
- No crashes, no data loading errors, tensor shapes correct

**GPU training recommendation:**
- Command: `python src/model/train/train.py --config src/model/configs/train_config.yaml --num_epochs 200`
- Expected duration on GPU: ~4-8 hours for 200 epochs (depends on GPU)
- Expected duration on CPU: ~460 hours (19 days) — GPU required for practical training
- Checkpoint interval: every 10 epochs → 20 checkpoints
- Final model: `models/v1/generator.onnx` (export via `src/model/train/export.py`)

**Key learnings:**
1. **Windows file locking:** TTFont must be explicitly closed before unlinking files, even with lazy=True
2. **Python 3.14 fontTools:** Package installed as `fontTools` (capital T), not `fonttools` lowercase
3. **Path resolution in configs:** When running training from repo root, all config paths must be relative to repo root, not config file location
4. **Dataset scale:** 718 fonts provide good diversity; more is better but not strictly required
5. **Training scale:** Real cGAN training requires GPU for practical timescales; CPU validation sufficient to confirm pipeline correctness

### 2026-02-25: Synthetic Training Mode and CLI Overrides

**Feature:** Added synthetic training mode and CLI parameter overrides to training pipeline.

**Implementation:**
- Created `SyntheticFontDataset` class in `dataset.py` that generates random noise tensors without requiring any font files
- Added three new CLI flags to `train.py`:
  - `--synthetic`: Boolean flag to use synthetic data instead of real fonts
  - `--batch_size N`: Override batch_size from config
  - `--num_epochs N`: Override epochs from config
- Made `--config` optional when `--synthetic` is used (sensible defaults provided)
- CLI overrides applied AFTER config loading, allowing config as base with CLI as final override

**Technical details:**
- `SyntheticFontDataset` generates tensors matching exact contract: `style_glyphs [N, 1, 128, 128]`, `target_glyph [1, 128, 128]`, `char_index` scalar int64
- Random tensors use `torch.randn().clamp(-1, 1)` to match normalized training data range
- Default synthetic dataset size: 1000 samples
- Synthetic mode uses sensible config defaults (batch_size=32, epochs=50, lr=0.0002, etc.)

**Use cases:**
- Quick pipeline validation without downloading font files
- CI/CD testing without large data dependencies
- Development iteration on model architecture
- Debugging training loop issues

**Verification:**
- Tested: `python src/model/train/train.py --synthetic --batch_size 16 --num_epochs 50`
- Confirmed batch size override: 950 samples / 16 = 60 batches (vs default 32 = 30 batches)
- Confirmed synthetic data generates correct tensor shapes and value ranges
- Imports validated, argparse tested, training loop runs successfully

**Key learning:** Synthetic datasets are valuable for rapid iteration and testing without data dependencies. By matching the exact tensor contract (shapes, dtypes, value ranges), synthetic data can validate the entire training pipeline end-to-end even though it won't produce useful models.

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
