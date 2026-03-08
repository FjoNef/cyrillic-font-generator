# Major — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **My focus:** AI model design, training, ONNX export, client-side inference strategy.

## Core Context

### Prior Investigation & Decisions (Feb 25 – Mar 6)
- ✅ Issue #21: ROOT CAUSE of INT8 quantization failure identified and fixed (strip_initializer_value_info() removes stale value_info entries before quantize_dynamic)
- ✅ base_filters 64→32 tradeoff analyzed (2.80× param reduction: 60.3M→21.6M total; StyleEncoder fixed at 7.1M); decision: retrain from scratch at nf=32
- ✅ cond_proj bottleneck identified (hardcoded 512 output oversized for nf=32 bottleneck 256, minor future micro-opt)
- ✅ ConvTranspose limitation confirmed: no ConvTransposeInteger in ONNX; 7 decoder layers ~10.5M params remain FP32 (unavoidable)
- ✅ Export pipeline hierarchy established: INT8 primary (~16 MB delivered), FP16 fallback (~13 MB), FP32 last resort (~25 MB)
- ✅ PR #8 merged (model training), PR #9 merged (backend integration), frontend ready for model

### Training Progress
- Retrained from scratch at base_filters=32 (target: ≤200 epochs at ~15 min/epoch on GPU)
- Checkpoint: epoch_0200 completed with quantization validation passing

## Learnings

### 2026-03-07: Style-Invariant Output — Root Cause Diagnosis

**Task:** Investigate why the model produces identical output regardless of input font style.

**Findings:**

- **Training data is correct:** Real fonts, `DEFAULT_STYLE_CHARS = ["A","B","C","D","E","H","I","O","R","X"]`, 45,207 samples — style glyphs are genuinely varied.
- **ONNX export is correct:** `style_glyphs` and `char_index` are both dynamic ONNX inputs. Constant folding folds the blank-canvas encoder path but not the style pathway. PR #40 (SharedArrayBuffer fix) was correct.
- **Root cause #1 — Architecture:** `UNetGenerator.forward()` feeds `torch.zeros(B, 1, 128, 128)` through the encoder. This makes all six U-Net skip connections (e1–e6) **deterministic constants** — identical regardless of font style. Style conditioning via `cond_spatial` enters only at the 1×1 bottleneck (a single injection point). After 6 decoder stages each mixed with constant skip connections, the style signal is overwhelmed.
- **Root cause #2 — Training loss:** `lambda_l1=100` dominates. No feature matching loss, no perceptual loss, no style supervision. The model is incentivized to minimize L1 against average glyph shapes, which requires no style sensitivity.
- **Root cause #3 — GAN instability:** Training logs (epochs 11–22) show D loss falling (0.31→0.26) while G loss rises (10.8→11.1), indicating discriminator dominance / precursor to mode collapse. Final loss state (epochs 23–200) is unlogged.
- **Training log gap:** TensorBoard event files in `models/logs/` are all 88 bytes (empty). Logs only cover epochs 11–22 from one training session. The full 200-epoch trajectory is not observable from files on disk.
- **Inference contract discrepancy:** `inference_contract.md` lists wrong style chars (`"g","n","o","p","s","x"` instead of `"C","D","E","I","R","X"`). `decisions.md` confirms frontend code is correct; the contract doc is outdated.

**Fix required:** Retrain.
1. Replace blank canvas encoder input with `style_glyphs[:, 0]` — gives skip connections per-font structure at every scale.
2. Add discriminator feature matching loss.
3. Reduce `lambda_l1` from 100 to 10.
4. Optionally: inject `cond_spatial` at multiple decoder scales (FiLM/AdaIN), not just bottleneck.

**Full diagnosis:** `.squad/decisions/inbox/major-style-conditioning-diagnosis.md`

---

### 2026-03-07T14:31:04Z: ONNX Export SUCCESS — Model Delivered (epoch_0200)

**Task:** Export trained cGAN to ONNX INT8 for browser delivery.

**Deliverable:**
- **File:** models/v1/generator.onnx
- **Size:** 53.1 MB (INT8 quantized)
- **Compressed:** ~15.9 MB brotli (meets ≤20 MB browser delivery target ✅)
- **Output shape:** (1, 1, 128, 128) float32, range [-1.0, 1.0]
- **Validation:** Sanity check passed; CPU inference SUCCESS

**Key Implementation:**
- Applied strip_initializer_value_info() to FP32 model before quantize_dynamic
- INT8 quantization applied to Conv/MatMul; ConvTranspose layers remain FP32
- Result: 53 MB uncompressed, 16 MB delivered (compressed), well under 20 MB target

**Rationale for 53 MB (not 23 MB):**
- quantize_dynamic has no ConvTransposeInteger op in ONNX IntegerOpsRegistry
- 7 decoder ConvTranspose layers (~10.5M params, 42 MB raw) forced to remain FP32
- All-INT8 architecture (~23 MB) would require custom static quantization pipeline or model redesign
- Current INT8+FP32 hybrid meets delivered target and is production-ready

**Implications:**
- ✅ **Togusa:** Frontend inference pipeline now fully functional (model ready for onnxruntime-web)
- ✅ **Batou:** 53.1 MB file needs HTTP brotli compression (~16 MB over wire)
- ✅ **Performance:** INT8 ~1.5–2× faster than FP32 on CPU
- 🔮 **Future:** Custom static quantization could reach ~23 MB if needed


### 2026-03-07: Style-Invariant Output — Diagnosis Complete

**Coordination with Togusa:**

Togusa's debug logging confirmed the JS layer is correct. No inference-time input format issues.

**Diagnosis Result:**

Three compounding causes:
1. **Architecture:** Blank canvas encoder kills skip connections (they're constant across all inputs).
2. **Loss:** L1-dominated training (lambda_l1=100) with no style supervision.
3. **GAN instability:** D dominance precursor suggests mode collapse.

**Action Items:**
1. Implement architectural fix: Use style_glyphs[:, 0] as encoder input.
2. Add feature matching loss.
3. Reduce lambda_l1 to 10.
4. Optionally validate with 10-epoch test on small font set before full retrain.

**Blocking Issue:** Model must be retrained. No inference-time changes can fix trained-out style conditioning.

---

### 2026-03-07: Style-Conditioning Fix Applied

**Task:** Fix the two compounding bugs that caused style-invariant output.

**Changes made:**

1. **`UNetGenerator.forward()` — Bug 1 fixed:** Replaced `torch.zeros(B, 1, 128, 128)` encoder input with `style_glyph_0` (the first style reference glyph, shape `[B, 1, 128, 128]`). The method now takes a third parameter `style_glyph_0`. All 6 U-Net skip connections (e1–e6) now carry real per-font structure at every spatial scale instead of being constant zeros.

2. **`PatchDiscriminator` — Refactored for feature matching:** Split `self.model` (a single `nn.Sequential`) into named layers `layer1`–`layer4` + `final`. Added `forward_with_features()` method returning logits + a list of 4 intermediate feature tensors for use in discriminator feature-matching loss.

3. **Loss rebalancing — Bug 2 fixed:** 
   - `lambda_l1`: 100 → 10 (in `train_config.yaml`, synthetic defaults in `train.py`)
   - `lambda_fm = 10.0` added: feature-matching loss `Σ L1(D_feat(G(z)), D_feat(y))` applied across all 4 discriminator layers, multiplied by `lambda_fm`. Added `lambda_fm` key to `train_config.yaml`.
   - `torch.nn.functional as F` added to `train.py` imports.

4. **`export_onnx.py` — ONNX wrapper updated:** `FontGeneratorONNX.forward()` now extracts `style_glyphs[:, 0]` and passes it as `style_glyph_0` to `generator()`. ONNX input contract unchanged: `style_glyphs [B, 10, 1, 128, 128]` and `char_index [B]` remain the two ONNX-visible inputs.

**Key lesson:** The U-Net skip connections are the primary carrier of spatial structure across decoder scales. Feeding them with constant zeros (blank canvas) made the generator learn structure-independent decoding — style only entered at the 1×1 bottleneck and was diluted across 6 decoder stages before reaching output resolution.

**Status:** Code committed 2026-03-07T21:06:51Z. Requires retraining from scratch (existing epoch_0200 incompatible with new architecture).

---

### 2026-03-07T21:10:19Z: Regression Test Suite — Saito (Tester)

**Cross-Agent Update from Saito**

Wrote 9 regression tests in `src/model/tests/test_style_conditioning.py` to guard against regressions on the two bug fixes:

**Encoder-Not-Zeros (3 tests via PyTorch forward hooks on generator.enc1):**
- Validates that `UNetGenerator.forward()` receives non-zero style glyph input
- Guards against reversion to blank canvas encoder (bug root cause #1)

**Loss Config (2 tests via YAML/AST validation):**
- `lambda_l1` value check (≤ 20)
- `PatchDiscriminator.forward_with_features()` method existence
- Guards against loss misconfiguration (bug root cause #2)

**Additional Coverage (4 tests):**
- Style glyph input shape [B, 10, 1, 128, 128]
- Character index encoding semantics
- Forward pass without errors
- Feature matching tensor shapes

**Result:** All 9 tests pass in 2.3s. Test suite now prevents regression when retraining begins.

---

### 2026-03-08: Training Pipeline GPU Optimizations (Issue #42)

**Task:** Optimize training for NVIDIA RTX 3070Ti Laptop GPU (8GB VRAM).

**Changes made:**

1. **AMP (Automatic Mixed Precision)** — Added `torch.cuda.amp.GradScaler` + `autocast` context manager around all forward + loss computation in both D and G training steps. Used two separate scalers (`scaler_g`, `scaler_d`) — one per optimizer — to handle independent backward passes cleanly. `GradScaler(enabled=use_amp)` is a no-op on CPU so training remains compatible.

2. **cudnn.benchmark = True** — Added immediately after CUDA device detection. Lets cuDNN auto-benchmark and cache the fastest convolution algorithm for our fixed 128×128 spatial dims. Adds ~30 s on first epoch; pays off over 200 epochs.

3. **DataLoader `persistent_workers=True`** — Prevents Python worker processes from being destroyed and respawned at epoch boundaries. Especially valuable on Windows where process creation is expensive. Already had `num_workers=min(4, cpu_count)` and `pin_memory=True`.

4. **Epoch timing** — Added `import time` and `epoch_start = time.time()` at top of each epoch loop; prints `Epoch N completed in X.Xs` after each epoch for before/after benchmarking.

5. **TRAINING.md** — New file at `src/model/TRAINING.md` documenting: how to run training on RTX 3070Ti Laptop, power mode setup (Windows + Linux), `nvidia-smi dmon` monitoring, AMP rationale, batch size VRAM table, expected epoch time benchmarks.

6. **Config comment** — Added VRAM guidance comment to `batch_size` in `train_config.yaml`.

**Key files changed:**
- `src/model/train/train.py` — AMP, cudnn.benchmark, persistent_workers, epoch timing
- `src/model/configs/train_config.yaml` — batch_size comment
- `src/model/TRAINING.md` — new training guide

**All 9 regression tests pass** after changes.

**PR:** #43 (dev → main), closes #42.

---

### 2026-03-07: GPU Training Optimizations (Issue #42) [Orchestrated]

Decision: Two independent GradScalers (one per G/D optimizer) for AMP training. Batch size remains 32; VRAM envelope 16–64 documented for user tuning. Decision merged to decisions.md. PR #43 ready for merge.

---

### 2026-03-07T21:44:55Z: PR #43 Restructure & PR #45 Creation

**By:** Batou (ML engineer)

**Actions:**
- **Closed PR #43** — Wrong branch target (dev→main instead of dev feature) and wrong parent (mixed style-conditioning with training perf)
- **Reverted commit 171c92d from dev** — Training perf changes removed, dev restored to clean state
- **Created squad/42-training-perf branch** — Isolated feature branch for training optimizations
- **Applied Saito's review fixes** — persistent_workers conditional, 6 AMP smoke tests, torch.amp import, TRAINING.md alignment
- **Opened PR #45** — squad/42-training-perf → dev (all training optimizations + all review fixes included)

**Outcome:** PR #45 ready for merge (supersedes PR #43). dev branch clean. Training perf changes properly isolated.


---

### 2026-03-08: Training Speed Profiling & Optimization

**Task:** Profile training loop, target 1 epoch under 60 s on RTX 3070 Laptop GPU.

**Environment:** PyTorch 2.10.0+cu128, Python 3.14.3, Windows 11, RTX 3070 Laptop (8GB VRAM, 40 SMs).

**Key Findings:**

1. **Target already achieved:** The current config (AMP + cudnn.benchmark + w=4) runs the 1000-sample synthetic benchmark in **5.47s** (epoch 2) — 11× under the 60 s target. Previous TRAINING.md estimate of 45–60 s was conservative.

2. **Primary bottleneck: G_backward (58% of compute)** — not data loading. G_backward is the UNet decoder + feature-matching backward pass. No DataLoader tuning can fix this.

3. **cudnn.benchmark warm-up:** Epoch 1 costs 16.71s (11.24s overhead). This is one-time per training run — unavoidable with current fixed spatial dims.

4. **AMP is active:** Scaler values 16384–65536 confirm FP16 is working correctly. No numerical instability.

5. **num_workers=4 is optimal.** 0 is slower (6.24s), 2–4 are equal, 6–8 add overhead.

6. **batch_size:** 32, 64, 128 all give ~5.45–5.48s on synthetic. B=64 marginally best. Updated config.

7. **prefetch_factor=2 better than 4** when compute-bound.

8. **torch.compile: FAILED on Windows** — Triton not installed. Graceful try/except fallback in place. Expected ~20% G_backward speedup on Linux.

9. **Real data bottleneck:** 130k samples/epoch = ~12 min GPU compute. Data loading with w=4 (78ms/batch) is hidden behind GPU compute (175ms/batch). Pipeline is compute-bound.

10. **Cached .pt dataset implemented** (`data/build_cache.py` + `CachedFontDataset`). Per-font .pt files with LRU cache (maxsize=256). Practical speedup marginal for full dataset (already compute-bound), but useful for CPU thermal relief and CI.

**Winning config:** B=64, w=4, pf=2, AMP, cudnn.benchmark — already the default after this task.

**Files changed:** `configs/train_config.yaml`, `data/dataset.py`, `data/build_cache.py`, `train/train.py`, `src/model/TRAINING.md`.

**Decision:** `.squad/decisions.md` (Training Speed Optimization section)

---

### 2026-03-08: torch.compile + num_fonts Configuration (Issue #46)

**Task:** Verify torch.compile works with Triton on Windows and add configurable font count option.

**Environment:** PyTorch 2.10.0+cu128, Python 3.14.3, Windows 11, RTX 3070 Laptop (8GB VRAM).

**Key Findings:**

1. **torch.compile SUCCESS:** Triton now works on Windows (previously failed in issue #42). Smoke test passed — `torch.compile` compiles simple functions on CUDA tensors without errors.

2. **Benchmark results (1000-sample synthetic, B=64, AMP on):**
   - **Baseline (no compile):** 16.91s (epoch 1), 6.25s (epoch 2), 6.18s (epoch 3) → median **6.25s**
   - **torch.compile:** 124.07s (epoch 1), 5.62s (epoch 2), 5.79s (epoch 3) → median **5.79s**
   - **Speedup:** 1.08× (7.9% improvement)
   - **Compilation overhead:** ~108s (first epoch only; amortized over 200 epochs: +0.54s/epoch)

3. **Recommendation:** Keep `use_compile: false` by default. The ~8% speedup is below the 10% threshold, and the 124s first-epoch overhead is significant for short training runs. Enable manually for 200+ epoch runs where the amortized cost is negligible.

4. **num_fonts config option:** Added to `train_config.yaml` (default: null). When set, limits dataset to first N fonts sorted alphabetically. Wired through `CyrillicFontDataset.__init__(num_fonts)` and `CachedFontDataset.__init__(num_fonts)`. Useful for quick experiments and debugging.

5. **Design choice:** `num_fonts` limits the number of **font families** (not glyphs, not style references). This is the most semantically meaningful level for font generalization experiments. Each font still produces 66 samples (one per Cyrillic character).

**Changes:**
- `configs/train_config.yaml` — added `use_compile: false` and `num_fonts` (commented example)
- `train/train.py` — torch.compile integration with graceful fallback; wire `num_fonts` to dataset construction
- `data/dataset.py` — added `num_fonts` parameter to `CyrillicFontDataset` and `CachedFontDataset`
- `TRAINING.md` — updated Strategy 4 (torch.compile) section with new benchmark results
- `train/benchmark_compile.py` — automated benchmark script for compile testing

**Git workflow:** Moved commit `aa89456` (training speed optimization) to feature branch `squad/46-training-triton-fonts`, reverted from `dev`, added new torch.compile + num_fonts work on top.

**PR:** #47 (squad/46-training-triton-fonts → dev), closes #46.

---

### 2026-03-08: PR #47 Merge Conflict Resolution

**Task:** Resolve merge conflicts on PR #47 after `dev` branch revert.

**Context:**
- PR #47 (`squad/46-training-triton-fonts`) added torch.compile support and num_fonts configuration
- `dev` branch HEAD at `1d8ec45` reverted training speed optimizations (commit `aa89456`)
- Revert removed: cached dataset (`CachedFontDataset`), batch size changes, profiling sections
- PR built on top of these reverted features → merge conflicts

**Conflict Resolution:**
1. **Rebased** `squad/46-training-triton-fonts` onto `origin/dev` (git rebase origin/dev)
2. **Resolved conflicts** in 4 files:
   - `train_config.yaml`: Kept `num_fonts` option, removed `fonts_cache_dir`
   - `train.py`: Kept `num_fonts` wiring, removed `CachedFontDataset` branch
   - `dataset.py`: Removed `CachedFontDataset` class and `_load_font_pt` helper
   - `TRAINING.md`: Kept only torch.compile section (lines 206-231), removed all reverted profiling sections
3. **Fixed tests**: Removed `CachedFontDataset` import and skipped the cached dataset test in `test_compile_and_num_fonts.py`
4. **Verified**: All 21 tests pass (2 skipped: compile on Windows + cached dataset test)

**Key Decisions:**
- Preserved torch.compile support (core PR feature) with simplified documentation
- Preserved num_fonts configuration (core PR feature)
- Removed all references to reverted cached dataset feature
- TRAINING.md simplified: removed detailed profiling tables, kept only torch.compile benchmark

**Outcome:**
- Branch rebased: 8 commits ahead of `origin/dev`
- Force-pushed: `git push --force-with-lease origin squad/46-training-triton-fonts`
- PR #47 status: **MERGEABLE** (conflicts resolved)
- Tests passing: 21 passed, 2 skipped

**Learning:** When rebasing after upstream reverts, carefully extract only the intended PR features and remove all dependencies on reverted code. Test coverage must be updated to match the resolved state.

---

### 2026-03-08: Restore Reverted Training Speed Optimization

**Task:** Revert the revert commit `1d8ec45` (which accidentally removed the training speed optimization `aa89456`) back onto `dev`.

**Context:**
- Revert commit `1d8ec45` was applied to `dev` before PR #47 was merged
- PR #47 (`c67a03b`) had already been merged at `dev` HEAD when this restore was requested
- PR #47 conflict resolution had stripped `CachedFontDataset` references from its own files

**Actions:**

1. Ran `git revert 1d8ec45 --no-edit`
2. Resolved 2 conflicts:
   - `TRAINING.md`: Kept full Performance Tuning profiling section (from optimization); updated Strategy 4 with real torch.compile benchmarks from PR #47 (works on Windows now, was "FAILED" in original)
   - `train_config.yaml`: Kept BOTH `num_fonts` comment (PR #47) AND `fonts_cache_dir` comment (optimization)
3. Auto-merged: `dataset.py` (CachedFontDataset restored), `train.py`, profiling scripts, `build_cache.py`
4. Post-revert: Added `num_fonts` parameter to `CachedFontDataset` (was missing; PR #47 was supposed to wire it but stripped the class during conflict resolution)
5. Re-enabled `test_cached_dataset_num_fonts_limit` test (was skipped post-conflict)

**Final state:** 22 tests pass, 1 skipped (compile on CPU). `dev` pushed to `origin/dev` at `d2519bc`.

**Key Lesson:** When a revert is followed by additional PRs that strip the reverted feature's code (conflict resolution), a second revert creates compound conflicts. Must carefully re-add follow-on features (like `num_fonts` in `CachedFontDataset`) that were dropped during the intermediate PR's conflict resolution.

---



**Cross-agent update:** PR #47 merge conflicts resolved via Scribe orchestration.

**Summary:**
- Rebased `squad/46-training-triton-fonts` onto `origin/dev`
- Resolved 4-file conflicts (train_config.yaml, train.py, dataset.py, TRAINING.md)
- Extracted torch.compile + num_fonts features; removed CachedFontDataset dependencies
- All 21 tests pass (2 skipped: compile on Windows + cached dataset)
- Force-pushed; PR now MERGEABLE

**Artifacts:**
- `.squad/orchestration-log/2026-03-08T012713Z-major.md`
- `.squad/log/2026-03-08T012713Z-pr47-conflict-resolution.md`
- `.squad/decisions.md` (appended PR #47 conflict resolution decision)

---

### 2026-03-08: ONNX Re-Export — Post-Retrain (epoch_0200, style-conditioning fixed)

**Task:** Export the fresh epoch_0200 checkpoint to `models/v1/generator.onnx` after the full retrain with fixed style conditioning.

**Checkpoint:** `models/checkpoints/epoch_0200.pth` (latest, 2026-03-08 06:28:51, 291.8 MB)

**Issue encountered:** `torch.compile` was enabled during training (`use_compile: true` in `train_config.yaml`). PyTorch wraps compiled modules in a `OptimizedModule` object; saved state dict keys carry `_orig_mod.` prefix (e.g. `_orig_mod.char_embedding.weight`). The export script called `load_state_dict()` on an unwrapped `UNetGenerator` — key mismatch caused `RuntimeError`.

**Fix applied to `export_onnx.py`:** Added `_strip_orig_mod()` helper that removes the `_orig_mod.` prefix from all checkpoint keys before loading. Applied to both `style_encoder_state` and `generator_state`.

**Export result:**
- Format: INT8 dynamic quantization (primary path succeeded)
- Size: 53.1 MB (same as prior export — model arch unchanged, only weights differ)
- Estimated brotli delivery: ~15.9 MB ✅ (under 20 MB target)
- Validation: onnxruntime CPU forward pass → shape (1, 1, 128, 128), range [-1.0, 1.0] ✅

**Committed:** `ceab05d` on `dev`
- `models/v1/generator.onnx` — new model from retrained checkpoint
- `src/model/export/export_onnx.py` — `_strip_orig_mod()` fix
- `src/model/configs/train_config.yaml` — un-commented `num_fonts: 180` and `fonts_cache_dir` (reflects actual retrain config)

**Decision recorded:** `.squad/decisions/inbox/major-onnx-export.md`

**Key rule:** Whenever `use_compile: true` in `train_config.yaml`, checkpoints will have `_orig_mod.` prefixed keys. The export script now handles this transparently.
