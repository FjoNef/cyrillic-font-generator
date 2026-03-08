# Major — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **My focus:** AI model design, training, ONNX export, client-side inference strategy.

## Core Context

### Prior Investigation & Decisions (Feb 25 – Mar 6)
- ✅ Issue #21: ROOT CAUSE of INT8 quantization failure identified and fixed (`strip_initializer_value_info()` removes stale value_info entries before `quantize_dynamic`)
- ✅ base_filters 64→32 tradeoff analyzed (2.80× param reduction: 60.3M→21.6M total; StyleEncoder fixed at 7.1M); decision: retrain from scratch at nf=32
- ✅ ConvTranspose limitation confirmed: no ConvTransposeInteger in ONNX; 7 decoder layers ~10.5M params remain FP32 (unavoidable)
- ✅ Export pipeline hierarchy: INT8 primary (~16 MB delivered), FP16 fallback (~13 MB), FP32 last resort (~25 MB)
- ✅ PR #8 merged (model training), PR #9 merged (backend integration), frontend ready for model

### Style-Conditioning Bug Diagnosis & Fix (Mar 7)
Root cause of style-invariant output was three compounding issues: (1) `UNetGenerator.forward()` fed `torch.zeros` through the encoder, making all 6 U-Net skip connections (e1–e6) constant regardless of font input — style only entered at the 1×1 bottleneck and was overwhelmed; (2) `lambda_l1=100` dominated training with no style supervision; (3) GAN instability (D loss falling while G rises by epoch 22). Fix: replaced blank encoder input with `style_glyphs[:, 0]`, refactored `PatchDiscriminator` into named layers with `forward_with_features()`, set `lambda_l1=10` and `lambda_fm=10`. ONNX export wrapper updated accordingly. Saito wrote 9 regression tests (`src/model/tests/test_style_conditioning.py`) covering encoder-not-zeros, loss config, and feature matching — all passing in 2.3s. First export of epoch_0200 (`models/v1/generator.onnx`, 53.1 MB INT8, ~15.9 MB brotli) was done before the style-conditioning fix was applied, resulting in blank glyph output.

### Training Pipeline Optimizations (Mar 7–8, Issues #42, #46)
AMP (`GradScaler` × 2, one per G/D optimizer), `cudnn.benchmark=True`, `persistent_workers=True`, and epoch timing added to `train.py`. Profiling on RTX 3070 Laptop (8 GB VRAM) confirmed primary bottleneck is G_backward (58% of compute), not data loading; 1000-sample synthetic benchmark runs in 5.47 s (11× under 60 s target). `torch.compile` now works on Windows (previously failed): ~8% speedup (1.08×), 124 s first-epoch overhead — default `use_compile: false`, enable manually for 200+ epoch runs. `num_fonts` config option added to `train_config.yaml` (limits dataset to first N font families). `CachedFontDataset` with per-font `.pt` files and LRU cache implemented in `data/build_cache.py`. PR #43 was closed (wrong branch target), superseded by PR #45; PR #47 (`squad/46-training-triton-fonts`) required rebase after `dev` revert of commit `aa89456` — conflict resolution removed `CachedFontDataset` references, which were later restored via `git revert 1d8ec45`. Final state: 22 tests pass, 1 skipped (compile on CPU), `dev` at `d2519bc`. Key config: B=64, w=4, pf=2, AMP, cudnn.benchmark. Files changed: `train.py`, `dataset.py`, `data/build_cache.py`, `configs/train_config.yaml`, `src/model/TRAINING.md`.

## Learnings

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

---

### 2026-03-08: Model Sanity Check Script — Blank Glyph Detection

**Task:** Implement a fast Python ONNX sanity check to catch the blank-glyph bug that the browser smoke test missed.

**Root cause of the gap:** The browser smoke test only checked relative MAD between two style inputs (~0.28 ≈ pass). The model was still responding *relatively* to style, but all absolute output values were clustered near -1.0 (all white/background). No check existed for absolute output quality.

**Deliverables:**

- `src/model/export/check_model.py` — 5 checks: output range, non-blank, style conditioning, character isolation, regression baseline (optional). Exit code 0/1 for CI.
- `src/model/export/export_onnx.py` — Added `--check` flag; `export()` now returns the output path.
- `src/model/TRAINING.md` — "## Model Sanity Check" section with usage, convention table, regression baseline workflow.
- `.squad/decisions/inbox/major-blank-glyph-finding.md` — Diagnosis for Togusa.
- `.squad/decisions/inbox/major-model-sanity-check.md` — Full implementation summary.

**Key design decision — non-blank check:**

Model output space: `+1.0` = black ink, `-1.0` = white background (postprocessing: `((1-output)/2)*255`).  
Non-blank check: **at least 1 % of pixels must be above 0.0** (ink region).  
A blank/all-background model (all output ≈ -1.0) has zero pixels above 0.0 → check fails.  
The old smoke test missed this because it only compared *differences* between runs, not absolute values.

**Regression baseline workflow:**
After confirming a good model, `--save-baselines DIR` saves `.npy` files per input config.  
Subsequent checks with `--baselines DIR` compare against them (MAD ≤ 0.1).

**CI integration:**
`python export/check_model.py <model.onnx>` exits 0/1.  
`python export/export_onnx.py --checkpoint ... --check` chains export + check in one command.

---

### 2026-03-09: Blank Inference Output — Root Cause & Fix

**Task:** Diagnose blank preview glyphs after frontend/backend rebuild.  
**Status:** ✅ FIXED

**Root Cause:**
- Manifest endpoint returns absolute URL: `http://localhost:5000/api/model/...`
- Web workers bypass Vite proxy, fetch directly from port 5000
- Silent failure due to CORS/connection issues, produces blank output

**Solution Applied:**
App.tsx now extracts pathname from absolute URL before passing to worker:
`	ypescript
const url = new URL(manifest.downloadUrl, window.location.origin);
const modelPath = url.pathname; // /api/model/v1/generator.onnx
await modelLoader.load(modelPath, ...);
`

**Additional Changes:**
1. inferenceWorker.ts: Added debug logging for output range verification
2. FontAssembler.ts: Fixed TypeScript error (glyph.name can be null)

**Impact:**
- Development: Critical fix (local dev environment now works)
- Production: No impact (no proxy involved)

**Recommendation:** Consider backend returning relative URLs instead of absolute (would eliminate pathname extraction need).

**Artifacts:**
- Decision merged to decisions.md
- Orchestration log: 2026-03-08T193433Z-major.md

---

### 2026-03-09: Issue #48 — Blank Cyrillic Glyphs Root Cause & Fix

**Task:** Investigate blank Cyrillic glyphs in downloaded font (Latin worked, Cyrillic blank).  
**Status:** ✅ FIXED — commit `cbabc67` on `squad/48-blank-cyrillic-glyphs`

**Root Cause:**
`ort.env.wasm.wasmPaths` was not set in `inferenceWorker.ts`. ORT 1.20 auto-infers the WASM file URL from its own script URL — unreliable inside a Vite-bundled worker chunk (hashed filename). Without valid WASM, ORT silently falls back to a JS path that does not implement INT8 QLinear ops correctly, producing all -1.0 (background) output for every glyph.

**Fixes applied:**
1. `ort.env.wasm.wasmPaths = '/ort-wasm/'` — must be set before any `InferenceSession.create()`
2. `ort.env.wasm.numThreads = 1` — avoid nested proxy worker (we're already in a dedicated worker); also avoids SAB/COOP requirement
3. SharedArrayBuffer guard for `styleGlyphs` before passing to `ort.Tensor` (mirrors OnnxInference.ts)
4. `console.warn` when `max(output[0:512]) <= 0.0` — visible in DevTools without filtering
5. `scripts/copy-ort-wasm.cjs` copies `ort-wasm-simd-threaded.{mjs,wasm}` from node_modules to `public/ort-wasm/`; run via `postinstall`/`dev`/`build`

**Key rule:** ORT WASM files must be served from a stable non-hashed URL. Always use `public/ort-wasm/` + the copy script. Never rely on ORT's auto-detection inside a Vite worker.

**All other 6 checklist items were already correct** (App.tsx pathname, worker message flow, canGenerate gate, tensor shapes, output copy, FontLoader normalisation).

**Artifacts:** `.squad/decisions/inbox/major-blank-cyrillic-findings.md`


---

### 2026-03-09: WASM Path Configuration — PR #49 Merged

**Task:** Fix blank Cyrillic glyphs after Vite build — diagnose and repair browser ONNX inference.  
**Status:** RESOLVED & MERGED

**Root Cause Diagnosis:**
ONNX Runtime 1.20 auto-infers WASM path from import.meta.url of session initialization script. Inside Vite-bundled Web Worker, import.meta.url resolves to blob: protocol, causing WASM auto-discovery to fail silently. Fallback to INT8 WebGL execution provider. WebGL does not support QLinear operations — all-background output (constant -1.0).

**Five-Part Fix:**
1. Explicit ort.env.wasm.wasmPaths = '/ort-wasm/' before InferenceSession.create()
2. ort.env.wasm.numThreads = 1 (WASM inline, no nested proxy worker, avoids SharedArrayBuffer overhead)
3. scripts/copy-ort-wasm.cjs — postinstall script copies WASM files from node_modules to public/ort-wasm/
4. .gitignore updated for src/frontend/public/ort-wasm/
5. SAB defensive guard (copy to plain ArrayBuffer if needed)

**Companion Fixes by Togusa:**
- App.tsx: Added uploadedFont to useCallback deps (stale closure fix)
- GlyphVectorizer.ts: Zero-command warning for diagnostics
- GlyphVectorizer.ts: Fixed misleading "CW" to "CCW" comment

**Validation:**
- 111 frontend tests: all pass
- No regressions
- Togusa full code audit: frontend pipeline confirmed correct, root cause definitely in browser inference

**PR:** #49 — fix(inference): blank Cyrillic glyphs — configure ORT WASM paths  
**Merged:** squad/48-blank-cyrillic-glyphs to dev (squash merge)  
**Branch deleted**

**Key Learning:** ORT 1.20 inside Vite workers requires explicit wasmPaths. Pattern: set BEFORE InferenceSession.create(), copy files on build/dev/postinstall hooks, use numThreads=1 to avoid SharedArrayBuffer in dedicated worker context.

---

### 2026-03-08: ORT WASM Copy Script — Expand to All 8 Variants (PR #50)

**Task:** Fix 404 errors when browser probes for ORT WASM variant files during inference initialization.

**Status:** ✅ APPROVED & MERGED to dev by Saito

**Problem:** ORT 1.20 probes for 4 WASM variants (base, JSEP, asyncify, JSPI) — 2 files per variant (`.mjs` + `.wasm`). Copy script was only copying base variant (2 files). Browser 404s on variant probes.

**Solution:** Expand `FILES` array in `src/frontend/scripts/copy-ort-wasm.cjs` from 2 entries to 8 entries. All variants now copied during `postinstall`, `dev`, and `build` hooks.

**Changes:**
- `src/frontend/scripts/copy-ort-wasm.cjs`: 2 → 8 files (base, JSEP, asyncify, JSPI)

**Verification:**
- ✅ All 8 files present in `src/frontend/public/ort-wasm/`
- ✅ 111 unit tests pass
- ⚠️ E2E failures pre-existing on dev (style-conditioning model issue, unrelated)

**PR:** #50 (`squad/fix-ort-wasm-missing-files` → `dev`)  
**Merged:** Squash merge, commit be94cba  
**Impact:** Zero regression risk. All ORT 1.20 variants now available for browser fallback chain.

