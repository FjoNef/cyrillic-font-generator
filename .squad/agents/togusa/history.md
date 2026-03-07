# Togusa — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **My focus:** Web UI + browser-side inference integration.

## Core Context

### Prior Work Completed (Feb 26 – Mar 5)
- ✅ PR #15: Fixed opentype.js CJS/ESM interop in Vitest (resolved TypeError on module load; added vitest.config.ts with resolve.alias)
- ✅ PR #14: Fixed CI test failures (jsdom install, ModelLoader singleton, async Promise semantics)
- ✅ PR #13: Fixed build TS errors by excluding test files from tsconfig
- ✅ PR #4: Font assembly pipeline (GlyphVectorizer, FontAssembler, FontDownloader) with OFL metadata
- ✅ Issue #16: Fixed model fetch URL (/api/models/v1/generator.onnx → /api/model), all 41 tests passing
- ✅ Frontend inference end-to-end: Web Worker + ModelLoader singleton + vectorization + font assembly complete
- **Dependency:** Waiting for Major's ONNX model at models/v1/generator.onnx — once available, inference pipeline fully functional

## Learnings

### 2026-03-07T14:31:04Z: Major exports ONNX model — Ready for Integration

- **Status:** ✅ COMPLETE
- **Model file:** models/v1/generator.onnx
- **Size:** 53.1 MB (INT8 quantized)
- **Compressed delivery:** ~15.9 MB (meets ≤20 MB browser target)
- **Output shape:** (1, 1, 128, 128)
- **Output dtype:** float32
- **Value range:** [-1.0, 1.0]
- **Validation:** CPU inference SUCCESS
- **Implementation:** INT8 dynamic quantization; ConvTranspose FP32 fallback
- **Key note:** Model ready for onnxruntime-web integration with ModelLoader singleton pattern

### 2026-03-07: ONNX Browser Integration Complete — Cross-Validated by Saito

- **OnnxInference.ts**: Updated to match confirmed contract — `style_glyphs` tensor shape `[1, 10, 1, 128, 128]` (batch dim required), output from `generated_glyph` key. All TODO placeholders resolved.
- **Normalisation convention confirmed**: Training data renders white glyphs on black background (`glyph=255 → +1.0`, `bg=0 → -1.0`). FontLoader's `1 - brightness * 2` formula is correct (inverts rendered black-glyph-on-white to match training). Postprocessing `((1 - output) / 2) * 255` is correct.
- **Style chars confirmed**: Model trained on `["A","B","C","D","E","H","I","O","R","X"]` (see `dataset.py` DEFAULT_STYLE_CHARS). FontLoader was already correct. Inference contract doc had wrong chars — not a code issue.
- **browserSupport.ts created**: Detects WASM / Workers / WebGL / SharedArrayBuffer; returns recommended execution providers and human-readable error for unsupported browsers.
- **Tests**: 92/92 passing. Saito flagged HIGH risk (batch dim + output key bugs) **independently** → Togusa had **already fixed** these before Saito filed the risk. Cross-validation successful.
- **Dependency resolved**: inference pipeline is now fully wired to the actual exported model. ModelLoader fetches from `/api/model/manifest` → gets `downloadUrl` from Batou's versioned API endpoint.
- **Cross-agent sync**: Batou's `/api/model/v1/generator.onnx` endpoint matches Togusa's fetch URL exactly. Decision locked in both histories.

### Issue #27: E2E Glyph Generation Flow — Integration Tests Added

- **Branch**: `squad/27-e2e-glyph-generation-ui` (existing), **PR #31** (open, targeting dev)
- **Status**: ✅ COMPLETE — 96/96 tests passing
- **What was done**: Branch `squad/27-e2e-glyph-generation-ui` already contained the full E2E UI flow (App.tsx orchestrates FontUpload → ModelLoader.infer × 66 → assembleFontFromGlyphs → downloadFont). Added real integration tests replacing all placeholders in `inference/__tests__/integration.test.ts`.
- **Integration tests cover**:
  - Full 66-glyph inference loop: all char indices 0–65 requested exactly once, style_glyphs tensor size asserted
  - Valid OTF output: magic-byte check (OTTO / 0x00010000) after assembleFontFromGlyphs
  - Monotonic progress: 1→66 increments verified
  - Error propagation: mid-loop inference failure surfaces to caller
  - Guard: inference rejects before model loaded
  - downloadFont: correct MIME type (font/otf), filename, URL.revokeObjectURL cleanup
- **Key note**: When creating a feature branch, check for existing branches/PRs for the same issue first. `squad/27-e2e-glyph-generation-ui` pre-existed with the implementation; duplicate branch avoided.


- **Branch:** fix/togusa-ci-ts-errors
- **PR:** #13 → dev
- **Problem:** `npm run build` (tsc && vite build) included `src/**/__tests__/**` and `**/*.test.ts` files. With `noUnusedLocals` and `noUnusedParameters` enabled, 7 TS6133 errors in test files broke CI.
- **Root cause:** `tsconfig.json` had `"include": ["src"]` with no `exclude`, so all test files were compiled.
- **Fix:** Added `"exclude": ["src/**/__tests__/**", "src/**/*.test.ts", "src/**/*.spec.ts"]` to `src/frontend/tsconfig.json`.
- **Verified:** `npx tsc --noEmit` exits 0 after fix.
- **Key lesson:** Always exclude test files from the build tsconfig when `noUnusedLocals`/`noUnusedParameters` are enabled. Vitest has its own tsconfig or runs with relaxed settings; the build tsc should not see test files.

### 2026-02-26: Font assembly pipeline implemented
- **Branch:** feat/togusa-font-assembly
- **Files created:**
  - `src/frontend/src/GlyphVectorizer.ts` — scanline vectorizer with correct coordinate mapping
  - `src/frontend/src/FontAssembler.ts` — OTF assembly from `Map<number, Float32Array>` with OFL license metadata
  - `src/frontend/src/FontDownloader.ts` — `downloadFont(buffer, filename)` helper
- **Files modified:**
  - `src/frontend/src/App.tsx` — single inference pass (no double inference), wired to FontAssembler + FontDownloader
- **Key bug fixed:** FontLoader.vectorizeGlyph had wrong X scale (1000/128 instead of 600/128) and missing Y ascender offset. GlyphVectorizer uses corrected math.
- **Architecture:** FontAssembler takes `Map<number, Float32Array>` (model index → raw output); App.tsx collects this during the generation loop and passes it post-loop. No re-inference for assembly.
- **OFL:** License text (name ID 13) and URL (name ID 14) written into opentype.js name table via `font.names.license` / `font.names.licenseURL`.
- **Build status:** My new files introduce 0 TypeScript errors. 7 pre-existing errors in test files (unused `vi` import, unused params) remain; they predate this branch.
- **Cross-team notes:** Saito wrote fontPipeline.test.ts (15 test cases) to spec first; tests validate coordinate math, glyph assembly, and download lifecycle. Ready for code review and merge.

### 2026-02-25T162500: Fixed PR #8 doc conflict (Reviewer Rejection Lockout Protocol)
- **Timestamp:** 2026-02-25T162500
- **Branch:** feat/major-model-training (PR #8)
- **Trigger:** Saito review flagged `.squad/decisions.md` line 184 had wrong Latin style reference chars
- **Error:** Listed as `A, B, H, O, g, n, o, p, s, x` (mixed-case, wrong set)
- **Correct chars:** A, B, C, D, E, H, I, O, R, X (10 uppercase Latin)
- **Fixes applied:**
  1. Corrected `.squad/decisions.md` line 184 to match actual code
  2. Added clarifying comment to `models/train/model.py` docstring (line 202-204) documenting char set
  3. Verified `models/train/dataset.py` already correct (line 17-18)
  4. Verified `models/train/README.md` already correct (lines 120, 154)
- **Protocol:** Major was locked out after submitting PR (Reviewer Rejection Lockout); I applied fix per Saito's request
- **Outcome:** Committed fix to feat/major-model-training; posted comment on PR #8
- **Result:** Saito re-reviewed, approved PR; ready for merge

### 2026-02-25T165444: PR #8 & PR #9 merged — awaiting Major's model export
- **Status:** Both PRs approved and merged to dev. Frontend (PR #4) is live but non-functional until model available.
- **PR #8 Merge (Major — cGAN Training):**
  - Fixed by Togusa: Doc conflict (decisions.md wrong Latin chars) resolved under Reviewer Lockout Protocol
  - Approved by Saito; merged with both original code and doc fix
  - **Next:** Major trains on Google Fonts, exports ONNX to models/v1/generator.onnx
- **PR #9 Merge (Batou — Backend Integration):**
  - Approved by Aramaki; all 4 integration tests passing
  - /api/model endpoint ready to serve trained model
- **Inference Pipeline Status:**
  - PR #4 already merged to dev (feat/togusa-inference-pipeline)
  - Web Worker protocol complete; model loader singleton implemented
  - Vectorization (scanline raster-to-path) complete
  - Font assembly (1000 UPM fixed metrics, 66 Cyrillic glyphs) complete
  - **Waiting:** ONNX model from Major at models/v1/generator.onnx
  - Once model available, frontend inference end-to-end functional without code changes
- **Cross-agent:** Frontend is dependency; unblocked once Major exports model

### 2026-02-25T152319: PR #4 color inversion bug fixed by Major
- **Branch:** feat/togusa-inference-pipeline
- **Issue:** Inverted color output in App.tsx line 67 (formula mapped -1→black, 1→white; should be opposite)
- **Fix by:** Major (AI/ML Engineer) — formula corrected to `((1 - output[px]) / 2) * 255`
- **Files modified:** App.tsx, OnnxInference.ts
- **Status:** Committed to feat/togusa-inference-pipeline; PR #4 unblocked and ready for re-review by Saito
- **Context:** Demonstrates importance of explicit model output convention documentation and integration testing with real ONNX output

### 2026-02-25T143900: Issue #3 — Inference pipeline fully wired
- **Branch:** `feat/togusa-inference-pipeline`
- **PR:** #4 → dev
- **Files created/modified:**
  - NEW `src/frontend/src/inference/worker/inferenceWorker.ts` — Web Worker for ONNX inference (WebGL preferred, WASM fallback)
  - NEW `src/frontend/src/inference/ModelLoader.ts` — Promise-based singleton wrapping worker with request ID management
  - MODIFIED `src/frontend/src/font/FontLoader.ts` — Implemented `assembleCyrillicFont` with threshold-based vectorization (scanline approach)
  - MODIFIED `src/frontend/src/components/FontUpload.tsx` — Extract style glyphs on font upload
  - MODIFIED `src/frontend/src/App.tsx` — Full pipeline: model load on mount, generate all 66 glyphs, download .otf
  - MODIFIED `src/frontend/src/stores/appStore.ts` — Added `styleGlyphs`, `generationProgress`, `fontBuffer`, `reset()` action
  - `package.json` — Added `@types/opentype.js` devDependency

- **Key implementation decisions:**
  - **Web Worker message protocol:** `{ type: 'load' | 'infer' | 'progress' | 'loaded' | 'result' | 'error', ... }` with per-request IDs for concurrent safety
  - **Vectorization approach:** Potrace is not browser-friendly; implemented simple scanline-based raster-to-path conversion (draws horizontal segment rectangles per row)
  - **Font metrics:** 1000 UPM, ascender 800, descender -200, advance width 600 (fixed for all glyphs)
  - **Tensor contract:** Confirmed from decisions.md — `style_glyphs` [1,10,1,128,128] float32, `char_index` [1] int64, output `generated_glyph` [1,1,128,128] float32 range [-1,1]
  - **Vite Worker syntax:** `new Worker(new URL('./worker/inferenceWorker.ts', import.meta.url), { type: 'module' })` for proper HMR and bundling

- **Known issues/decisions:**
  - Vectorization is basic (scanline rectangles) — not smooth curves. Future enhancement: proper contour tracing or potrace port.
  - No error boundary UI yet — errors logged to console.
  - Model load progress shown in ModelLoadingBar component (already existed in scaffold).
  - Generation shows progress counter in button text: "Generating… (N/66)".

### 2026-02-25: Frontend scaffold created
- Full project in `src/frontend/` — React 18 + TypeScript + Vite + Tailwind + Zustand
- `src/frontend/src/inference/OnnxInference.ts` — ONNX Runtime Web wrapper; tensor input names are placeholders, **Major must confirm**: input names (`style_glyphs`, `char_index`), output name (`output`), and exact tensor shapes `[10,1,128,128]`
- `src/frontend/src/font/FontLoader.ts` — renders 10 Latin glyphs (A B C D E H I O R X) at 128×128 to Float32Array [-1,1]; `assembleCyrillicFont` is a TODO stub pending potrace integration
- `src/frontend/src/font/cyrillicCharset.ts` — 66 chars: uppercase indices 0-32 (А-Я index 0-31, Ё index 32), lowercase indices 33-65 (а-я 33-64, ё 65)
- Vite dev server proxies `/api` → `http://localhost:5000` (Batou's backend)
- onnxruntime-web excluded from Vite dep optimization; WASM files declared as assets
- Model loaded lazily via `fetch` with streaming progress — hooks into Zustand `modelLoadProgress`
- Key integration point with **Major**: ONNX model input/output tensor names and shapes
- Key integration point with **Batou**: `/api` proxy target, model file served at what URL?

### 2026-02-26: Fixed model URL mismatch (Issue #16)

- **Issue:** Frontend was fetching ONNX model from wrong URL — used `/api/models/v1/generator.onnx` but backend serves at `/api/model`
- **Files changed:**
  - `src/frontend/src/App.tsx` (line 34)
  - `src/frontend/src/inference/__tests__/ModelLoader.test.ts` (9 occurrences across all test cases)
- **Fix:** Changed all model fetch URLs from `/api/models/v1/generator.onnx` → `/api/model`
- **Verification:** grep confirmed zero stale references remain in src/frontend/
- **Result:** Model loading will now succeed in production; frontend correctly calls Batou's `/api/model` endpoint


### 2026-03-07: E2E Glyph Generation UI Wired (Issue #27, PR #31)

- **Branch:** squad/27-e2e-glyph-generation-ui → dev (PR #31)
- **Status:** ✅ COMPLETE — 96/96 tests passing

**What was done:**
- Added `BrowserUnsupported.tsx` component: amber alert shown when WASM or Web Workers are absent, listing missing capabilities and recommending modern browsers
- Updated `App.tsx` with `detectBrowserSupport()` at module load; model fetch skipped on unsupported browsers; `BrowserUnsupported` banner rendered in `<main>` when gate fails
- Updated generation progress label to `Generating Cyrillic glyphs: X/66…` (per issue spec)

**E2E flow (all wired since PR #4, gate added this PR):**
1. Upload TTF/OTF → `FontLoader.extractStyleGlyphs()` → `Float32Array [10×1×128×128]`
2. `ModelLoader.load('/api/model')` once at mount; singleton reused for all 66 inferences
3. Loop over `CYRILLIC_CHARS`: `modelLoader.infer(styleGlyphs, index)` → raw `Float32Array [-1,1]`
4. Pixel denorm: `((1 - output[i]) / 2) * 255` → `ImageData` for preview
5. `assembleFontFromGlyphs(rawGlyphs, fontName)` → `ArrayBuffer`
6. `downloadFont(buffer, '*.otf')` on button click

**Key notes:**
- `BrowserUnsupported.test.tsx` (4 tests) already existed as untracked from prior branch work — committed here with the component
- Model URL confirmed as `/api/model` — backend `HandleModelDownload` at that route
- `detectBrowserSupport()` called synchronously at module load (zero React overhead)

