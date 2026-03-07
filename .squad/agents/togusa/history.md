# Togusa — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **My focus:** Web UI + browser-side inference integration.

## Learnings
<!-- Append new entries below -->

### 2026-02-26: Fixed opentype.js CJS/ESM interop in Vitest (PR #15 MERGED)

- **Branch:** fix/togusa-opentype-vitest-interop
- **PR:** #15 → dev ✅ MERGED
- **Problem:** `fontPipeline.test.ts` and `FontLoader.test.ts` crashed at suite load time with `TypeError: Cannot assign to read only property 'load'`. 0 tests ran in either suite; the other 3 suites and their 20 tests were unaffected.
- **Root cause:** opentype.js ships a CJS/UMD bundle (`dist/opentype.js`) and a native ESM build (`dist/opentype.module.js`). Vitest resolved `opentype.js` to the CJS bundle. Its UMD factory does `exports.load = load` at module-evaluation time, but Vitest's ESM strict mode seals the `exports` object, making that assignment throw. Neither `server.deps.inline`, `deps.inline`, nor `deps.optimizer.web.include` prevented the issue in practice with Vitest 1.6.
- **Fix (vitest.config.ts):** Created `src/frontend/vitest.config.ts` with `resolve.alias: { 'opentype.js': '.../dist/opentype.module.js' }`. This forces Vitest to always resolve the ESM build, bypassing the CJS bundle entirely.
- **Secondary fixes revealed by the above:**
  1. `FontLoader.test.ts` was missing `// @vitest-environment jsdom` — `extractStyleGlyphs` calls `document.createElement('canvas')`, which requires a browser env.
  2. jsdom doesn't implement `URL.createObjectURL/revokeObjectURL`, `HTMLCanvasElement.getContext('2d')`, or `Path2D`. Added `src/frontend/src/test-setup.ts` to stub all of these, guarded with `typeof` checks so node-env inference tests are unaffected.
- **Result:** All 41 tests across all 5 suites pass (up from 20 passing with 2 suites crashing).
- **Key lesson:** When a CJS/UMD package has a `module` field pointing to a native ESM build, the most reliable Vitest fix is `resolve.alias` to force the ESM file — not `deps.inline`/`server.deps.inline`, which in practice didn't intercept the load in Vitest 1.x jsdom pool. Also: always add `@vitest-environment jsdom` to any test file that touches `document`, `canvas`, or other DOM APIs.

### 2026-02-26: Fixed CI test failures — jsdom + ModelLoader singleton/async pattern

- **Branch:** fix/togusa-ci-test-failures
- **PR:** #14 → dev
- **Problem 1:** `Cannot find package 'jsdom'` — `fontPipeline.test.ts` uses `// @vitest-environment jsdom` but jsdom was absent from devDependencies.
- **Fix 1:** `npm install --save-dev jsdom` added jsdom to package.json devDependencies.
- **Problem 2:** `mockWorker.onmessage is not a function` at lines 85, 94, 102, 138, 174 in ModelLoader.test.ts — three interrelated root causes:
  1. `modelLoader` singleton retained `loadPromise`/`worker` across tests; subsequent tests got fresh `mockWorker` from `beforeEach` but `onmessage` handler was set on the OLD worker.
  2. `load()` was `async`, so every call returned a new Promise wrapper — the `toBe` same-reference assertion in "same promise for concurrent loads" always failed.
  3. `infer()` internally `await this.loadPromise` before calling `postMessage`; even when resolved, `await` defers to microtask queue — tests reading `mock.calls` synchronously right after `infer()` saw empty arrays.
- **Fix 2a:** Exported `ModelLoader` class; test uses `new ModelLoader()` in `beforeEach` for a clean instance.
- **Fix 2b:** Removed `async` from `load()` — it already builds and returns a `Promise<void>` manually, so returning `this.loadPromise` directly gives same object reference.
- **Fix 2c:** Added `await Promise.resolve()` microtask flush in 3 tests after calling `infer()` before reading `mockWorker.postMessage.mock.calls`.
- **Key lesson:** Never export a stateful singleton as the only test surface — export the class too. And `async` on a function that manually returns a Promise wraps it in a second Promise, breaking reference equality. Finally, `await resolvedPromise` still defers to the microtask queue; synchronous mock reads after `infer()` must flush first.


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


### 2026-03-05: Sprint Complete --- #16 Closed
**Issue:** #16 (Model fetch URL fix)  
**Status:** OK IMPLEMENTATION COMPLETE  
**Dependencies:** Batou's #17 (backend model path fix)

**Change:** App.tsx model fetch endpoint  
- Old: /api/models/v1/generator.onnx  
- New: /api/model

Updated test file to match new endpoint. All 41 frontend tests passing.

Backend now exposes /api/model endpoint (Batou's fix resolves actual model file via ContentRootPath).

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
