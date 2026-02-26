# Togusa — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **My focus:** Web UI + browser-side inference integration.

## Learnings
<!-- Append new entries below -->

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
