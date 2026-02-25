# Togusa — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **My focus:** Web UI + browser-side inference integration.

## Learnings
<!-- Append new entries below -->

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
