# Togusa — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **My focus:** Web UI + browser-side inference integration.

## Learnings
<!-- Append new entries below -->

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
