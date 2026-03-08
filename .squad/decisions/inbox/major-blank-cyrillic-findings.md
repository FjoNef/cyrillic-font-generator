# major-blank-cyrillic-findings.md
**Date:** 2026-03-09  
**Author:** Major  
**Issue:** #48 — Cyrillic glyphs blank in downloaded font despite font-merge working

---

## Summary

The inference pipeline was producing all-background (-1.0) output for every Cyrillic glyph. Latin glyphs in the merged font appeared correctly because those come from the uploaded font directly (no inference). The bug was confirmed to be in the browser ONNX inference layer, not in the model weights or the font assembly code.

---

## Root Cause

**`ort.env.wasm.wasmPaths` was not set in `inferenceWorker.ts`.**

ORT 1.20 resolves the WASM file URL by inspecting its own bundled script URL (`inferWasmPathPrefixFromScriptSrc`). Inside a Vite-bundled worker, this URL is a hashed chunk path (e.g., `assets/inferenceWorker-Abc123.js`), and ORT cannot derive the correct location of `ort-wasm-simd-threaded.wasm` and `ort-wasm-simd-threaded.mjs` from it.

Without a valid WASM binary, ORT either:
- throws during `InferenceSession.create()` (caught → "model error" state), OR
- silently falls back to a JS-only execution path that does not correctly implement INT8 QLinear operators, returning all -1.0 output.

The INT8-quantized model (`models/v1/generator.onnx`) depends on QLinearConv/QLinearMatMul ops that only work correctly on the real WASM backend. Any fallback produces all-background output.

---

## What Was Already Correct

After thorough code review, all other suspected causes were confirmed correct:

| Item | Status |
|---|---|
| App.tsx: `new URL(manifest.downloadUrl, window.location.origin).pathname` | ✅ already fixed (prior session) |
| Worker message: `modelUrl` sent as `{ type: 'load', modelUrl }` | ✅ correct |
| `canGenerate = styleGlyphs !== null && modelStatus === 'ready'` | ✅ correct |
| Tensor shapes: `style_glyphs [1,10,1,128,128]` float32, `char_index [1]` int64 | ✅ correct |
| Output copy: `new Float32Array(outputData)` in worker, `new Float32Array(msg.output)` in ModelLoader | ✅ correct (double copy, defensive) |
| FontLoader normalization: `1 - brightness*2` → glyph=+1, bg=-1 | ✅ matches training convention |

---

## Fixes Applied

### 1. Set `ort.env.wasm.wasmPaths = '/ort-wasm/'`

Added before any `InferenceSession.create()` call. ORT now loads:
- `/ort-wasm/ort-wasm-simd-threaded.mjs` — inner worker shim
- `/ort-wasm/ort-wasm-simd-threaded.wasm` — WASM binary (~12 MB)

### 2. Set `ort.env.wasm.numThreads = 1`

Disables multi-threaded WASM execution. Since `inferenceWorker.ts` is already a dedicated worker, spinning up a nested proxy worker (ORT's default for threading) is unnecessary. Single-threaded mode also avoids the SharedArrayBuffer + COOP/COEP header requirement.

### 3. Add SharedArrayBuffer guard for `styleGlyphs`

Mirrors the existing guard in `OnnxInference.ts`. If `styleGlyphs.buffer instanceof SharedArrayBuffer`, the data is copied to a plain ArrayBuffer before passing to `ort.Tensor`. ORT WASM cannot accept SAB-backed views.

### 4. Upgrade blank-output check to `console.warn`

Replaced the `console.debug` range check (first 100 values) with a `console.warn` that fires when `max(output[0:512]) <= 0.0`. This immediately surfaces blank-output bugs in the browser console without requiring DevTools filtering.

### 5. `scripts/copy-ort-wasm.cjs`

New Node.js script that copies the two required ORT files from `node_modules/onnxruntime-web/dist/` to `public/ort-wasm/`. Runs via `postinstall`, `dev`, and `build` hooks in `package.json`. The copied files are excluded via `.gitignore`.

---

## Decision: WASM Serving Strategy

ORT WASM files must be served from a stable, non-hashed URL. Using Vite's `public/` directory (served as-is, no hash) is the correct approach. The copy script ensures the files are always current after `npm install`. 

Alternative (`?url` imports with Vite hashing) was considered but rejected: the inner `.mjs` shim has its own internal imports that would break if the binary URL changed independently.

---

## Files Changed

- `src/frontend/src/inference/worker/inferenceWorker.ts`
- `src/frontend/scripts/copy-ort-wasm.cjs` (new)
- `src/frontend/package.json`
- `.gitignore`
