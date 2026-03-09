# Blank Glyph Browser Fix — Root Cause & Resolution

**Date:** 2026-03-09  
**Agent:** Togusa (Frontend Dev)  
**Branch:** squad/57-fix-ort-wasm-vite-error  
**Status:** ✅ Fixed, all 111 unit tests passing

---

## Root Cause: Wrong ORT Bundle → JSEP WASM Variant

### What was happening

`inferenceWorker.ts` imported from `onnxruntime-web` (default bundle).

In ORT 1.24.x (installed via `^1.20.0`), the default bundle `ort.bundle.min.mjs` hardcodes loading `ort-wasm-simd-threaded.jsep.mjs` (the JSEP/WebGPU variant) as its WASM runtime shim. This is embedded in the bundle source itself — it's NOT controlled by `executionProviders`, `ort.env.wasm.proxy`, or any runtime configuration.

The JSEP (JavaScript Execution Provider) WASM variant is compiled with WebGPU interop and does **not** correctly support INT8 QLinear ops (`QLinearConv`, `QLinearMatMul`, etc.). It silently produces all-background output (values ≈ −1.0) when these ops are present.

The production model `generator.onnx` (50.6 MB, INT8 dynamic quantization) uses QLinear ops extensively → JSEP produces blank output.

The mini model `mini_generator.onnx` (1.26 MB, FP32) has **no** QLinear ops → JSEP handles it correctly → FP32 model appeared to work.

### Why proxy = false didn't fix it

`ort.env.wasm.proxy = false` disables ORT's proxy worker mechanism (a thread that proxies WASM calls). It does NOT affect which WASM variant (jsep vs standard) is loaded. The variant selection is determined at bundle compile time, not at runtime configuration.

### Evidence

Inspecting the ORT 1.24.2 bundles:

```
# ort.bundle.min.mjs (default import — JSEP variant hardcoded):
let i = "ort-wasm-simd-threaded.jsep.mjs";  // ← always JSEP
...
return [..., await Af(d)];

# ort.wasm.bundle.min.mjs (wasm sub-path — standard variant hardcoded):
let d = "ort-wasm-simd-threaded.mjs";  // ← always standard, no JSEP reference
```

---

## Fix Applied

### 1. `inferenceWorker.ts` — change import to `/wasm` sub-path

```diff
- import * as ort from 'onnxruntime-web';
+ import * as ort from 'onnxruntime-web/wasm';
```

`onnxruntime-web/wasm` resolves to `ort.wasm.bundle.min.mjs`, which:
- Hardcodes `ort-wasm-simd-threaded.mjs` (standard non-JSEP variant)
- Contains zero JSEP references
- Correctly handles INT8 QLinear operations

### 2. `OnnxInference.ts` — same import fix (used in unit/E2E tests)

```diff
- import * as ort from 'onnxruntime-web';
+ import * as ort from 'onnxruntime-web/wasm';
```

### 3. `vite.config.ts` — add new import path to optimizeDeps.exclude

```diff
  optimizeDeps: {
-   exclude: ['onnxruntime-web'],
+   exclude: ['onnxruntime-web', 'onnxruntime-web/wasm'],
  },
```

ORT must not be pre-bundled by Vite because it loads WASM files at runtime via dynamic `import()`.

### 4. Test files — update mock paths

Both `onnxContract.test.ts` and `styleConditioning.test.ts` mocked `'onnxruntime-web'`. Updated to `'onnxruntime-web/wasm'` to match the new import.

---

## Verification

- `npx vitest run`: **111/111 tests pass** ✅
- `npm run build`: **Build succeeds** ✅ (standard WASM file emitted to dist/assets)

---

## Normalization Convention (unchanged, confirmed correct)

- Training: white glyph on black bg → glyph=+1.0, bg=−1.0
- FontLoader: `1 - brightness * 2` converts black-on-white render to match training
- Output postprocessing: `((1 - output) / 2) * 255` → +1.0=0 (black ink), −1.0=255 (white bg)
- GlyphVectorizer threshold: `data > 0` is correct for raw [−1, 1] float32 output ✅

---

## Key Lesson

> When ORT WASM produces blank/incorrect output for INT8 models but works for FP32 models, suspect **which WASM bundle is being loaded**, not just which execution provider is configured. The `onnxruntime-web` package exports multiple bundles with different WASM variant selections baked in at compile time.
>
> `ort.env.wasm.proxy`, `executionProviders`, and other runtime flags do NOT control WASM variant selection. Use the `/wasm` sub-path import to guarantee the standard non-JSEP variant.

---

## Decision

**ADOPT**: Always import ORT as `onnxruntime-web/wasm` (not `onnxruntime-web`) when the inference target is CPU WASM with INT8 quantized models. Document in code with the reason.
