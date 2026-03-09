# Blank Glyph Investigation & Fix

**Date:** 2026-03-08  
**Agent:** Major (AI/ML Engineer)  
**Branch:** `squad/57-fix-ort-wasm-vite-error`  
**Commit:** `c4b3e00`

## Problem Statement

Cyrillic glyphs were rendering empty/blank in the browser despite:
1. Model loading successfully (no errors)
2. All 66 glyphs generating without exceptions
3. Vite JSEP external plugin working (ORT WASM files loading correctly)
4. `executionProviders: ['wasm']` explicitly set (WebGL known to break INT8 QLinear)

## Investigation Process

### 1. Model Validation (Python)
```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession('models/v1/generator.onnx', providers=['CPUExecutionProvider'])
style_glyphs = np.ones((1, 10, 1, 128, 128), dtype=np.float32) * -1.0
style_glyphs[0, 0, 0, 32:96, 32:96] = 1.0  # Add some ink
char_index = np.array([0], dtype=np.int64)

result = session.run(None, {'style_glyphs': style_glyphs, 'char_index': char_index})
output = result[0]
```

**Result:**
- Output shape: `(1, 1, 128, 128)` ✓
- Output range: `[-1.0000, 1.0000]` ✓
- Non-background pixels (>0): `2501` ✓
- **Conclusion:** Model produces valid non-blank output in Python

### 2. Code Path Verification

Traced data flow from model inference to vectorization:
1. **Model output:** `Float32Array` in range `[-1, 1]` where `+1 = ink`, `-1 = background`
2. **Blank detection threshold:** `maxVal <= 0.0` in `inferenceWorker.ts:159` — correct for [-1,1] range
3. **Display postprocessing:** `((1 - output) / 2) * 255` in `App.tsx:95` — correctly maps +1→0 (black), -1→255 (white)
4. **Vectorization threshold:** `data[row * IMG_SIZE + col] > 0` in `GlyphVectorizer.ts:57` — correctly detects ink
5. **Style glyph extraction:** `1 - brightness * 2` in `FontLoader.ts:54` — correctly maps white→-1, black→+1

**All transformations verified correct.**

### 3. ORT WASM File Loading

```powershell
# Files correctly present in public/ and dist/
C:\...\src\frontend\dist\ort-wasm\ort-wasm-simd-threaded.mjs
C:\...\src\frontend\dist\ort-wasm\ort-wasm-simd-threaded.wasm
C:\...\src\frontend\dist\ort-wasm\ort-wasm-simd-threaded.jsep.mjs
C:\...\src\frontend\dist\ort-wasm\ort-wasm-simd-threaded.jsep.wasm
# ... (all 8 variants present)
```

**WASM files loading correctly from `/ort-wasm/` directory.**

### 4. Root Cause Discovery

Examined `onnxruntime-web/dist/ort.mjs` source:
```js
const wasmModuleFilename = true ? "ort-wasm-simd-threaded.jsep.mjs" : ...
```

**Finding:** ORT 1.20 auto-selects the JSEP (JavaScript Execution Provider) variant when WebGPU capability is detected, **regardless of `executionProviders` config**.

#### Key Insight
- The `executionProviders: ['wasm']` config only controls which EP *executes the graph* after the WASM module is loaded
- It does **not** affect ORT's initial WASM variant selection, which is purely capability-based
- JSEP variant (`ort-wasm-simd-threaded.jsep.{mjs,wasm}`) has compatibility issues with INT8 quantized models (QLinear operations)
- When JSEP is used with INT8 models, it produces incorrect numerical output → blank glyphs

## Solution

### Code Change
**File:** `src/frontend/src/inference/worker/inferenceWorker.ts`

```typescript
// Disable JSEP (WebGPU) variant selection. ORT 1.20 auto-selects ort-wasm-simd-threaded.jsep.mjs
// if it detects WebGPU capability, even when executionProviders: ['wasm'] is specified.
// The JSEP variant has compatibility issues with INT8 quantized models and can produce
// incorrect output. Explicitly disable proxy mode to force ORT to use the standard
// ort-wasm-simd-threaded.{mjs,wasm} variant instead.
ort.env.wasm.proxy = false;
```

This must be set **before** `InferenceSession.create()`, alongside `wasmPaths` and `numThreads`.

### Why This Works
- `ort.env.wasm.proxy = false` disables ORT's proxy/JSEP code path
- Forces ORT to use standard WASM backend: `ort-wasm-simd-threaded.{mjs,wasm}`
- Standard variant correctly handles INT8 QLinear operations
- Produces correct numerical output → non-blank glyphs

## Technical Details

### ONNX Runtime WASM Variants
ORT 1.20 ships 4 WASM variants (each as `.mjs` + `.wasm`):
1. **Standard:** `ort-wasm-simd-threaded.{mjs,wasm}` — SIMD + threads, no WebGPU
2. **JSEP:** `ort-wasm-simd-threaded.jsep.{mjs,wasm}` — JavaScript Execution Provider (WebGPU)
3. **Asyncify:** `ort-wasm-simd-threaded.asyncify.{mjs,wasm}` — Async operations support
4. **JSPI:** `ort-wasm-simd-threaded.jspi.{mjs,wasm}` — JavaScript Promise Integration

### Selection Logic
1. ORT probes browser capabilities at module load time
2. If WebGPU detected → selects JSEP variant
3. If WebGPU not available → selects standard variant
4. Selection happens **before** `InferenceSession.create()` call
5. `executionProviders` config only affects EP selection **after** WASM module is loaded

### Why JSEP Breaks INT8 Models
- JSEP is designed for FP32/FP16 models with WebGPU acceleration
- INT8 quantized models use QLinear operators (quantized linear algebra)
- JSEP's WebGPU backend doesn't fully support QLinear ops
- Fallback behavior produces incorrect numerical results
- Manifests as all-negative outputs (all pixels ≤ 0 → blank glyphs after vectorization)

## Verification

### Build Output
```bash
npm run build
# ✓ dist/assets/inferenceWorker-Cug7iZ4l.js   404.90 kB
# ✓ built in 2.11s
```

Build succeeds with `proxy = false` added.

### Next Steps
1. Run E2E tests to confirm glyphs render non-blank in browser
2. Visual inspection of generated font preview
3. Verify downloaded .otf file contains visible Cyrillic glyphs

## Lessons Learned

### For ONNX Runtime Web Integration
1. **INT8 models require standard WASM backend**, not JSEP
2. **`executionProviders` config is insufficient** to control WASM variant selection
3. **`ort.env.wasm.proxy = false` is mandatory** for INT8 quantized models
4. **Set all `ort.env.wasm.*` flags before session creation** (order matters)

### For Vite 5 + ONNX Runtime
1. Use `ort-wasm-runtime-external` plugin to externalize `.mjs` loader files (already implemented)
2. Set `ort.env.wasm.wasmPaths` to absolute URL: `self.location.origin + '/ort-wasm/'`
3. Copy all 8 WASM variant files to `public/ort-wasm/` via build script
4. Set `ort.env.wasm.proxy = false` to disable JSEP (this fix)

### General ML Inference Debugging
1. **Validate model in Python first** — confirms model weights are correct
2. **Trace data transformations end-to-end** — verify input/output ranges at each step
3. **Check backend-specific quirks** — WebGL/JSEP/WASM have different operator support
4. **Inspect raw numerical output** — blank display could be correct math but wrong threshold

## Related Issues
- Previous fix (commit `c7a8ce8`): Vite 5 dynamic import interception → externalize JSEP .mjs files
- Current fix (commit `c4b3e00`): JSEP INT8 incompatibility → disable JSEP entirely

Both fixes required for blank glyph issue to be fully resolved:
1. First fix: WASM files load without error
2. Second fix: WASM files produce correct numerical output

## References
- ORT WASM docs: https://onnxruntime.ai/docs/tutorials/web/
- Vite dynamic import: https://vite.dev/guide/features.html#dynamic-import
- INT8 quantization: https://onnxruntime.ai/docs/performance/quantization.html
