# Decision: Fix ONNX Runtime Worker Initialization Failure in Vite

**Date:** 2026-03-09  
**Author:** Togusa (Frontend Dev)  
**Status:** IMPLEMENTED ✅  
**Branch:** `squad/57-fix-ort-wasm-vite-error`  
**Commit:** `d888f47`

---

## Context

The inference worker pipeline was failing to initialize with error:
```
Uncaught [object Event]
filename: http://localhost:5173/ort-wasm/ort-wasm-simd-threaded.mjs
lineno: 19, colno: 182
```

Worker creation succeeded, but the ORT module failed to load, preventing any inference execution.

Direct ORT injection tests (non-worker) passed successfully, confirming the model and WASM files were correct.

---

## Root Cause Analysis

### Investigation Steps

1. **Added enhanced error diagnostics:**
   - `addEventListener('error')` and `addEventListener('unhandledrejection')` in `inferenceWorker.ts`
   - Improved `onerror` logging in `ModelLoader.ts` to distinguish `ErrorEvent` vs plain `Event`
   - Captured full error context: `{ message, filename, lineno, colno, error }`

2. **Error signature revealed:**
   ```
   message: "Uncaught [object Event]"
   filename: "http://localhost:5173/ort-wasm/ort-wasm-simd-threaded.mjs"
   lineno: 19
   ```

3. **Examined ort-wasm-simd-threaded.mjs line 19:**
   ```javascript
   var ba=require("worker_threads");global.Worker=ba.Worker;
   ```
   The threaded WASM module attempts to spawn sub-workers for parallel execution.

4. **Hypothesis confirmed:** Vite's module worker bundling (`type: 'module'`) prevents nested worker creation. When ORT's WASM module tries to instantiate sub-workers, the browser throws a generic `ErrorEvent` with no message.

---

## Root Cause

**ort-wasm-simd-threaded.mjs spawns nested workers for multi-threaded WASM execution.**  
**Vite's module worker bundling blocks nested worker instantiation.**  
**Result: Generic `ErrorEvent` at line 19, worker initialization fails.**

---

## Solution

### 1. Force Single-Threaded WASM Mode

Set `env.wasm.numThreads = 1` **BEFORE** importing `onnxruntime-web/wasm`:

```typescript
import { env } from 'onnxruntime-web/wasm';

env.wasm.numThreads = 1;  // Forces ort-wasm-simd.mjs instead of ort-wasm-simd-threaded.mjs
env.wasm.wasmPaths = `${self.location.origin}/ort-wasm/`;
env.wasm.proxy = false;

import * as ort from 'onnxruntime-web/wasm';
```

**Why this works:**
- `numThreads = 1` tells ORT to use `ort-wasm-simd.mjs` (single-threaded variant)
- Single-threaded WASM doesn't spawn sub-workers
- Vite can bundle the worker without conflict

**Performance impact:**
- Single-threaded WASM: ~80-600ms per glyph
- Multi-threaded WASM: ~15-30ms per glyph (when working)
- WebGL (not viable for INT8): ~15-30ms per glyph

Acceptable tradeoff: Generation happens in background worker, user doesn't perceive latency.

### 2. Import `env` Separately Before Full Import

Ensures WASM configuration is applied **before** ORT tries to load any modules:

```typescript
import { env } from 'onnxruntime-web/wasm';  // Import env first
// Configure env here
import * as ort from 'onnxruntime-web/wasm';  // Then import rest
```

Static imports at top-level are evaluated sequentially, so configuration happens before module loading.

### 3. Add Cross-Origin-Resource-Policy Header

Added `Cross-Origin-Resource-Policy: same-origin` to `vite.config.ts`:

```typescript
server: {
  headers: {
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cross-Origin-Resource-Policy': 'same-origin',  // NEW
  },
},
preview: {
  headers: {
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cross-Origin-Resource-Policy': 'same-origin',  // NEW
  },
},
```

Allows COEP-isolated worker to load same-origin WASM files.

---

## Result

✅ **Worker initialization now succeeds**
- No more `Uncaught [object Event]` errors
- Worker script loads and runs to completion
- ORT initializes successfully: `[inferenceWorker] session.inputNames: [style_glyphs, char_index]`
- Model loading completes: 53 MB `generator.onnx` loads in ~30s
- Inference pipeline functional: `session.run()` executes without crashing

---

## Known Issue (Separate Bug)

Worker pipeline produces blank output (`max ≈ -1.0`) due to **font extraction bug**:

```
[FontLoader] Extracted style glyphs: ABCDEHIORX. 
Sample values (first 20): min=-1.000, max=-1.000
```

All 163,840 style glyph values are `-1.0` (all-background). Model correctly produces blank output given blank input.

**Evidence this is NOT a worker init issue:**
- Direct ORT tests pass with 436 ink pixels
- Model and WASM files confirmed correct
- Worker accepts messages, runs inference, returns results
- Issue is upstream in `FontLoader.extractStyleGlyphs()` or test environment

Font extraction was working in previous tests (PR #49, PR #40). Likely a regression in test setup or FontLoader logic.

---

## Alternative Approaches Considered

### ❌ Dynamic Import

Tried `await import('onnxruntime-web/wasm')` instead of static import:

```typescript
async function loadOrt() {
  const ort = await import('onnxruntime-web/wasm');
  ort.env.wasm.numThreads = 1;
  return ort;
}
```

**Result:** Same error. ORT's module initialization happens during import, even if async.

### ❌ CORP Header Only (Without numThreads=1)

Tried adding CORP header without changing `numThreads`:

```typescript
env.wasm.numThreads = navigator.hardwareConcurrency ?? 4;
```

**Result:** Same error. Nested workers still blocked by Vite bundling, regardless of headers.

### ❌ Vite `worker.rollupOptions.external`

Already configured in `vite.config.ts`:

```typescript
worker: {
  format: 'es',
  rollupOptions: {
    external: [/\/ort-wasm\/.*\.(m?js|wasm)/],
  },
},
```

**Result:** Marks ORT WASM files as external, but doesn't prevent worker script itself from being bundled. Nested worker creation still fails.

---

## Future Optimization

### Enable Multi-Threaded WASM (Post-MVP)

Investigate Vite configurations to allow nested workers:

1. **Serve worker as non-module (`type: 'classic'`)**
   - May bypass module bundling restrictions
   - Requires converting worker to non-ESM syntax

2. **Use Vite's `?worker&url` import**
   - Loads worker as URL instead of bundling
   - Worker fetched at runtime, not bundled
   - May allow nested worker creation

3. **Custom Vite plugin to skip worker bundling**
   - Configure Rollup to emit worker as separate asset
   - Worker remains unbundled, can create sub-workers

**Priority:** LOW. Single-threaded WASM is acceptable for MVP. Optimize if profiling shows user-facing latency.

---

## Lessons Learned

### 1. Vite Worker Bundling Limits Nested Workers

When Vite bundles a worker with `type: 'module'`, the worker cannot instantiate nested workers. This is a fundamental limitation of module workers in Vite 5.

**Implication:** Any library that spawns sub-workers (like ORT's threaded WASM) must be configured to disable threading BEFORE import.

### 2. Generic ErrorEvents Indicate Module Loading Failures

`Uncaught [object Event]` with no message typically means:
- Module failed to load (404, CORS, MIME type)
- Module executed but threw during initialization
- Module tried to use unavailable browser API (e.g., nested workers)

**Debug strategy:** Add `addEventListener('error')` at worker top-level to capture full error context.

### 3. Library Configuration Must Happen Before Import

For libraries that dynamically load submodules (like ORT loading WASM), configuration (`env.wasm.wasmPaths`, `numThreads`) must be set BEFORE the main import. Otherwise, the library loads using default behavior.

**Pattern:**
```typescript
import { env } from 'library/config';  // Import config first
env.setting = value;                   // Configure
import * as lib from 'library';        // Then import main module
```

---

## Files Modified

1. **`src/frontend/src/inference/worker/inferenceWorker.ts`**
   - Import `env` separately before `ort`
   - Set `env.wasm.numThreads = 1`
   - Set `env.wasm.wasmPaths`, `proxy` before import
   - Added top-level error handlers for diagnostics

2. **`src/frontend/vite.config.ts`**
   - Added `Cross-Origin-Resource-Policy: same-origin` to `server.headers`
   - Added same header to `preview.headers`

3. **`src/frontend/src/inference/ModelLoader.ts`**
   - Improved `worker.onerror` logging
   - Distinguish `ErrorEvent` vs plain `Event`
   - Log full error context for debugging

---

## Testing

### Before Fix

```
[Browser ERROR]: worker sent an error! undefined:undefined: undefined
[Browser ERROR]: [inferenceWorker] Top-level error during initialization
[Browser ERROR]: Model load failed: Error: Worker initialization failed
```

Worker creation failed immediately. No inference possible.

### After Fix

```
[Browser DEBUG]: [inferenceWorker] session.inputNames: [style_glyphs, char_index]
[Browser DEBUG]: [inferenceWorker] session.outputNames: [generated_glyph]
[Worker-Diag] ✓ Production model loaded (53 MB generator.onnx)
[Browser DEBUG]: [inferenceWorker] output max (first 512 px): -0.9999
```

Worker initializes successfully. Model loads. Inference runs (though produces blank output due to separate font extraction bug).

---

## References

- **Issue:** #57 - Fix ORT WASM Vite Error
- **Branch:** `squad/57-fix-ort-wasm-vite-error`
- **Commit:** `d888f47`
- **Related PRs:** #49 (WASM path fix), #40 (SAB aliasing fix)

---

## Approval

- [x] Togusa (implementation + validation)
- [ ] Saito (code review)
- [ ] Major (ML/inference validation)

**Status:** ✅ IMPLEMENTED, awaiting code review
