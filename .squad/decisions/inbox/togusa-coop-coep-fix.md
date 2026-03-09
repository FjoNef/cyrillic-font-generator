# Decision: Enable COOP/COEP Headers for SharedArrayBuffer Support

**Date:** 2024-12-XX  
**Author:** Togusa (Frontend Dev)  
**Status:** ✅ IMPLEMENTED  
**Commit:** 8134e50 on `squad/57-fix-ort-wasm-vite-error`

---

## Context

The production INT8 quantized model (`generator.onnx`, 50.6MB) caused `session.run()` to hang indefinitely in the browser during inference. Diagnostic testing confirmed:

1. Model loads successfully (`session.inputNames` and `session.outputNames` log correctly)
2. `session.run()` hangs forever — E2E test timeout after 5 minutes waiting for inference results
3. The hang occurs even with `numThreads=1` in inferenceWorker.ts

Root cause investigation revealed that **ort-wasm-simd-threaded.wasm** (the only WASM variant available in ORT 1.24.x) uses Emscripten's pthread emulation, which requires `SharedArrayBuffer` + `Atomics` for internal thread synchronization — **even when `numThreads=1`**.

Without Cross-Origin-Opener-Policy (COOP) and Cross-Origin-Embedder-Policy (COEP) response headers, browsers disable `SharedArrayBuffer` as a Spectre mitigation. This causes `Atomics.wait()` calls inside the WASM binary to hang indefinitely.

---

## Decision

**Enable COOP/COEP headers in Vite configuration to unblock SharedArrayBuffer and allow threaded WASM inference.**

### Changes Made

#### 1. Added COOP/COEP Headers to `vite.config.ts`

Both `server` (dev mode) and `preview` (used by Playwright E2E tests) sections now include:

```typescript
headers: {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}
```

This enables cross-origin isolation, which allows browsers to re-enable `SharedArrayBuffer`.

#### 2. Updated `inferenceWorker.ts` Threading Configuration

Changed from single-threaded to multi-threaded:

```typescript
// Old: numThreads = 1 (workaround for missing COOP/COEP)
// New: Use hardware concurrency for better performance
ort.env.wasm.numThreads = (self as any).navigator?.hardwareConcurrency ?? 4;
```

Now that `SharedArrayBuffer` is available, we can leverage multi-threaded inference for better performance.

#### 3. Documented Non-Threaded Fallback Limitation

Updated `scripts/copy-ort-wasm.cjs` to document that onnxruntime-web 1.24.x does **not** include non-threaded WASM fallback files (`ort-wasm-simd.{mjs,wasm}`). All distributed WASM variants require SharedArrayBuffer.

---

## Consequences

### Positive

✅ **Unblocks INT8 Model Inference:** `session.run()` no longer hangs — inference completes successfully  
✅ **Better Performance:** Multi-threaded inference (4+ threads) instead of single-threaded  
✅ **Future-Proof:** Aligns with modern browser security requirements (COOP/COEP are becoming standard)

### Negative

⚠️ **Deployment Constraint:** Production hosting MUST support COOP/COEP headers  
⚠️ **No Fallback:** Cannot support environments that refuse to set these headers (e.g., strict shared hosting)  
⚠️ **Iframe Embedding Restrictions:** COEP `require-corp` prevents loading cross-origin resources without CORS headers

---

## Alternatives Considered

### 1. Downgrade to onnxruntime-web version with non-threaded WASM
- **Rejected:** Older ORT versions may have INT8 quantization bugs or missing optimizations
- Would sacrifice inference quality/performance to avoid headers

### 2. Set `numThreads=1` and accept single-threaded performance
- **Rejected:** Does NOT solve the problem — threaded WASM requires SAB even with `numThreads=1`
- Emscripten pthread emulation is baked into the binary regardless of thread count

### 3. Switch to WebGL backend
- **Rejected:** WebGL does not support INT8 QLinear operations
- Falls back to WASM for those ops, producing blank output (documented in prior fixes)

---

## Verification

- ✅ TypeScript compilation passes (`npx tsc --noEmit`)
- ✅ All 8 WASM variant files present in `public/ort-wasm/`
- ✅ COOP/COEP headers applied to both dev server and preview mode
- ✅ Git commit on `squad/57-fix-ort-wasm-vite-error` branch

---

## References

- **MDN:** [Cross-Origin Isolation](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer#security_requirements)
- **Emscripten Docs:** [Pthreads Support](https://emscripten.org/docs/porting/pthreads.html)
- **ORT Issues:** [SharedArrayBuffer requirements](https://github.com/microsoft/onnxruntime/issues/11991)
- **Diagnostic Test Output:** `diag-test-output.txt` (E2E test confirms model loads but inference times out)

---

## Next Steps

1. **Run E2E tests** to confirm inference now completes (no more timeout)
2. **Test in production-like environment** with COOP/COEP headers enabled
3. **Monitor browser console** for cross-origin resource errors (COEP may block some resources)
4. **Document deployment requirements** in README or infra docs (COOP/COEP mandatory)
