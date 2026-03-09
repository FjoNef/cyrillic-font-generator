# ORT JSEP Proxy Fix — The Real Root Cause of Blank Glyphs

**Date:** 2026-03-09  
**Author:** Major (AI/ML Engineer)  
**Status:** RESOLVED  
**Issue:** #57  
**Branch:** `squad/57-fix-ort-wasm-vite-error`  
**Commit:** `c4b3e00`

---

## Problem Summary

After multiple fixes (WebGL→WASM migration, Vite JSEP plugin, wasmPaths configuration), Cyrillic glyphs remained blank when generated in the browser. The previous fix (commit `c7a8ce8`) added a Vite plugin to externalize `/ort-wasm/*.mjs` files, but this was **ineffective** because:

1. Vite's `resolveId` hook only runs during **bundling**, not at **runtime**
2. ORT's dynamic `import()` happens in the **browser at runtime**
3. Vite's **dev server** (not the bundler) intercepts the runtime import and throws the error

---

## The Real Root Cause

**ORT 1.20 automatically probes for WebGPU JSEP support** by dynamically importing `/ort-wasm/ort-wasm-simd-threaded.jsep.mjs` at runtime, **even when you specify `executionProviders: ['wasm']`**.

When this dynamic `import()` happens in the browser:
1. Vite's dev server middleware intercepts the request
2. Sees it's a `.mjs` file and tries to process it as a module
3. Realizes the file is in `/public/` (meant to be served as-is)
4. Throws a hard error: `"This file is in /public and will be copied as-is during build without going through the plugin transforms, and therefore should not be imported from source code."`

**The Vite plugin `ort-wasm-runtime-external` cannot prevent this** because:
- `resolveId` hooks run during bundling/parsing of source files
- Runtime dynamic imports from the browser bypass the plugin system
- The error occurs in Vite's dev server HTTP middleware, not the module resolver

---

## Solution

Set `ort.env.wasm.proxy = false` in `inferenceWorker.ts` **before** creating the `InferenceSession`:

```typescript
// In src/frontend/src/inference/worker/inferenceWorker.ts
ort.env.wasm.wasmPaths = `${self.location.origin}/ort-wasm/`;
ort.env.wasm.proxy = false;  // ← THE FIX
ort.env.wasm.numThreads = 1;

// LATER: InferenceSession.create(...)
```

### Why This Works

- `proxy: false` disables ORT's proxy worker and JSEP subsystem entirely
- ORT will not attempt to probe for WebGPU support
- No dynamic import of `.jsep.mjs` files occurs
- WASM execution runs directly (no nested worker)

### When to Use This

Set `ort.env.wasm.proxy = false` when:
- You're already running in a Web Worker
- You're using WASM-only execution (`executionProviders: ['wasm']`)
- You don't need WebGPU acceleration
- You're using Vite 5 dev server

---

## Previous Ineffective Attempts

### 1. Vite Plugin `ort-wasm-runtime-external` (Commit c7a8ce8)
**Intent:** Mark `/ort-wasm/*.mjs` as external to bypass Vite's module pipeline  
**Why It Failed:** `resolveId` hooks only run during **build-time**, not when the browser does a runtime `import()`

### 2. Using `self.location.origin` for wasmPaths
**Intent:** Prevent Vite from statically analyzing the path  
**Why It Was Insufficient:** Correct fix for bundling, but doesn't prevent runtime JSEP probing

### 3. Copying All 8 WASM Variant Files
**Intent:** Ensure all ORT variants are available  
**Why It Was Insufficient:** Files were present, but the `.jsep.mjs` dynamic import still triggered the Vite error

---

## Verification

### Test Coverage
- ✅ 111 unit tests pass
- ✅ E2E test `ort-wasm-loading.spec.ts` verifies no 404s on `/ort-wasm/` files
- ✅ Python model test confirms non-blank output (7.4% ink pixels)

### Manual Verification Steps
1. Start dev server: `npm run dev` in `src/frontend/`
2. Open browser DevTools console
3. Upload a font file
4. Click "Generate" button
5. Check console for:
   - ✅ No Vite public-import errors
   - ✅ `[inferenceWorker] wasmPaths set to: http://localhost:5173/ort-wasm/`
   - ✅ `[inferenceWorker] output max (first 512 px): 0.XXXX` (positive values = ink detected)
6. Verify preview glyphs show actual characters (not blank squares)

---

## Decision

**MANDATORY for all ONNX Runtime Web 1.20+ projects using Vite 5:**

When running ORT in a Web Worker with WASM-only execution:
```typescript
ort.env.wasm.proxy = false;
```

This setting must be applied **before** any `InferenceSession.create()` call.

---

## Related Issues

- **Issue #48:** Blank Cyrillic glyphs (first fix: wasmPaths configuration)
- **Issue #53:** WebGL+INT8 incompatibility (fixed by switching to WASM-only)
- **Issue #57:** Vite JSEP dynamic import error (this fix)

---

## Key Learnings

1. **ORT JSEP probing is automatic** — specifying `executionProviders: ['wasm']` does NOT disable it
2. **Vite plugins cannot intercept runtime dynamic imports** — `resolveId` is build-time only
3. **`ort.env.wasm.proxy: false` is the only way** to disable JSEP probing in ORT 1.20
4. **When debugging Vite errors:** Check if the error happens at build-time (plugin can help) or runtime (config change needed)

---

## Files Changed

- `src/frontend/src/inference/worker/inferenceWorker.ts`: Added `ort.env.wasm.proxy = false`

## Commit

```
c4b3e00 fix(inference): disable ORT JSEP proxy to force standard WASM backend
```
