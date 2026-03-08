# Major ORT JSEP Fix

### 2026-03-08: Fix Vite JSEP dynamic import error

**By:** Major  
**What:** Added `ort-wasm-runtime-external` Vite plugin + absolute wasmPaths to prevent Vite 5 from intercepting ORT's dynamic import of `/ort-wasm/*.mjs` from `/public`  
**Why:** ORT 1.20 JSEP probe causes hard Vite 5 error; WASM-only inference doesn't need WebGPU JSEP

## Problem

Vite 5 hard-errors when ORT 1.20 dynamically imports the JSEP module at runtime:

```
Failed to load url /ort-wasm/ort-wasm-simd-threaded.jsep.mjs
(resolved id: /ort-wasm/ort-wasm-simd-threaded.jsep.mjs).
This file is in /public and will be copied as-is during build without going 
through the plugin transforms, and therefore should not be imported from source 
code. It can only be referenced via HTML tags.
```

**Why it happens:**
- ORT WASM files live in `public/ort-wasm/` (static assets served as-is)
- ORT 1.20 dynamically `import()`s the `.jsep.mjs` file at runtime inside the Vite-bundled worker (JSEP = JavaScript Execution Provider for WebGPU)
- Vite 5 intercepts that dynamic `import()`, resolves it against its module graph, finds the target is in `/public`, and throws a hard error
- We're using `executionProviders: ['wasm']` (WASM-only), so we don't want JSEP/WebGPU at all

## Solution

Two-part fix:

### Part 1: Vite plugin to externalize ORT runtime files

Added `ort-wasm-runtime-external` plugin to `vite.config.ts` that marks `/ort-wasm/*.mjs` paths as external, bypassing Vite's module pipeline:

```typescript
{
  name: 'ort-wasm-runtime-external',
  enforce: 'pre',
  resolveId(source) {
    // ORT 1.20 dynamically imports WASM runtime shims from /ort-wasm/.
    // These live in /public and must NOT go through Vite's module pipeline.
    if (/\/ort-wasm\/.*\.m?js/.test(source)) {
      return { id: source, external: true };
    }
  },
}
```

### Part 2: Absolute URL for wasmPaths

Changed `inferenceWorker.ts` to use absolute URL to prevent Vite static-analysis resolution:

```typescript
// Before:
ort.env.wasm.wasmPaths = '/ort-wasm/';

// After:
ort.env.wasm.wasmPaths = `${self.location.origin}/ort-wasm/`;
```

## Files Changed

- `src/frontend/vite.config.ts`: Added `ort-wasm-runtime-external` plugin
- `src/frontend/src/inference/worker/inferenceWorker.ts`: Changed wasmPaths to absolute URL

## Key Learning

Vite 5 aggressively intercepts dynamic imports during dev and build. Files in `/public` that are loaded at runtime via `import()` must be explicitly marked as external via Vite plugin, or Vite will hard-error. Using absolute URLs (`self.location.origin`) in runtime code prevents Vite from attempting static resolution.
