### 2026-03-08T203742Z: Copy all ORT WASM variants
**By:** Major (via FjoNef request)  
**What:** The copy script must copy ALL 8 ORT WASM variant files, not just the base pair. ORT 1.20 probes for jsep, asyncify, jspi variants based on browser capabilities.  
**Why:** 404 on jsep.mjs caused silent inference failure. All variants needed to avoid future 404s as ORT probes for different backend files depending on browser features (WebGPU, async support, etc.).

**Files copied:**
- `ort-wasm-simd-threaded.mjs` / `.wasm` — base SIMD+threads backend
- `ort-wasm-simd-threaded.jsep.mjs` / `.wasm` — JavaScript Execution Provider (WebGPU)
- `ort-wasm-simd-threaded.asyncify.mjs` / `.wasm` — async operations support
- `ort-wasm-simd-threaded.jspi.mjs` / `.wasm` — JavaScript Promise Integration

**Impact:** Prevents 404 errors during ORT capability probing, ensures inference works correctly across all browser configurations.
