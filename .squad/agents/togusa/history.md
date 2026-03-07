# Togusa — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **My focus:** Web UI + browser-side inference integration.

## Core Context

### Foundational Work Summary (Feb 25 – Mar 6)

**Completed Milestones:**
- PR #15: Fixed opentype.js CJS/ESM interop (vitest.config.ts with resolve.alias)
- PR #14: Fixed CI test failures (jsdom install, ModelLoader singleton async semantics)
- PR #13: Fixed build TS errors (excluded test files from tsconfig, noUnusedLocals/Parameters enabled)
- PR #4: Font assembly pipeline (GlyphVectorizer, FontAssembler, FontDownloader) with OFL metadata
- Issue #16: Fixed model fetch URL (/api/models/v1→ /api/model), verified URL in all test cases
- ONNX integration: Inference contract validated, normalisation convention confirmed (white-on-black training, black-on-white display)
- Web Worker + ModelLoader singleton: Full message protocol (load, infer, progress), request-ID multiplexing, promise-based API
- Vectorization: Scanline-based raster-to-path (basic; Future: proper contours or potrace)
- Font metrics: 1000 UPM fixed, ascender 800, descender -200, advance 600 for all glyphs
- E2E UI: FontUpload → StyleExtract → ModelLoad (singleton) → 66× Infer → AssembleFont → Download
- PR #40 merged: Fixed SharedArrayBuffer output aliasing (Float32Array view → explicit copy before postMessage)
- Test coverage: 92 inference tests + 96 E2E tests + 1 regression test (SAB aliasing), all passing

**Current Status:** Inference pipeline 100% operational. Awaiting Major's model retraining with style conditioning fixes.

**Execution Providers:** WebGL preferred (15–30 ms/glyph), WASM fallback (80–600 ms/glyph).
**Browser Support:** Detects WASM, Workers, WebGL, SharedArrayBuffer; renders unsupported banner if absent.

---

## Learnings

### 2026-03-07: Runtime Debug Instrumentation — Identical-Output Bug (post-PR #40)

- **Task:** Add debug logging to inferenceWorker.ts to diagnose why output glyphs are still identical regardless of input font, even after PR #40's SharedArrayBuffer copy fix.
- **Status:** ✅ Debug logging added. No fix yet — findings only.

**What was added** (inferenceWorker.ts):
1. After session creation in loadModel(): session.inputNames and session.outputNames logged — exposes the ACTUAL key names the loaded ONNX model expects, allowing us to check whether style_glyphs/char_index are correct or if the model expects different names (which would cause silent zero-input fallback).
2. Before session.run(feeds) in unInference(): char_index value, first 5 float values of style_glyphs, and both tensor shapes/dtypes are logged.
3. After session.run(feeds): resolved output key name and first 5 raw float values of outputTensor.data are logged.

**Key architectural findings (no code bugs found in the JS layer):**
- Feed keys style_glyphs and char_index match the documented contract — but there is NO code that reads session.inputNames to verify at runtime. If the actual ONNX model has different input names, ORT would silently use defaults (zeros), and we'd never know without this logging.
- Tensor shapes are correct: [1, 10, 1, 128, 128] float32, [1] int64.
- The double-copy fix from PR #40 is correctly in place (worker + ModelLoader).
- styleGlyphs is a plain 
ew Float32Array(total) from FontLoader.extractStyleGlyphs() — not SAB-backed. postMessage will structurally clone it correctly.
- ModelLoader.loadPromise is only set once. A second font upload updates the store's styleGlyphs and React re-renders handleGenerate with the new value (it's in the dep array). This should be fine.

**Suspected root cause (not yet confirmed):** The ONNX model's actual input names may differ from style_glyphs/char_index. The session.inputNames log will confirm or deny this when run in the browser console.

### 2026-03-07: Style-Invariant Output Root Cause Confirmed

**Coordination with Major:**

Major's investigation revealed the root cause is **NOT** a frontend tensor path issue (as Togusa suspected). Instead:

1. **Architecture weakness:** UNetGenerator encodes 	orch.zeros() → all 6 U-Net skip connections are constant → style signal overwhelmed.
2. **Training loss insufficient:** lambda_l1=100 dominates, no style supervision → model learned to ignore style.
3. **GAN instability:** D loss falling, G loss rising by epoch 22 → mode collapse precursor.

**Implications for Togusa:**
- The 7 debug console.log statements added to inferenceWorker.ts remain valuable for **verification**, not root-cause discovery.
- When browser console logs are captured, they should show:
  - session.inputNames correctly populated with ['style_glyphs', 'char_index'] (they are)
  - Style glyphs first 5 values **different per font** (they should be)
  - Output values **identical** across char_index (confirming model ignores style signal)
- This will serve as validation that the model was trained to ignore style, not that there's a JS-layer bug.

**Current Status:** Debug logging in place, awaiting browser validation. Actual fix requires model retraining by Major.

