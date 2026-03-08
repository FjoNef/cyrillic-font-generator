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

### 2026-03-07: SAB Alias Fix — OnnxInference.ts (Issue #41)

- **Task:** Fix SharedArrayBuffer aliasing vulnerability in `OnnxInference.ts` (test-only inference path).
- **Status:** ✅ Fixed. PR #44 opened against `dev`.

**What was found:**
- `OnnxInference.generateGlyph()` at line 83 passed `styleGlyphs` directly to `new ort.Tensor('float32', styleGlyphs, ...)` without checking if it was SAB-backed.
- PR #40 had already fixed the output-side aliasing in `inferenceWorker.ts` (line 131: `return new Float32Array(outputData)`), but the input path in `OnnxInference.ts` was missed.
- The `inferenceWorker.ts` input tensor construction at line 98 also lacks an explicit SAB guard — but that path receives data via `postMessage` (which structurally clones non-SAB arrays), so it is lower risk.

**What was fixed:**
- Added SAB detection before creating `styleTensor` in `OnnxInference.ts`:
  ```ts
  const safeStyleGlyphs = styleGlyphs.buffer instanceof SharedArrayBuffer
    ? new Float32Array(styleGlyphs.buffer.slice(styleGlyphs.byteOffset, styleGlyphs.byteOffset + styleGlyphs.byteLength))
    : styleGlyphs;
  ```
- 38 existing onnxContract tests all pass with the fix.
- Branch: `fix/sab-alias-onnxinference-41`, PR #44.

**Key file paths:**
- Bug site: `src/frontend/src/inference/OnnxInference.ts` (line 83)
- Reference pattern: `src/frontend/src/inference/worker/inferenceWorker.ts` (line 131)
- Tests: `src/frontend/src/inference/__tests__/onnxContract.test.ts`

---

### 2026-03-07: SAB Alias Fix in OnnxInference.ts (Issue #41) [Orchestrated]

Decision: Conditional SAB copy strategy (not unconditional) to avoid 640 KB unnecessary allocation on normal test paths. Consistent with PR #40 output fix. Decision merged to decisions.md. PR #44 ready for review.

---

### 2026-03-07: ImageData Mock Fix for Vitest jsdom (PR #47 CI)

- **Task:** Fix `ReferenceError: ImageData is not defined` in styleConditioning.test.ts (6 tests failing in PR #47 CI).
- **Status:** ✅ Fixed. Committed to `squad/46-training-triton-fonts` (sha: 7863589).

**Root Cause:**
- `OnnxInference.generateGlyph()` line 111 uses `new ImageData(pixels, size, size)` to construct browser ImageData objects.
- jsdom (Vitest's default test environment) does not provide a native `ImageData` constructor.
- The test file already had `@vitest-environment jsdom` directive, but jsdom lacks Canvas API constructors.

**Fix Applied:**
- Added minimal `ImageData` class mock to `src/frontend/src/test-setup.ts` (lines 38-53).
- Supports both Canvas API constructors:
  1. `new ImageData(data: Uint8ClampedArray, width: number, height: number)`
  2. `new ImageData(width: number, height: number)`
- Mock only applied if `globalThis.ImageData` is undefined (guards against future jsdom/happy-dom upgrades).
- Pattern consistent with existing `Path2D` and `getImageData` mocks in same file.

**Verification:**
- styleConditioning.test.ts: 6/6 passing (previously 0/6)
- Full frontend suite: 108/108 passing (no regressions)

**Key Files:**
- Fix: `src/frontend/src/test-setup.ts`
- Test: `src/frontend/src/inference/__tests__/styleConditioning.test.ts`
- Production: `src/frontend/src/inference/OnnxInference.ts` (unchanged)

