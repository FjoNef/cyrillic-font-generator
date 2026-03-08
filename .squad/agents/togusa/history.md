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

**Current Status:** Inference pipeline 100% operational. ✅ **MAJOR UPDATE (2026-03-08):** Fresh ONNX model exported and validated. Ready for inference test cycle. ✅ **SMOKE TEST COMPLETE (2026-03-08T11:43:53Z):** Style conditioning confirmed working after retraining. All 128 tests green.

**Execution Providers:** WebGL preferred (15–30 ms/glyph), WASM fallback (80–600 ms/glyph).
**Browser Support:** Detects WASM, Workers, WebGL, SharedArrayBuffer; renders unsupported banner if absent.

### Inference Bug Fixes & Test Hardening (Mar 7–8)
SAB aliasing fixed in `OnnxInference.ts` (PR #44): conditional copy before creating `styleTensor` guards the test-only inference path; the worker path was already fixed in PR #40. `ImageData` class mock added to `src/frontend/src/test-setup.ts` for jsdom compatibility (jsdom lacks Canvas API constructors). Style conditioning smoke test suite (`src/frontend/e2e/style-conditioning-real.spec.ts`) created against real `models/v1/generator.onnx`: 3 tests (shape/range, style conditioning MAD=0.281, determinism), Chromium-only due to 53 MB WASM compile cost. INT8 epsilon finding: quantized model can produce values like `-1.0000001192092896` — assertions require ±1e-6 tolerance. Debug instrumentation added to `inferenceWorker.ts` (7 console.log statements logging session.inputNames, feed tensor shapes/values, and raw output range) — confirmed JS layer is correct; root cause of style-invariant output was model-level (Major's `torch.zeros` encoder bug), not a frontend issue. Key lesson: **MAD-only smoke tests are insufficient** — a style-differentiated but mode-collapsed model passes MAD > 0.01 while still producing near-blank output; always add absolute non-blank assertion (`max > threshold`, `std > threshold`).

---

## Cross-Agent Updates

### 2026-03-08: Fresh ONNX Model Ready (Major)

Major has successfully exported and validated a fresh ONNX model from the retrained checkpoint (epoch_0200). 

**Model Details:**
- **Checkpoint:** `models/checkpoints/epoch_0200.pth` (200 epoch full retrain with `use_compile: true`)
- **Export:** `models/v1/generator.onnx` (53.1 MB, INT8 dynamic quantization)
- **Forward pass validation:** ✅ Input (1,1,128,128) → Output (1,1,128,128) float32, values in [-1.0, 1.0]
- **Compression estimate:** ~15.9 MB with Brotli
- **Fix applied:** torch.compile `_orig_mod.` prefix stripping in export script (backward-compatible)

**Implication for Togusa:**
You can now pick up this fresh model for the next inference validation cycle. The 7 debug console.log statements already added to `inferenceWorker.ts` will serve as verification that the retrained model correctly processes style signals (Major's retraining should have fixed the style-invariant output issue).

**Artifacts:**
- Decision: `.squad/decisions.md` (ONNX Export section)
- Orchestration: `.squad/orchestration-log/2026-03-08T105807Z-major.md`
- Session log: `.squad/log/2026-03-08T105807Z-onnx-export.md`
- Commit: `ceab05d` on `dev`

---


## Learnings

### 2026-03-08 (re-run): Full Smoke Test Verification — Fresh ONNX Model (epoch_0200)

- **Task:** Re-run complete test suite against `models/v1/generator.onnx` (53.1 MB INT8, freshly retrained epoch_0200) to confirm style conditioning and baseline test health.
- **Status:** ✅ All tests passing. Style conditioning CONFIRMED working.

**Unit tests (Vitest):** 108/108 ✅
- colorMapping: 5/5
- performance: 9/9
- ModelLoader: 8/8
- BrowserUnsupported: 4/4
- fontPipeline: 15/15
- integration: 8/8
- onnxContract: 38/38
- styleConditioning: 6/6
- FontLoader: 6/6
- fontLoader.styleVariation: 5/5

**Playwright E2E (Chromium only — includes real model):** 20/20 ✅
- Performance: Model Load Time: 3/3
- Performance: Per-Glyph Inference Latency: 3/3
- Performance: Full Font Generation: 4/4
- Performance: Memory Budget: 3/3
- Cross-Browser Smoke: 4/4
- Style Conditioning Real Model: 3/3

**Style conditioning key metrics (real 53 MB model, Chromium WASM):**
- `inputNames`: `style_glyphs`, `char_index` ✅
- Font A (+1.0 all) vs Font B (-1.0 all): Mean Absolute Diff = **0.2811** (threshold > 0.01) ✅
- Determinism: same inputs → bit-identical outputs ✅
- Output shape: [1, 1, 128, 128] ✅
- Output range: within [-1, 1] (±1e-6 for INT8 epsilon) ✅
- `areIdentical: false` ✅ — model responds to style signal

**Total test suite duration:** unit 2.66s + E2E 10.2s (chromium-only)

**Conclusion:** Major's epoch_0200 retrain fixed the style-invariant output bug. The model is now properly conditioned on `style_glyphs` input. Frontend pipeline is production-ready for style-conditional Cyrillic generation.

---

### 2026-03-09: Font Merge Feature — Generated Cyrillic + Uploaded Font

- **Task:** Modify the font assembly pipeline to merge AI-generated Cyrillic glyphs into the uploaded font, producing a complete font with both original Latin glyphs AND new Cyrillic glyphs.
- **Status:** ✅ Implemented and tested. All 111 tests passing.

**What changed:**

1. **FontAssembler.ts:**
   - Updated `assembleFontFromGlyphs()` signature: now takes `(glyphImages, uploadedFont, baseFamilyName)`
   - When `uploadedFont` is provided (ArrayBuffer):
     - Parses the uploaded font with `opentype.parse()`
     - Copies all non-Cyrillic glyphs from uploaded font (skips Unicode range 0x0400-0x04FF)
     - Adds AI-generated Cyrillic glyphs
     - Preserves uploaded font metrics (unitsPerEm, ascender, descender)
     - Sets family name to `{existingFamilyName} Cyrillic`
   - When `uploadedFont` is `null`: creates standalone Cyrillic-only font with default 1000 UPM metrics (backward-compatible fallback)

2. **GlyphVectorizer.ts:**
   - Added `targetUpm` parameter to `vectorizeGlyph()` (default: 1000)
   - Scales all path coordinates proportionally to target UPM
   - Allows Cyrillic glyphs to match uploaded font's coordinate system

3. **App.tsx:**
   - Updated `handleGenerate()` to pass `uploadedFont` (from store) to `assembleFontFromGlyphs()`

4. **Tests:**
   - Updated all test calls to use new 3-parameter signature
   - Added 3 new tests (13-15) for font merging:
     - Test 13: Verifies merged font contains both Latin and Cyrillic glyphs
     - Test 14: Verifies family name ends with " Cyrillic"
     - Test 15: Verifies existing Cyrillic glyphs are replaced (not duplicated)
   - All 111 tests passing (10 test files)

**Key patterns learned:**

- opentype.js API: `font.glyphs.get(i)` to iterate all glyphs, `font.charToGlyphIndex(char)` to look up by character
- Cyrillic Unicode range: 0x0400-0x04FF (must skip when copying from uploaded font to avoid duplicates)
- UPM scaling: advance width scales proportionally: `Math.round(600 * targetUpm / 1000)`
- Font name extraction: `font.names.fontFamily?.en || font.names.fullName?.en || fallback`
- GlyphVectorizer scales coordinates by `targetUpm / 1000` factor to match uploaded font metrics

**Key files:**
- Modified: `src/frontend/src/FontAssembler.ts`
- Modified: `src/frontend/src/GlyphVectorizer.ts`
- Modified: `src/frontend/src/App.tsx`
- Modified: `src/frontend/src/fontPipeline.test.ts` (added 3 merge tests)
- Modified: `src/frontend/src/inference/__tests__/integration.test.ts` (updated signature)


---

### 2026-03-09: Font Merge Feature — Generated Cyrillic + Uploaded Font

**Task:** Modify font assembly to merge AI-generated Cyrillic glyphs into uploaded font.  
**Status:** ✅ Implemented and tested. All 111 tests passing.

**What changed:**

1. **FontAssembler.ts:**
   - New signature: `assembleFontFromGlyphs(glyphImages, uploadedFont, baseFamilyName)`
   - Parses uploaded font, copies non-Cyrillic glyphs, adds AI-generated Cyrillic
   - Skips Unicode range 0x0400-0x04FF to avoid duplicates
   - Preserves uploaded font metrics (unitsPerEm, ascender, descender)
   - Sets family name to `{existingFamilyName} Cyrillic`

2. **GlyphVectorizer.ts:**
   - Added `targetUpm` parameter (default: 1000)
   - Scales path coordinates by `targetUpm / 1000` factor
   - Allows Cyrillic glyphs to match uploaded font's coordinate system

3. **App.tsx:**
   - Passes `uploadedFont` from store to `assembleFontFromGlyphs()`

4. **Tests:**
   - Added 3 new tests (13-15) for font merging
   - Test 13: Verifies both Latin and Cyrillic glyphs present
   - Test 14: Family name ends with " Cyrillic"
   - Test 15: Existing Cyrillic glyphs replaced (not duplicated)

**User Impact:**
- Before: Users received Cyrillic-only .otf, had to manually merge
- After: Complete merged font with original + AI-generated Cyrillic glyphs

**Artifacts:**
- Decision merged to decisions.md
- Orchestration log: 2026-03-08T193433Z-togusa.md
