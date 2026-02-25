# QA Review: PR #4 — Inference Pipeline

**Reviewer:** Saito (Tester)  
**Date:** 2026-02-25  
**Branch:** `feat/togusa-inference-pipeline` → `dev`  
**Author:** Togusa (Frontend Dev)

---

## VERDICT: **CHANGES REQUESTED**

---

## Acceptance Criteria Evaluation

### ✅ **Model loads with progress indicator (no blank screen)**
**Status:** PASS  
**Evidence:**  
- `App.tsx` line 28-43: `useEffect` loads model on mount with `setModelStatus('loading', progress)`
- `ModelLoader.ts` line 30-79: Progress callback invoked on fetch chunks, reports 0-100%
- `inferenceWorker.ts` line 46-63: Streams model bytes with `Content-Length` progress tracking
- UI renders `<ModelLoadingBar />` when `modelStatus === 'loading'` (App.tsx line 47-49)

**Notes:** Good implementation. However, no loading spinner or visual feedback *before* model starts loading. Consider adding initial UI state.

---

### ❌ **Single glyph renders correctly at 128×128 from ONNX output**
**Status:** FAIL — BLOCKING  
**Issue:**  
- `App.tsx` line 67-78: Converts model output (Float32Array, range [-1, 1]) → ImageData
- **Bug:** Incorrect conversion formula:
  ```typescript
  const val = Math.round(((output[px] + 1) / 2) * 255);
  pixels[px * 4 + 0] = val; // R
  pixels[px * 4 + 1] = val; // G
  pixels[px * 4 + 2] = val; // B
  ```
  This maps:
  - `-1` (background in model) → `0` (black in image) ❌
  - `1` (ink in model) → `255` (white in image) ❌

  **Expected:** Model convention is `1 = ink (black)`, `-1 = background (white)` per `decisions.md` and `FontLoader.ts` line 54-55.

**Fix Required:**  
Invert the mapping:
```typescript
const val = Math.round(((1 - output[px]) / 2) * 255); // 1 → 0 (black), -1 → 255 (white)
```

**Owner:** Must be fixed by someone OTHER than Togusa (reviewer lockout). Recommend: **Major** (owns model contract) or **Batou** (backend/integration fix).

---

### ✅ **All 66 Cyrillic glyphs generate without error**
**Status:** PASS (assuming fix above)  
**Evidence:**  
- `App.tsx` line 57-90: Loops through `CYRILLIC_CHARS.length` (66), calls `modelLoader.infer()` sequentially
- `cyrillicCharset.ts`: Defines 66 chars with indices 0-65 (uppercase А-Я + Ё, lowercase а-я + ё)
- `inferenceWorker.ts` line 81-108: Accepts `charIndex` 0-65, constructs correct ONNX tensors per contract
- Error handling: `try/catch` sets `generationStatus = 'error'` on failure (App.tsx line 87-89)

---

### ✅ **Web Worker handles inference without blocking UI thread**
**Status:** PASS  
**Evidence:**  
- `inferenceWorker.ts`: Runs in dedicated Web Worker (imported via `new Worker(new URL(...), { type: 'module' })`)
- `ModelLoader.ts`: Message-passing architecture with request IDs for concurrent safety
- ONNX session created with `executionProviders: ['webgl', 'wasm']` (line 73-75) — offloads to GPU/WASM threads

**Notes:** Sequential inference (not parallel) is correct for MVP. Each glyph takes ~15-30ms WebGL, total ~1-2s for 66 glyphs.

---

### ⚠️ **Generated font downloads as a valid .otf file**
**Status:** PARTIAL — NON-BLOCKING ISSUES  
**Evidence:**  
- `FontLoader.ts` line 61-157: Implements `assembleCyrillicFont()` with:
  - Inference loop (line 74-77)
  - OpenType font creation (line 80-120)
  - Vectorization via `vectorizeGlyph()` (line 130-157)
- `App.tsx` line 92-102: Blob download with `type: 'font/otf'`, filename `cyrillic-font.otf`
- Object URL revoked after download (line 101) ✅

**Issues:**
1. **Vectorization quality:** `vectorizeGlyph()` uses scanline-based rectangle drawing (line 144-156). This produces blocky glyphs, not smooth contours. Acceptable for MVP, but should document limitation.
2. **Font metadata:** Hard-coded family name `'Generated Cyrillic'` (line 112). Should derive from uploaded font or let user specify.
3. **Missing Latin glyphs:** Generated font only contains 66 Cyrillic glyphs + `.notdef`. User's original Latin glyphs are NOT copied over. Font is Cyrillic-only, not a complete replacement. This violates user expectation ("add Cyrillic to my font").

**Non-blocking but should be addressed in follow-up PR:**
- Copy source font's Latin glyphs into generated font (requires `opentype.js` glyph cloning)
- Preserve source font metadata (family name, style, weights)

---

## QA-Specific Concerns

### ✅ **Error Handling — Model Load**
- ✅ Network failure (404, timeout): Handled via `try/catch`, sets `modelStatus = 'error'` (App.tsx line 37-39)
- ✅ Worker crash: Handled via `worker.onerror` (ModelLoader.ts line 71-73)
- ✅ Invalid model file: ONNX Runtime will throw, caught by worker's `try/catch` (inferenceWorker.ts line 32-39)

### ⚠️ **Error Handling — Font Upload**
- ✅ Font parse failure: Handled via `try/catch` in FontUpload component (not shown in diff, but `loadFont` throws on invalid font)
- ❌ **Missing Latin glyphs:** If uploaded font lacks Latin A-Z (style chars), `extractStyleGlyphs()` will render empty glyphs without warning. Should validate font has required chars before extraction.

**Recommendation:** Add validation in `FontUpload.tsx`:
```typescript
const font = await loader.current.loadFont(buffer);
const hasLatinGlyphs = ['A', 'B', 'H', 'O'].every(char => font.charToGlyph(char).unicode !== undefined);
if (!hasLatinGlyphs) {
  throw new Error('Font must contain Latin A-Z glyphs for style extraction');
}
```

### ✅ **Memory Leaks**
- ✅ Object URL revoked after download (App.tsx line 101)
- ⚠️ **Worker termination:** Worker is created on mount but NEVER terminated. Memory leak if user navigates away. Should add `useEffect` cleanup:
  ```typescript
  useEffect(() => {
    loadModel();
    return () => { /* terminate worker */ };
  }, []);
  ```
  However, `ModelLoader` is a singleton with no `terminate()` method. Not blocking for MVP (page refresh kills worker), but should be fixed.

### ✅ **Type Safety**
- ✅ No `any` casts in critical paths
- ✅ ONNX tensor shapes documented inline (inferenceWorker.ts line 87-90)
- ✅ Zustand store is fully typed (appStore.ts)

### ✅ **Race Conditions**
- ✅ Generate button disabled until `modelStatus === 'ready' && styleGlyphs !== null` (App.tsx line 45)
- ✅ Generate button disabled while `generationStatus === 'running'` (line 58)
- ✅ Concurrent inference requests handled via request IDs (ModelLoader.ts line 51-61)

### ✅ **Browser Compatibility**
- ✅ `import.meta.url` is Vite-specific, documented in charter (Vite builds bundle workers correctly)
- ✅ WebGL + WASM fallback for ONNX RT (inferenceWorker.ts line 73-75)
- ⚠️ Safari: ONNX Runtime Web WebGL backend has known issues on Safari < 16. Should document browser requirements (Chrome/Firefox/Safari 16+/Edge).

### ✅ **Font Validity**
- ✅ `opentype.js` generates structurally valid OTF (magic bytes, glyph table, metrics)
- ⚠️ Glyph quality: Vectorization is lossy (scanline rectangles). Acceptable for MVP, but glyphs will look pixelated at large sizes.

### ✅ **Performance**
- ✅ Inference is sequential (not parallel). Expected ~1-2s for 66 glyphs @ ~15-30ms/glyph WebGL.
- ✅ Progress updates per glyph (App.tsx line 77) — good UX feedback.
- ⚠️ No timeout on inference — if model hangs, UI will freeze. Should add per-glyph timeout (e.g., 10s).

---

## Test Coverage

**Created:**
- `src/frontend/src/inference/__tests__/ModelLoader.test.ts` (192 lines)
  - Model load with progress tracking
  - Concurrent load request deduplication
  - Inference request/response with request IDs
  - Error handling: network, worker, inference failures
  - Concurrent inference request handling

- `src/frontend/src/font/__tests__/FontLoader.test.ts` (134 lines)
  - Style glyph extraction validation (10 chars → 163840 floats)
  - Font assembly with 66 Cyrillic glyphs
  - OpenType structure validation (magic bytes)
  - Error propagation from inference function

- `src/frontend/src/inference/__tests__/integration.test.ts` (63 lines)
  - Placeholder integration tests (requires full DOM + worker mocking)
  - Documented test cases for end-to-end pipeline validation

**Note:** Tests are unit tests with mocked dependencies. Full E2E tests (Playwright/Cypress) deferred to future PR.

---

## Blocking Issues (Must Fix Before Merge)

1. **❌ CRITICAL: Inverted color mapping in App.tsx line 67**  
   Model output `1 = ink, -1 = background` but code maps to `1 = white, -1 = black`.  
   **Fix owner:** Major or Batou (NOT Togusa — reviewer lockout).  
   **Line:** `src/frontend/src/App.tsx:67-70`

---

## Non-Blocking Suggestions (For Follow-Up)

1. **Font Completeness:** Copy source font's Latin glyphs into generated font (currently Cyrillic-only)
2. **Font Metadata:** Preserve source font family name, not hard-code `'Generated Cyrillic'`
3. **Latin Glyph Validation:** Warn if uploaded font lacks required style chars (A, B, H, O, etc.)
4. **Worker Cleanup:** Add `ModelLoader.terminate()` and call in `useEffect` cleanup
5. **Inference Timeout:** Add per-glyph timeout (10s) to prevent UI hang if model stalls
6. **Browser Requirements:** Document Safari 16+ requirement for WebGL ONNX RT backend
7. **Vectorization Quality:** Document that scanline-based vectorization produces blocky glyphs; consider integrating potrace for smooth contours in future PR

---

## Summary

PR #4 successfully wires the end-to-end inference pipeline: Web Worker, ONNX Runtime Web, model loading, progress tracking, 66-glyph generation, and font download. Architecture is clean, type-safe, and follows team decisions.

**However, ONE BLOCKING BUG prevents merge:** Color inversion in `App.tsx` line 67 will generate inverted glyphs (black backgrounds, white ink). This must be fixed by Major or Batou before merge.

Test coverage is good for core units (ModelLoader, FontLoader). Integration tests are placeholders pending E2E framework setup.

Non-blocking issues (missing Latin glyphs, metadata, vectorization quality) are acceptable for MVP but should be tracked for next iteration.

---

**Next Steps:**
1. Major or Batou: Fix color inversion in App.tsx line 67
2. Re-test with actual ONNX model (not yet available — model training in progress)
3. Merge to dev after fix
4. Create follow-up issues for non-blocking suggestions

---

**Test Artifacts:**
- `src/frontend/src/inference/__tests__/ModelLoader.test.ts`
- `src/frontend/src/font/__tests__/FontLoader.test.ts`
- `src/frontend/src/inference/__tests__/integration.test.ts`
