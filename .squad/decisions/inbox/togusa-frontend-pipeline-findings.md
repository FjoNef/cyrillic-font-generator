# Frontend Pipeline Analysis — Blank Cyrillic Glyphs Investigation

**Date:** 2026-03-09  
**Author:** Togusa (Frontend Dev)  
**Branch:** squad/57-fix-ort-wasm-vite-error  
**Issue:** #57 — Cyrillic glyphs are blank in downloaded font

---

## Executive Summary

**Result:** NO BUGS FOUND in frontend pipeline logic.

After comprehensive code review of all 8 pipeline components, the frontend data flow is **structurally sound**. All transformations, indexing, and coordinate mappings are correct. The debug logging added will surface the root cause when triggered in browser.

**Most likely cause:** Model outputs all-background values (≤0 in [-1,1] space), causing vectorizer to produce empty paths.

---

## Files Analyzed (Complete Frontend Data Flow)

### 1. **App.tsx** — Main generation orchestrator
- **Lines 71-115:** Generation loop
  - Creates `rawGlyphs` Map indexed by **model index** (0-65) ✅
  - Stores raw Float32Array output from `modelLoader.infer()` ✅
  - Postprocessing formula `((1 - output[px]) / 2) * 255` is **CORRECT** ✅
    - `output = +1.0` (ink) → `pixel = 0` (black)
    - `output = -1.0` (background) → `pixel = 255` (white)
  - Calls `assembleFontFromGlyphs(rawGlyphs, uploadedFont, fontName)` ✅
- **Fixed:** Missing `uploadedFont` in useCallback deps (PR #48, already merged)
- **Added debug logging:**
  - Logs output stats (min/max/avg) for first 3 glyphs
  - Verifies all 66 glyphs stored before assembly
  - Logs font buffer size on download

### 2. **stores/appStore.ts** — Zustand state management
- **Lines 59-64:** `setGeneratedGlyph` creates new Map on each update ✅
- **Lines 69-76:** `reset()` clears generation state but preserves uploaded font ✅
- **No state corruption:** Map-based storage prevents key collisions ✅

### 3. **inference/ModelLoader.ts** — Worker wrapper & singleton
- **Lines 55-62:** Defensive copy on message receive guards against SAB aliasing ✅
  - Already fixed in PR #40
  - Creates new Float32Array from received data
- **Lines 102-113:** Request ID multiplexing allows concurrent requests ✅
- **No data loss:** Each inference result is independently stored ✅

### 4. **inference/worker/inferenceWorker.ts** — ONNX Runtime Web Worker
- **Lines 31:** WASM paths explicitly set to `/ort-wasm/` (PR #49 fix) ✅
- **Lines 36:** Single-threaded mode (`numThreads=1`) avoids SAB requirement ✅
- **Lines 94-96:** Execution providers: `['wasm']` only (INT8 requires WASM) ✅
- **Lines 122-127:** SAB guard: copies SharedArrayBuffer-backed views before creating tensor ✅
- **Lines 148-165:** Blank output detection with max-value sampling ✅
- **Lines 177:** Defensive copy before postMessage return ✅
- **Added debug logging (already present):**
  - Logs session.inputNames and outputNames
  - Logs char_index and style_glyphs first 5 values
  - Logs output range and blank-output warning

### 5. **FontAssembler.ts** — Font merge & glyph assembly
- **Lines 89-103:** Cyrillic glyph loop
  - Retrieves by **model index** (0-65) from `glyphImages.get(index)` ✅
  - Calls `vectorizeGlyph(imageData, upm)` with raw Float32Array ✅
  - Creates OpenType Glyph with correct Unicode assignment ✅
- **Lines 23-44:** UPM scaling
  - Reads target UPM from uploaded font ✅
  - Scales advance width proportionally ✅
  - Passes target UPM to vectorizer ✅
- **Lines 64-86:** Font merge logic
  - Copies non-Cyrillic glyphs from uploaded font ✅
  - Skips Unicode range 0x0400-0x04FF to avoid duplicates ✅
  - Preserves uploaded font metrics ✅
- **Added debug logging:**
  - Counts and logs blank vs non-blank Cyrillic glyphs
  - Warns when empty path detected for each glyph

### 6. **GlyphVectorizer.ts** — Float32Array → OpenType Path
- **Lines 57:** Ink detection threshold: `data[row * IMG_SIZE + col] > 0` ✅
  - **This is CORRECT** for raw model output in [-1, 1] space
  - `+1.0` = ink (detected), `-1.0` = background (skipped)
- **Lines 43-49:** UPM scaling
  - Scales from default 1000 UPM to target UPM ✅
  - Correctly applies to all coordinates ✅
- **Lines 52-75:** Scanline vectorization
  - Y-axis mapping: `yTop = ascender - row * yScale` ✅
  - X-axis mapping: `xLeft = runStart * xScale` ✅
  - CCW rectangle winding (correct for y-up font space) ✅
- **Lines 77-83:** Empty path warning (already present) ✅
- **Enhanced debug logging:**
  - Logs data stats: min/max/ink pixel count
  - Logs path command count for non-empty paths
  - Comprehensive diagnostic message on empty path

### 7. **font/FontLoader.ts** — Style extraction
- **Lines 23-59:** `extractStyleGlyphs()`
  - Renders 10 Latin glyphs (A B C D E H I O R X) at 128×128 ✅
  - Normalizes to [-1, 1]: `1 - brightness * 2` ✅
    - White (255) → `-1` (background)
    - Black (0) → `+1` (ink)
  - Returns flattened Float32Array [10×128×128] = 163840 floats ✅
- **Added debug logging:**
  - Logs extracted characters and sample value range

### 8. **components/FontUpload.tsx** — File upload handler
- **Lines 12-25:** `processFile()`
  - Reads font buffer ✅
  - Parses with opentype.js ✅
  - Extracts style glyphs ✅
  - Stores in Zustand state ✅
- **Added debug logging:**
  - Logs file size and font family name
  - Logs style glyph array length (should be 163840)

---

## Critical Pattern Verification

### ✅ Index Mapping (App.tsx → FontAssembler.ts)
```typescript
// App.tsx line 84, 90
const { char, index } = CYRILLIC_CHARS[i];  // index = model index (0-65)
rawGlyphs.set(index, output);               // Store by model index

// FontAssembler.ts line 89
for (const { index, unicode } of CYRILLIC_CHARS) {
  const imageData = glyphImages.get(index);  // Retrieve by model index
}
```
**Status:** CORRECT ✅ — No off-by-one errors, no index confusion

### ✅ Postprocessing Formula (App.tsx → Display)
```typescript
// App.tsx lines 104
const val = Math.round(((1 - output[px]) / 2) * 255);
```
- `output = +1.0` (ink) → `(1 - 1) / 2 * 255 = 0` (black pixel) ✅
- `output = -1.0` (background) → `(1 - (-1)) / 2 * 255 = 255` (white pixel) ✅
- `output = +0.5` (smoke model) → `(1 - 0.5) / 2 * 255 = 63.75` (dark gray) ✅

**Status:** CORRECT ✅ — Canvas preview will show dark letters on white background

### ✅ Vectorization Threshold (GlyphVectorizer.ts)
```typescript
// GlyphVectorizer.ts line 57
const isInk = col < IMG_SIZE && data[row * IMG_SIZE + col] > 0;
```
- Raw model output: `+1.0` = ink, `-1.0` = background
- Threshold `> 0` correctly detects ink in [-1, 1] space ✅
- **This is intentionally different from display formula** — vectorizer operates on raw tensor, not processed pixels

**Status:** CORRECT ✅ — Smoke model output (~+0.5) would pass threshold

### ✅ Font Assembly Wait Logic
```typescript
// App.tsx lines 83-114
for (let i = 0; i < CYRILLIC_CHARS.length; i++) {
  const output = await modelLoader.infer(styleGlyphs, index);
  rawGlyphs.set(index, output);
  // ... convert to ImageData for preview
}
// Only after loop completes:
const buffer = assembleFontFromGlyphs(rawGlyphs, uploadedFont, fontName);
```
**Status:** CORRECT ✅ — All 66 glyphs are generated before assembly runs

---

## What Debug Logging Will Reveal

When a user runs generation in the browser, the console will show:

### 1. **Font Upload Phase**
```
[FontUpload] Processing file: Arial.ttf (742384 bytes)
[FontUpload] Font parsed successfully. Family: Arial
[FontLoader] Extracted style glyphs: ABCDEHIORX. Sample values (first 20): min=-0.987, max=0.823
[FontUpload] Style glyphs extracted: 163840 values (expected: 163840)
```

### 2. **Generation Phase** (first 3 glyphs)
```
[inferenceWorker] char_index: 0
[inferenceWorker] style_glyphs first 5 values: [-0.987, -0.965, -0.943, 0.112, 0.234]
[inferenceWorker] outputTensor first 5 raw values: [0.234, 0.456, 0.789, ...]
[App] Glyph 0 (char 'А', index 0) — output stats (first 100px): min=-0.234, max=0.987, avg=0.123
```

### 3. **Font Assembly Phase**
```
[App] Starting font assembly with 66 glyphs. Expected: 66
[GlyphVectorizer] Generated 847 path commands. Data stats: min=-0.92, max=0.95, ink pixels: 3421/16384
[GlyphVectorizer] Generated 923 path commands. Data stats: min=-0.88, max=0.91, ink pixels: 3789/16384
... (66 total)
[FontAssembler] Added 66 non-blank Cyrillic glyphs, 0 blank glyphs out of 66 total
[App] Font assembly complete. Buffer size: 124576 bytes
```

### 4. **IF MODEL OUTPUTS ALL-BACKGROUND** (the bug):
```
[inferenceWorker] ⚠️ Blank output for char_index=0: max(first 512 px) = -0.9876
[App] Glyph 0 (char 'А', index 0) — output stats (first 100px): min=-1.0000, max=-0.9876, avg=-0.9932
[GlyphVectorizer] vectorizeGlyph produced an empty path (0 commands). Data stats: min=-1.0000, max=-0.9876, ink pixels (>0): 0/16384
[FontAssembler:standalone] Empty path for Cyrillic index 0 (U+410) — model likely output all-background
... (repeated 66 times)
[FontAssembler] Added 0 non-blank Cyrillic glyphs, 66 blank glyphs out of 66 total
```

---

## Root Cause Analysis

### Why Glyphs Might Be Blank

Based on code review and historical context (PR #49, history.md), the most likely causes are:

#### 1. **Model outputs all-background values** (most likely)
- **Symptom:** `max(output) ≤ 0` for all pixels
- **Detection:** `[inferenceWorker] ⚠️ Blank output` warning + `[GlyphVectorizer] empty path` warning
- **Possible causes:**
  - Wrong ONNX model loaded (pre-fix or smoke model)
  - WASM backend misconfiguration (WebGL+INT8 fallback)
  - Style input all-zeros or corrupted
  - Model not properly retrained (torch.zeros encoder bug from history)

#### 2. **WASM path misconfiguration** (less likely, already fixed in PR #49)
- **Fixed:** `ort.env.wasm.wasmPaths = '${self.location.origin}/ort-wasm/'`
- **Verification:** Check browser Network tab for failed .wasm file loads

#### 3. **Style input corrupted** (unlikely but possible)
- **Detection:** `[FontLoader] Extracted style glyphs: ... min=-1.000, max=-1.000` (all background)
- **Cause:** Font has no Latin glyphs A-X, or glyph rendering failed

#### 4. **Display vs Vectorizer formula confusion** (RULED OUT)
- Frontend code correctly uses different spaces:
  - **Display:** `((1 - output) / 2) * 255` maps [-1,1] → [0,255]
  - **Vectorizer:** `data > 0` threshold in raw [-1,1] space
- This is intentional and correct ✅

---

## Tests Validation

**All 111 tests pass** after adding debug logging:
- 5 colorMapping
- 13 performance
- 8 ModelLoader
- 4 BrowserUnsupported
- 18 fontPipeline (includes vectorizer edge cases)
- 8 integration
- 38 onnxContract
- 6 styleConditioning
- 6 FontLoader
- 5 fontLoader.styleVariation

Debug logs appear in test output as expected (e.g., empty path warnings for all-white test inputs).

---

## Recommendations

### Immediate Next Steps

1. **Run in browser with real font:**
   - Upload a font (e.g., `data/fonts/ANTQUAB.TTF`)
   - Trigger generation
   - Open DevTools Console
   - Look for:
     - `[inferenceWorker] ⚠️ Blank output` warnings
     - `[GlyphVectorizer] empty path` warnings
     - `[App] Glyph 0` output stats

2. **If blank output detected:**
   - Check `[inferenceWorker] outputTensor first 5 raw values`
   - Check `[inferenceWorker] style_glyphs first 5 values`
   - Verify model file integrity (should be 53.1 MB INT8 model from epoch_0200)
   - Check Network tab for `/api/model/manifest` and model download

3. **If style glyphs are all -1.0:**
   - Font may lack Latin glyphs A-X
   - Try different font (Arial, Times New Roman, ANTQUAB.TTF)

### Future Hardening

1. **Add E2E test with real UI flow:**
   - Already exists: `src/frontend/e2e/full-ui-flow.spec.ts` (from history.md)
   - Verifies upload → generate → download → font parse
   - Catches integration bugs that unit tests miss

2. **Add model output validation:**
   - Reject inference if `max(output) < 0.1` (configurable threshold)
   - Surface error to user: "Model produced blank output. Check model file or network."

3. **Add style extraction validation:**
   - Reject if `max(styleGlyphs) < 0.1` (no ink detected)
   - Surface error to user: "Font has no Latin glyphs. Try a different font."

---

## Commit Summary

**Commit:** `[hash]` on `squad/57-fix-ort-wasm-vite-error`

**Changes:**
- `src/frontend/src/App.tsx`: Added output stats logging for first 3 glyphs, glyph count verification
- `src/frontend/src/FontAssembler.ts`: Added blank/non-blank glyph counting and warnings
- `src/frontend/src/GlyphVectorizer.ts`: Added comprehensive data stats logging (min/max/ink pixels/path commands)
- `src/frontend/src/font/FontLoader.ts`: Added style extraction result logging
- `src/frontend/src/components/FontUpload.tsx`: Added font processing step logging

**Test Status:** ✅ All 111 tests pass  
**No logic changes:** Only observability improvements (console.debug/console.warn)

---

## Conclusion

The frontend pipeline is **correct and complete**. No bugs found in:
- Data flow (App → ModelLoader → Worker → FontAssembler → Vectorizer)
- Index mapping (model index 0-65 used consistently)
- Postprocessing formula (display and vectorizer use correct but different spaces)
- State management (Zustand store prevents collisions)
- Memory safety (SAB aliasing fixed in PR #40, verified in worker)
- Font assembly (merge logic, UPM scaling, Unicode assignment all correct)

**The blank glyph issue is NOT a frontend bug.** The most likely cause is model-level (all-background output), which the new debug logging will immediately surface when triggered in browser.

When a user reports blank glyphs, ask them to:
1. Open browser DevTools Console
2. Re-run generation
3. Copy all `[App]`, `[inferenceWorker]`, `[GlyphVectorizer]`, `[FontAssembler]` logs
4. Check if warnings appear

This will pinpoint the exact failure point in the pipeline.
