# UI Pipeline Audit — Blank Glyph Investigation (squad/57-fix-ort-wasm-vite-error)

**Date:** 2026-03-09  
**Agent:** Togusa (Frontend Dev)  
**Context:** Comprehensive audit of React UI data pipeline on branch `squad/57-fix-ort-wasm-vite-error` to find potential bugs causing blank Cyrillic glyph rendering.

---

## Audit Outcome: ✅ NO UI PIPELINE BUGS FOUND

After exhaustively auditing every file in the UI data pipeline from font upload through inference to font assembly, **zero bugs were found that would cause blank Cyrillic output**. The code is structurally correct.

---

## Files Audited

### 1. **FontLoader.ts** (lines 1-173) — Style Glyph Extraction ✅

**Checked:**
- Normalization formula (lines 53-54): `brightness = pixel / 255; result = 1 - brightness * 2`
  - Black glyph (0) → brightness = 0 → result = **+1.0** ✅ (ink, correct)
  - White bg (255) → brightness = 1 → result = **-1.0** ✅ (background, correct)
  
**Training data comparison:**
- Training (dataset.py line 89): Renders white glyph on black background, then normalizes with `(x - 0.5) / 0.5 = 2x - 1`
  - White glyph (255) → 1.0 → 2×1 - 1 = **+1.0** (ink)
  - Black bg (0) → 0.0 → 2×0 - 1 = **-1.0** (background)
  
- Frontend: Renders black glyph on white background, then normalizes with `1 - (pixel/255) * 2`
  - Black glyph (0) → 0 → 1 - 0 = **+1.0** (ink)
  - White bg (255) → 1 → 1 - 2 = **-1.0** (background)
  
**Conclusion:** Normalization is mathematically equivalent. The frontend inverts the rendering color scheme (black-on-white instead of white-on-black) and applies a matching inverted normalization to achieve the same `[-1, 1]` tensor values.

**Shape:** Returns `Float32Array` of shape `[10, 1, 128, 128]` = 163,840 floats. Batch dimension `[1, ...]` is prepended later by `ModelLoader` and `inferenceWorker` when creating the ONNX tensor.

**No bugs found.**

---

### 2. **App.tsx** (lines 1-237) — Main Orchestration ✅

**Checked:**
- Style glyphs retrieved from Zustand store (line 20): `styleGlyphs`
- Inference loop (lines 83-114): Sequential, awaits each inference before proceeding
- Raw glyph storage (line 99): `rawGlyphs.set(index, output)` — no race conditions
- Canvas rendering (lines 102-112): Formula `((1 - output[px]) / 2) * 255` correctly maps:
  - `+1.0` (ink) → `((1 - 1) / 2) * 255 = 0` (black pixel)
  - `-1.0` (background) → `((1 - (-1)) / 2) * 255 = 255` (white pixel)
  
**Added in this branch:**
- Debug logging (lines 89-96, 117-127, 141-143): Logs output stats, missing glyphs, buffer sizes

**No bugs found.** Data flow is correct.

---

### 3. **ModelLoader.ts** (lines 1-125) — Worker Communication ✅

**Checked:**
- postMessage (lines 107-112): Does NOT use transferable list `[styleGlyphs.buffer]` — this is correct, avoids buffer detachment
- Worker response handling (lines 52-62): Correctly uses `requestId` to match responses to requests
- Defensive copy (line 60): `new Float32Array(msg.output)` — prevents SharedArrayBuffer aliasing
  
**No bugs found.** Communication protocol is correct.

---

### 4. **inferenceWorker.ts** (lines 1-179) — WASM Inference ✅

**Checked:**
- Tensor creation (line 129): `styleTensor = new ort.Tensor('float32', safeStyleGlyphs, [1, 10, 1, 128, 128])` — correct shape
- SharedArrayBuffer guard (line 122): `typeof SharedArrayBuffer !== 'undefined' && styleGlyphs.buffer instanceof SharedArrayBuffer` — correct
- Output copy (line 177): `return new Float32Array(outputData)` — prevents WASM memory aliasing
  
**No bugs found.** Worker correctly creates a copy before returning.

---

### 5. **FontAssembler.ts** (lines 1-208) — Font Assembly ✅

**Checked:**
- Glyph mapping (lines 92-114): Correctly maps model index (0-65) → Unicode codepoint via `CYRILLIC_CHARS[index].unicode`
- Vectorization (line 95): Calls `vectorizeGlyph(imageData, upm)` with correct target UPM scaling
- Font creation (lines 110-127): Correctly constructs opentype.Font with proper metadata
  
**Added in this branch:**
- Debug logging (lines 89-117, 160-189): Warns about missing image data or empty paths, logs glyph counts

**No bugs found.** Assembly logic is correct.

---

### 6. **GlyphVectorizer.ts** (lines 1-109) — Path Vectorization ✅

**Checked:**
- Ink threshold (line 72): `data[row * IMG_SIZE + col] > 0` — CORRECT for raw `[-1, 1]` model output
  - Model outputs: `+1.0` = ink (foreground), `-1.0` = background
  - Threshold `> 0` correctly detects ink pixels
  
**Critical distinction:** The vectorizer operates on **raw model output** in `[-1, 1]` space, NOT on display pixel values `[0, 255]`. The display postprocessing formula `((1 - output) / 2) * 255` is only applied in `App.tsx` for canvas rendering — the vectorizer never sees it.

**Added in this branch:**
- Debug logging (lines 51-105): Logs min/max/ink pixel count, warns if zero commands generated

**No bugs found.** Threshold is correct for the data space it operates on.

---

### 7. **appStore.ts** (lines 1-77) — Zustand State Management ✅

**Checked:**
- State mutations (lines 41-67): All immutable updates, no direct mutations
- Map cloning (lines 60-62): `new Map(state.generatedGlyphs)` prevents reference bugs
  
**No bugs found.** State management is correct.

---

### 8. **FontUpload.tsx** (lines 1-67) — Font Upload Component ✅

**Checked:**
- Style glyph extraction (line 19): `loader.current.extractStyleGlyphs(font)` — called immediately after font parsing
- Store update (line 22): `setStyleGlyphs(styleGlyphs)` — stores result in Zustand
  
**Added in this branch:**
- Debug logging (lines 13, 17, 23): Logs file name/size, font family, style glyph array length

**No bugs found.** Upload flow is correct.

---

## Historical Context from Git Log

### Issue #48 (Previously Fixed — Commit bd7eb91)

**Root cause (TRAINING BUG, not UI bug):**
- `UNetGenerator.forward()` used `torch.zeros` as encoder input instead of `style_glyphs[:, 0]`
- All six skip connections were constant-zero
- Model minimized L1 loss by predicting all-background (`-1.0`)
- Postprocessing: `((1 - (-1)) / 2) * 255 = 255` → all-white pixels → blank glyphs

**Fix:** Retrained model (epoch_0200) with correct style conditioning  
**Status:** Fixed in PR #48, merged to `dev`

### Current Branch (squad/57-fix-ort-wasm-vite-error)

**Issue:** ORT 1.20 dynamic import of `ort-wasm-simd-threaded.jsep.mjs` (for WebGPU JSEP feature detection) is intercepted by Vite 5, causing build errors.

**Fix (commit c7a8ce8):**
- Added `ort-wasm-runtime-external` Vite plugin to mark `/ort-wasm/*.mjs` as external
- Changed `wasmPaths` to use `self.location.origin` instead of relative path to prevent Vite static-analysis resolution

**Not a UI pipeline bug.** This is a build-time/runtime WASM loading issue.

---

## Debugging Additions in This Branch

All changes on this branch are **debug logging only** — no logic changes:

1. **App.tsx** (lines 89-96): Log output stats (min/max/avg) for first 3 glyphs
2. **App.tsx** (lines 117-127): Log missing glyph indices, font assembly start/completion
3. **App.tsx** (lines 141-143): Log download trigger
4. **FontAssembler.ts** (lines 89-117, 160-189): Warn about missing/blank glyphs, log counts
5. **GlyphVectorizer.ts** (lines 51-105): Log min/max/ink pixel count, warn if zero commands
6. **FontUpload.tsx** (lines 13, 17, 23): Log file upload, font parse, style extraction

These logs help diagnose blank glyph issues at runtime but do NOT fix any bugs — because no bugs exist in the UI code.

---

## Conclusion

The **UI pipeline is clean**. All data flow, normalization, tensor creation, worker communication, and font assembly logic is correct. 

### If Blank Glyphs Still Occur, Check:

1. **Backend/Model:** Is the correct model being served? Is the model file corrupted? Is it the pre-fix model (with `torch.zeros` bug)?
2. **ORT WASM Loading:** Is the Vite plugin fix (commit c7a8ce8) working? Are WASM files loading correctly? Check browser console for 404 errors or WASM instantiation failures.
3. **Browser Execution Provider:** Is WASM backend being used? If WebGL is selected, INT8 quantized models will produce all-background output (QLinear ops not supported).

The frontend is production-ready. Any issues are at the **inference infrastructure layer** (WASM loading, execution provider selection) or **model layer** (wrong model version, training bug), not in the React UI code.

---

## Recommendations

1. ✅ Keep the debug logging from this branch — it provides valuable runtime diagnostics
2. ✅ Add E2E test that validates actual glyph ink presence (not just shape/range), similar to `style-conditioning-real.spec.ts` but for the full UI flow
3. ✅ Document the normalization equivalence (black-on-white frontend vs white-on-black training) in inline comments or decisions.md to prevent future confusion

---

**Agent:** Togusa  
**Branch:** squad/57-fix-ort-wasm-vite-error  
**Files Audited:** 8 files (FontLoader.ts, App.tsx, ModelLoader.ts, inferenceWorker.ts, FontAssembler.ts, GlyphVectorizer.ts, appStore.ts, FontUpload.tsx)  
**Bugs Found:** 0  
**Verdict:** UI pipeline is correct. Blank glyphs (if occurring) are caused by backend/model/WASM issues, not UI code.
