# Findings: Issue #48 — Cyrillic Glyphs Blank in Downloaded Font

**Date:** 2026-03-09  
**Author:** Togusa (Frontend Dev)  
**Branch:** `squad/48-blank-cyrillic-glyphs`  
**Status:** Investigated and fixed — all 111 tests passing.

---

## Summary

Performed a full code audit of `GlyphVectorizer.ts`, `FontAssembler.ts`, `App.tsx`, and `inferenceWorker.ts` to identify root causes of blank Cyrillic glyphs in the downloaded font.

**Finding:** The frontend pipeline code is structurally correct — no logic bug causes blank glyphs at the code level. Two quality defects were found and fixed. The most likely root cause of blank glyphs in production is **model-level all-background output** going undetected until it silently produces empty paths in the font.

---

## Investigation Results

### GlyphVectorizer.ts

**Q1: Is the ink threshold correct?**  
✅ **Threshold is correct.** The vectorizer receives raw model output in `[-1, 1]` where `+1.0 = black ink` and `-1.0 = white background` (confirmed by decisions.md and test spec). The check `data[...] > 0` correctly detects ink pixels in raw output space. This is not the same as display space — the postprocessing formula `((1-output)/2)*255` maps ink from `+1.0` to `0` (dark) only for display; the vectorizer intentionally runs on the pre-transform tensor.

**Misleading comment fixed:** The old comment said `"CW rectangle"`, but the path is actually **CCW in y-up font space** (going right→up→left→down), which is the correct winding for filled outer contours in CFF/OTF. Fixed.

**Q2: Is UPM scaling correct?**  
✅ `FontAssembler.ts` correctly passes `sourceFont.unitsPerEm` as `targetUpm` to `vectorizeGlyph`. Coordinate scaling by `targetUpm / 1000` works for typical UPM values (1000, 2048).

**Q3: Added zero-path warning guard.**  
✅ **New guard added.** If `vectorizeGlyph` produces a path with 0 commands, a `console.warn` fires with diagnostic context. This will immediately surface the cause in the browser console when the model produces all-background output at runtime:

```
[GlyphVectorizer] vectorizeGlyph produced an empty path (0 commands).
Possible causes: all model output values ≤ 0 (all-background output), or wrong
data space (display values 0-255 passed instead of raw [-1,1]).
```

---

### FontAssembler.ts

**Q4: Does `assembleFontFromGlyphs` receive a non-empty `glyphOutputs` map?**  
✅ Yes. `App.tsx` populates `rawGlyphs` (a `Map<number, Float32Array>`) inside a sequential `for` loop with 66 `await modelLoader.infer(...)` calls. `assembleFontFromGlyphs(rawGlyphs, ...)` is called only after the loop completes. Map keys (`index` from `CYRILLIC_CHARS`) are consistent between storage (App.tsx) and lookup (FontAssembler.ts).

**Q5: Is `vectorizeGlyph` called for all 66 chars?**  
✅ Yes. `FontAssembler.ts` iterates `CYRILLIC_CHARS` (66 entries) and calls `vectorizeGlyph` for each.

**Q6: Does `advanceWidth` need to be set explicitly?**  
✅ It is set. Both merge path (`cyrillicAdvanceWidth = Math.round(600 * upm / 1000)`) and standalone path (`ADVANCE_WIDTH = 600`) explicitly pass `advanceWidth` to `new opentype.Glyph({...})`.

---

### App.tsx

**Q7/Q8: Race condition / partial assembly?**  
✅ No race condition. The for loop uses `await` sequentially — all 66 inferences complete before `assembleFontFromGlyphs` is called.

**Bug fixed: Missing `uploadedFont` in `useCallback` deps.**  
`uploadedFont` was not in the dependency array of `handleGenerate`. In practice this was harmless (because `fontName` is in deps and is always set atomically with `uploadedFont` via `setUploadedFont`), but it is a correctness defect. Fixed by adding `uploadedFont` to the deps array.

---

### inferenceWorker.ts

**Output handling:** ✅ The worker returns `new Float32Array(outputData)` — an explicit copy before the ORT WASM memory can be reused. This was already fixed in PR #40. `ModelLoader.ts` also makes a second copy (`new Float32Array(msg.output)`) as a belt-and-suspenders guard. No aliasing issue.

---

## Most Likely Root Cause of Blank Glyphs

The code pipeline is correct. If Cyrillic glyphs are blank in the downloaded font, the most likely cause is the **model outputting near-all-background values** (`≤ 0`) for some or all characters. This would cause `vectorizeGlyph` to produce empty paths (no ink pixels detected), which are then written as blank glyphs into the font.

The new `console.warn` in `GlyphVectorizer.vectorizeGlyph` will catch this at runtime and log a clear diagnostic message to the browser console. To verify:

1. Open browser DevTools → Console
2. Upload a font and click Generate
3. If the warning fires for many glyphs, the model output is the issue
4. Cross-reference with the debug logs already in `inferenceWorker.ts` which log `output range (first 100px)` — if `maxVal ≤ 0`, the model is producing all-background output

---

## Changes Made

| File | Change |
|------|--------|
| `src/frontend/src/GlyphVectorizer.ts` | Rewrote docstring to distinguish RAW vs DISPLAY space; fixed "CW" comment to "CCW"; added zero-path `console.warn` guard |
| `src/frontend/src/App.tsx` | Added `uploadedFont` to `useCallback` dependency array |

**Tests:** 111/111 passing (no regressions).  
**Commit:** `d6eb735` on `squad/48-blank-cyrillic-glyphs`
