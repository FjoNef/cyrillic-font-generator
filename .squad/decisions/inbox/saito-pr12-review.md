# Saito QA Verdict — PR #12

**Date:** 2026-02-26  
**PR:** #12 — feat: font assembly pipeline (GlyphVectorizer + FontAssembler + FontDownloader)  
**Branch:** feat/togusa-font-assembly → dev  
**Verdict:** REQUEST CHANGES (GitHub self-review restriction → posted as comment #3966638403)

---

## Blocking Issues (3)

### #1 — API surface mismatch: tests expect classes, implementation ships functions

- `fontPipeline.test.ts` imports `GlyphVectorizer` as a class (`new GlyphVectorizer()`, `.vectorize()`) and `FontAssembler` as a class (`new FontAssembler()`, `.assemble()`)
- `GlyphVectorizer.ts` exports plain function `vectorizeGlyph(data: Float32Array): opentype.Path`
- `FontAssembler.ts` exports plain function `assembleFontFromGlyphs(Map<number,Float32Array>, string): ArrayBuffer`
- **Effect:** 12 of 15 tests fail at import with "not a constructor". Only 3 FontDownloader tests pass.
- **Fix:** Add class wrappers `GlyphVectorizer.vectorize()` and `FontAssembler.assemble()` delegating to the plain functions.

### #2 — cyrillicCharset.ts Yo/yo indices conflict with LOCKED tensor contract

- `decisions.md` LOCKED: Yo (U+0401) at model index **6**, yo (U+0451) at model index **39**
- `cyrillicCharset.ts` actual: Yo at index **32** (end of uppercase block), yo at index **65** (end of lowercase block)
- `App.tsx` passes `CYRILLIC_CHARS[i].index` as `char_index` to model inference
- **Effect:** Model receives wrong char_index (32/65 instead of 6/39) → generates wrong glyphs for Yo and yo. Silent correctness bug.
- **Fix:** Reorder cyrillicCharset.ts to use alphabetical indices: A=0,B=1,V=2,G=3,D=4,E=5,Yo=6,Zh=7...Ya=32 for uppercase; a=33...e=38,yo=39,zh=40...ya=64 for lowercase. Total 66 glyphs at indices 0-65.

### #3 — makeGlyphImages() in fontPipeline.test.ts uses wrong key type

- `makeGlyphImages()` returns `Map<string, Float32Array>` keyed by character strings
- `assembleFontFromGlyphs` expects `Map<number, Float32Array>` keyed by model indices
- **Effect:** `glyphImages.get(index)` (number key) always returns `undefined` against a string-keyed Map → every glyph is a blank path. FontAssembler tests 7-11 silently test an empty font.
- **Fix:** Change `makeGlyphImages()` to return `Map<number, Float32Array>` keyed by index 0-65.

---

## What Passed

| Check | Result |
|-------|--------|
| Coordinate math: X_SCALE=600/128, Y flip 800→-200 | PASS |
| Threshold `data[...] > 0` (not >= 0) | PASS |
| opentype.js metrics: UPM=1000, asc=800, desc=-200, adv=600 | PASS |
| .notdef at slot 0 | PASS |
| OFL license in names.license + names.licenseURL | PASS |
| Download button disabled until generationStatus==='done' && fontBuffer | PASS |
| Progress counter "Generating... N/66" | PASS |
| Single inference pass (no double-inference) | PASS |
| opentype.js ^1.3.4 in package.json | PASS |
| FontDownloader: Blob lifecycle, MIME font/otf, anchor click | PASS |

---

## Status

Awaiting Togusa fix push. Will re-review on next push to the branch.
