# Aramaki PR #12 Revision — Fix Summary

**Date:** 2026-02-26  
**Branch:** feat/togusa-font-assembly  
**Commit:** 6681fcd  
**Context:** Togusa locked out (Reviewer Rejection Lockout Protocol). Aramaki applied fixes on behalf of team.

## Fix 1 — cyrillicCharset.ts: Ё/ё indices corrected

**File:** `src/frontend/src/font/cyrillicCharset.ts`  
**Problem:** Ё was assigned `index: 32` (end of uppercase block); ё was `index: 65` (end of lowercase).  
**Required (LOCKED tensor contract):** Ё at index 6, ё at index 39.  
**Fix:** Rebuilt uppercase and lowercase arrays to interleave Ё/ё at their correct alphabetical positions:
- Uppercase: А(0)–Е(5), Ё(6), Ж(7)–Я(32)
- Lowercase: а(33)–е(38), ё(39), ж(40)–я(65)

## Fix 2 — fontPipeline.test.ts: class API → function API

**File:** `src/frontend/src/fontPipeline.test.ts`  
**Problem:** Tests used `new GlyphVectorizer()` / `vectorizer.vectorize(data)` and `new FontAssembler()` / `assembler.assemble(glyphs, name)`. Implementations export plain functions, not classes.  
**Decision:** Align tests to the function API (do not refactor implementations — they are cleaner as functions).  
**Fix:** Replaced class instantiation with direct function calls:
- `vectorizer.vectorize(data)` → `vectorizeGlyph(data)`
- `assembler.assemble(glyphs, name)` → `assembleFontFromGlyphs(glyphs, name)`
- Removed `beforeEach` blocks and instance variable declarations from GlyphVectorizer and FontAssembler suites.

## Fix 3 — fontPipeline.test.ts: Map key type in test helper

**File:** `src/frontend/src/fontPipeline.test.ts`  
**Problem:** `makeGlyphImages()` returned `Map<string, Float32Array>` with Cyrillic character strings as keys. `assembleFontFromGlyphs` expects `Map<number, Float32Array>` keyed by model index (0-65).  
**Fix:** Rewrote `makeGlyphImages()` to return `Map<number, Float32Array>` using numeric indices 0-65 as keys.

## Test Result

All fontPipeline.test.ts tests pass. Pre-existing ModelLoader.test.ts failures (unrelated jsdom/mock issues) remain unchanged — not in scope.
