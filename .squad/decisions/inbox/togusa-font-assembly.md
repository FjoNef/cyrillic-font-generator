# Togusa — Font Assembly Pipeline Decisions

**By:** Togusa (Frontend Dev)  
**Branch:** feat/togusa-font-assembly  
**Date:** 2026-02-26

## Key Decisions

### 1. Separate GlyphVectorizer, FontAssembler, FontDownloader modules
Instead of keeping vectorization and assembly inside FontLoader, created three focused modules. FontLoader continues to exist for style glyph extraction; it is not replaced.

### 2. Correct font coordinate mapping in GlyphVectorizer
The existing FontLoader.vectorizeGlyph had two bugs:
- **X scale**: used `1000/128` (UPM/pixels), which mapped columns 0–128 → 0–1000. Correct is `600/128` (advance width / pixels), mapping 0–600.
- **Y offset**: used `(size - row) * scale` giving range 0–1000, outside ascender/descender. Correct is `800 - row * (1000/128)`, placing row 0 at ascender 800 and row 128 at descender -200.

GlyphVectorizer uses the corrected mapping:
- `X_SCALE = 600/128 ≈ 4.6875`
- `Y_SCALE = 1000/128 ≈ 7.8125`
- `yTop = 800 - row * Y_SCALE`, `yBottom = 800 - (row+1) * Y_SCALE`

### 3. Single inference pass — no double inference
The old App.tsx flow ran inference twice: once to build the ImageData preview (Map<string, ImageData>), then again inside `assembleCyrillicFont` which called `inferFn` per glyph. New flow: one inference loop stores both the raw `Float32Array` (for FontAssembler) and the converted `ImageData` (for GlyphPreview). FontAssembler receives `Map<number, Float32Array>` keyed by model index.

### 4. OFL license in name table
FontAssembler writes SIL OFL 1.1 license text and URL into opentype.js `font.names.license` / `font.names.licenseURL` (name IDs 13/14). Satisfies the licensing requirement from `.squad/decisions.md`.

### 5. Download button gating
The Download button is disabled (`disabled={generationStatus !== 'done' || !fontBuffer}`) until all 66 glyphs are complete and the OTF buffer is assembled. A "Generating glyphs… N/66" progress line appears in the download section while generation is running.

### 6. FontAssembler API
`assembleFontFromGlyphs(glyphImages: Map<number, Float32Array>, familyName: string): ArrayBuffer`  
- Synchronous (vectorization is CPU-only, no async needed)  
- Blank `.notdef` glyph always at index 0  
- Falls back to blank path for any missing glyph index

### 7. FontDownloader cleans up Blob URL synchronously
`URL.revokeObjectURL` is called immediately after the anchor click event is dispatched. This is safe because the download is already queued by the browser at click time.
