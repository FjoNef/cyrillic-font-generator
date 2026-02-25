# Decision: Inference Pipeline Implementation

**Date:** 2026-02-25T143900  
**Agent:** Togusa (Frontend Dev)  
**Branch:** feat/togusa-inference-pipeline  
**PR:** #4

## Context
Implemented the end-to-end browser-based inference pipeline for Cyrillic glyph generation as specified in issue #3. The pipeline includes model loading, inference via Web Worker, glyph rendering, and font assembly.

## Key Decisions

### 1. Web Worker Architecture
**Decision:** Wrap ONNX Runtime Web in a dedicated Web Worker with promise-based message protocol.

**Rationale:**
- Keeps inference off the main thread, preventing UI blocking during the ~15-30ms WebGL inference per glyph (or ~80-150ms WASM)
- Message protocol uses per-request IDs to support concurrent inference safely
- Vite's `new Worker(new URL(...), { type: 'module' })` syntax ensures proper bundling and HMR support

**Protocol:**
```typescript
Host → Worker:
  { type: 'load', modelUrl: string }
  { type: 'infer', styleGlyphs: Float32Array, charIndex: number, requestId: string }

Worker → Host:
  { type: 'progress', progress: number }
  { type: 'loaded' }
  { type: 'result', output: Float32Array, requestId: string }
  { type: 'error', message: string, requestId?: string }
```

### 2. Glyph Vectorization
**Decision:** Implement scanline-based raster-to-path conversion instead of potrace.

**Rationale:**
- Potrace has no maintained browser-compatible port
- Scanline approach is simple and deterministic: for each row, find runs of ink pixels and draw horizontal segment rectangles
- Good enough for MVP — glyphs are recognizable and functional, though not smooth curves
- Future enhancement: proper contour tracing or integrate a potrace WASM port

**Tradeoffs:**
- ✅ Works in browser, no external dependencies
- ✅ Fast and predictable
- ❌ No bezier curves — output has stair-stepping on diagonals
- ❌ Font file size larger than optimal (more path commands)

### 3. Font Metrics
**Decision:** Fixed metrics for all generated glyphs.

**Values:**
- Units per em (UPM): 1000
- Ascender: 800
- Descender: -200
- Advance width: 600 (all glyphs)

**Rationale:**
- Simplifies implementation — no need to analyze user's original font metrics
- Standard values ensure glyphs are legible and spaced reasonably
- Future enhancement: derive metrics from user's uploaded font (especially advance widths based on Latin reference glyphs)

### 4. Model Loader Singleton
**Decision:** Export a singleton `modelLoader` instance, not a class.

**Rationale:**
- Only one model instance needed per session
- Prevents accidental multiple model loads (wasteful)
- Simplifies usage: `import { modelLoader } from './inference/ModelLoader'`
- Load promise is cached — safe to call `load()` multiple times

### 5. Style Glyph Extraction Timing
**Decision:** Extract style glyphs immediately on font upload, store in Zustand.

**Rationale:**
- Decouples font upload from generation — user can upload font before model finishes loading
- Style glyphs are needed for every inference call — compute once, reuse 66 times
- Stored as Float32Array in app state (163,840 floats = ~640 KB) — acceptable for in-memory state

### 6. Progress Reporting
**Decision:** Two progress indicators — model load % and generation count (N/66).

**Rationale:**
- Model load progress: Streams from `fetch` with `Content-Length` header, reported 0-100%
- Generation progress: Simple counter incremented after each glyph completes
- UI shows "Generating… (N/66)" in button text — lightweight, no separate progress bar needed for generation

## Implementation Notes

### Tensor Contract (from decisions.md)
- Input `style_glyphs`: [1, 10, 1, 128, 128] float32
- Input `char_index`: [1] int64 (BigInt64Array required)
- Output `generated_glyph`: [1, 1, 128, 128] float32, range [-1, 1]

### Error Handling
- Worker errors posted back via `{ type: 'error', message, requestId }`
- App.tsx catches errors, sets `generationStatus: 'error'`
- No user-facing error UI yet — errors logged to console only

### Browser Compatibility
- onnxruntime-web automatically selects WebGL backend (preferred) or WASM fallback
- Web Workers supported in all modern browsers
- opentype.js uses standard Canvas2D APIs

## Open Questions for Team

1. **Vectorization quality:** Is scanline-based approach acceptable for MVP, or should we prioritize smoother curves before v1 release?
2. **Font metrics:** Should we derive advance widths from the user's uploaded font, or keep fixed 600 UPM?
3. **Error UX:** Should we add a toast/banner for model load failures, or is console logging sufficient for MVP?
4. **Progress UX:** Should generation progress show a progress bar, or is the button counter text sufficient?

## Files Changed
- NEW `src/frontend/src/inference/worker/inferenceWorker.ts` (120 lines)
- NEW `src/frontend/src/inference/ModelLoader.ts` (120 lines)
- MODIFIED `src/frontend/src/font/FontLoader.ts` (+110 lines)
- MODIFIED `src/frontend/src/components/FontUpload.tsx` (+4 lines)
- MODIFIED `src/frontend/src/App.tsx` (+60 lines)
- MODIFIED `src/frontend/src/stores/appStore.ts` (+15 lines)
- `package.json` — Added `@types/opentype.js` devDependency

## Next Steps
- Integration testing with Major's exported ONNX model (when available at `/api/models/v1/generator.onnx`)
- Manual QA: upload a font, verify all 66 glyphs generate, download and test .otf in font viewer
- Future enhancement: replace scanline vectorization with proper contour tracing
