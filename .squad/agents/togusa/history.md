# Togusa — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **My focus:** Web UI + browser-side inference integration.

## Core Context

### Prior Work Completed (Feb 26 – Mar 5)
- ✅ PR #15: Fixed opentype.js CJS/ESM interop in Vitest (resolved TypeError on module load; added vitest.config.ts with resolve.alias)
- ✅ PR #14: Fixed CI test failures (jsdom install, ModelLoader singleton, async Promise semantics)
- ✅ PR #13: Fixed build TS errors by excluding test files from tsconfig
- ✅ PR #4: Font assembly pipeline (GlyphVectorizer, FontAssembler, FontDownloader) with OFL metadata
- ✅ Issue #16: Fixed model fetch URL (/api/models/v1/generator.onnx → /api/model), all 41 tests passing
- ✅ Frontend inference end-to-end: Web Worker + ModelLoader singleton + vectorization + font assembly complete
- **Dependency:** Waiting for Major's ONNX model at models/v1/generator.onnx — once available, inference pipeline fully functional

## Learnings

### 2026-03-07T14:31:04Z: Major exports ONNX model — Ready for Integration

- **Status:** ✅ COMPLETE
- **Model file:** models/v1/generator.onnx
- **Size:** 53.1 MB (INT8 quantized)
- **Compressed delivery:** ~15.9 MB (meets ≤20 MB browser target)
- **Output shape:** (1, 1, 128, 128)
- **Output dtype:** float32
- **Value range:** [-1.0, 1.0]
- **Validation:** CPU inference SUCCESS
- **Implementation:** INT8 dynamic quantization; ConvTranspose FP32 fallback
- **Key note:** Model ready for onnxruntime-web integration with ModelLoader singleton pattern

### 2026-03-07: ONNX Browser Integration Complete — Cross-Validated by Saito

- **OnnxInference.ts**: Updated to match confirmed contract — `style_glyphs` tensor shape `[1, 10, 1, 128, 128]` (batch dim required), output from `generated_glyph` key. All TODO placeholders resolved.
- **Normalisation convention confirmed**: Training data renders white glyphs on black background (`glyph=255 → +1.0`, `bg=0 → -1.0`). FontLoader's `1 - brightness * 2` formula is correct (inverts rendered black-glyph-on-white to match training). Postprocessing `((1 - output) / 2) * 255` is correct.
- **Style chars confirmed**: Model trained on `["A","B","C","D","E","H","I","O","R","X"]` (see `dataset.py` DEFAULT_STYLE_CHARS). FontLoader was already correct. Inference contract doc had wrong chars — not a code issue.
- **browserSupport.ts created**: Detects WASM / Workers / WebGL / SharedArrayBuffer; returns recommended execution providers and human-readable error for unsupported browsers.
- **Tests**: 92/92 passing. Saito flagged HIGH risk (batch dim + output key bugs) **independently** → Togusa had **already fixed** these before Saito filed the risk. Cross-validation successful.
- **Dependency resolved**: inference pipeline is now fully wired to the actual exported model. ModelLoader fetches from `/api/model/manifest` → gets `downloadUrl` from Batou's versioned API endpoint.
- **Cross-agent sync**: Batou's `/api/model/v1/generator.onnx` endpoint matches Togusa's fetch URL exactly. Decision locked in both histories.

