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

