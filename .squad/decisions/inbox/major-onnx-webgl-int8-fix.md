# Decision: ONNX WebGL+INT8 Incompatibility Fix

**Date:** 2026-03-08  
**Author:** Major (AI/ML Engineer)  
**Status:** Implemented  
**Branch:** squad/53-full-e2e-pipeline-fix

## Context

The INT8 quantized ONNX model was producing all-background output (-1.0) when run through `OnnxInference.ts` in the browser, despite the same model working correctly in the Python sanity check script and in `inferenceWorker.ts`.

## Problem

`OnnxInference.ts` line 60 used:
```typescript
executionProviders: ['webgl', 'wasm']
```

The model is INT8 dynamically quantized and uses QLinear operations (QLinearConv, QLinearMatMul, etc.). **ONNX Runtime Web's WebGL backend does NOT support QLinear operations.** When WebGL is used as the primary execution provider:
- ORT attempts to run QLinear ops on WebGL
- Silently falls back to incorrect implementation
- All output values are -1.0 (background)
- No error is thrown — the failure is completely silent

This bug affected:
- Issue #48: Blank Cyrillic glyphs in downloaded font
- Any code path using `OnnxInference.ts` (preview, single-glyph generation)

## Decision

**Use WASM-only execution provider for INT8 quantized models.**

Changed `OnnxInference.ts` line 60 to:
```typescript
this.session = await ort.InferenceSession.create(buffer.buffer, {
  executionProviders: ['wasm'],  // INT8 quantized model requires WASM only
});
```

Added explanatory comments in both the class docstring and the `loadModel()` method documenting the WebGL+INT8 incompatibility.

## Verification

1. ✅ **inferenceWorker.ts already correct:** Uses `['wasm']` only (fixed in PR #49)
2. ✅ **All WASM files present:** 8 variant files in `public/ort-wasm/` (PR #50)
3. ✅ **SharedArrayBuffer guard:** Already present in both files (lines 85-87 in OnnxInference.ts)
4. ✅ **Output copy:** `ModelLoader.ts` copies defensively; `OnnxInference.ts` consumes synchronously (implicit copy)
5. ✅ **Tests pass:** All 52 inference tests pass (styleConditioning, onnxContract, integration)

## Related Issues

- **Closes #41:** SharedArrayBuffer alias vulnerability — already fixed in existing code
- **Closes #48:** Blank Cyrillic glyphs — root cause was WebGL+INT8 incompatibility

## Key Learning

**INT8 quantized ONNX models (QLinear ops) are NOT compatible with ORT Web's WebGL backend.**

Rules:
1. Always use `executionProviders: ['wasm']` for INT8 quantized models
2. The WebGL backend will silently produce incorrect output rather than throwing an error
3. This applies to all QLinear* operations (QLinearConv, QLinearMatMul, etc.)
4. FP32 and FP16 models CAN use WebGL — only INT8 is affected

## Impact

- **OnnxInference.ts:** Fixed to use WASM-only (1 line change + comments)
- **inferenceWorker.ts:** Already correct (no change needed)
- **browserSupport.ts:** No change needed — detection logic is for UI only, not used by inference
- **Performance:** WASM inference is slightly slower than WebGL on GPU-capable devices, but INT8 quantization offsets this (smaller model, faster decode). Net result: comparable or better performance than FP32+WebGL.

## Files Changed

- `src/frontend/src/inference/OnnxInference.ts`: executionProviders ['webgl', 'wasm'] → ['wasm']
- `.squad/agents/major/history.md`: New learning entry
- `.squad/decisions/inbox/major-onnx-webgl-int8-fix.md`: This decision document

## Next Steps

1. Commit changes with message: `fix(inference): use WASM-only for INT8 model, fix OnnxInference WebGL bug (closes #41, #48) (#53)`
2. Merge to dev
3. Close issues #41, #48
