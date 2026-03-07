# Decision: E2E Glyph Generation Flow — Integration Test Pattern

**Date:** 2026-03-07  
**Author:** Togusa  
**Issue:** #27

---

## Context

Issue #27 ("Wire up end-to-end glyph generation flow in UI") required:  
- Font upload → style glyph extraction (10 Latin chars)  
- ModelLoader.infer × 66 (Cyrillic char indices 0–65)  
- assembleFontFromGlyphs → downloadFont  
- Progress display during generation  

The UI flow was already implemented in `squad/27-e2e-glyph-generation-ui` (PR #31).  
The integration test file had 7 placeholder tests that all trivially passed.

---

## Decision

**Replace placeholder integration tests with real pipeline tests that mock at the Worker boundary.**

### Pattern

```typescript
// 1. Mock the Web Worker — intercept 'infer' messages, respond synchronously
const worker = { postMessage: vi.fn(msg => {
  if (msg.type === 'infer') {
    Promise.resolve().then(() =>
      worker.onmessage?.({ data: { type: 'result', output: new Float32Array(16384).fill(0.5), requestId: msg.requestId } })
    );
  }
}), ... };
global.Worker = vi.fn(() => worker);

// 2. Load model (trigger 'loaded' message from mock)
// 3. Run full 66-char loop, accumulate Map<number, Float32Array>
// 4. Call assembleFontFromGlyphs → check OTF magic bytes
// 5. Call downloadFont → assert MIME type, filename, URL.revokeObjectURL
```

### Why mock at the Worker boundary (not ONNX Runtime)

- Web Workers cannot run in jsdom/Vitest — mocking the Worker constructor is the only viable boundary.
- This tests the ModelLoader request/response multiplexing, which is the main risk point.
- Lower-level ONNX contract tests (tensor shapes, dtypes) live in `onnxContract.test.ts`.

---

## Consequences

- 7 placeholder tests → 7 real tests; total suite stays 96 tests (no regressions)
- Each aspect of the E2E loop now has a dedicated assertion (loop coverage, OTF validity, progress, error propagation, pre-load guard, download contract)
- Future agents adding inference features should follow this mock-at-Worker pattern for integration tests
