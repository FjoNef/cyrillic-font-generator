# Decision: ImageData Mock in test-setup.ts for jsdom

**Date:** 2026-03-07  
**Author:** Togusa (Frontend Dev)  
**Status:** Implemented (PR #47, sha: 7863589)

## Context

PR #47 CI was failing with `ReferenceError: ImageData is not defined` in `styleConditioning.test.ts`. The error occurred at `OnnxInference.ts:111`:

```typescript
return new ImageData(pixels, size, size);
```

This code runs in production browsers (where `ImageData` is native) but also in Vitest tests (where jsdom does not provide a Canvas API constructor).

## Problem

- **jsdom limitation:** jsdom provides DOM APIs but lacks native Canvas implementation.
- **OnnxInference contract:** `generateGlyph()` must return an `ImageData` object (browser-standard type).
- **Test coverage:** 6 new style conditioning tests invoke `generateGlyph()` and trigger the ImageData constructor.

## Options Considered

### Option A: Mock ImageData globally in test-setup.ts ✅ **CHOSEN**
**Pros:**
- Single source of truth for all test environment polyfills.
- Consistent with existing `Path2D` and `getImageData` mocks in same file.
- Zero production code changes.
- Future-proof: guards with `typeof globalThis.ImageData === 'undefined'`.

**Cons:**
- None.

### Option B: Mock generateGlyph in each test file
**Pros:**
- Test-local control.

**Cons:**
- Violates DRY (38 onnxContract tests + 6 styleConditioning tests).
- Mocking `generateGlyph` defeats the purpose of integration tests (they need to exercise the real ImageData return path).

### Option C: Switch to happy-dom
**Pros:**
- happy-dom provides more Canvas APIs.

**Cons:**
- Unknown compatibility with existing tests.
- Larger environment change for a single API.
- Not guaranteed to provide `ImageData` constructor.

## Decision

Implement **Option A**: Add a minimal `ImageData` class mock to `test-setup.ts`.

## Implementation

Added to `src/frontend/src/test-setup.ts` (lines 38-53):

```typescript
// ImageData is not available in jsdom; provide a minimal implementation.
if (typeof (globalThis as any).ImageData === 'undefined') {
  (globalThis as any).ImageData = class ImageData {
    data: Uint8ClampedArray;
    width: number;
    height: number;

    constructor(dataOrWidth: Uint8ClampedArray | number, widthOrHeight: number, height?: number) {
      if (dataOrWidth instanceof Uint8ClampedArray) {
        this.data = dataOrWidth;
        this.width = widthOrHeight;
        this.height = height!;
      } else {
        this.width = dataOrWidth;
        this.height = widthOrHeight;
        this.data = new Uint8ClampedArray(this.width * this.height * 4);
      }
    }
  };
}
```

Supports both Canvas API constructors:
1. `new ImageData(data: Uint8ClampedArray, width: number, height: number)` — used by `OnnxInference.ts`
2. `new ImageData(width: number, height: number)` — Canvas API standard

## Verification

- **Before:** 6/6 styleConditioning tests failing, 102/108 total passing.
- **After:** 108/108 tests passing, zero regressions.

## Consequences

- ✅ CI unblocked for PR #47.
- ✅ All future tests using `ImageData` will work without additional mocking.
- ✅ Pattern established for adding jsdom polyfills (use test-setup.ts, guard with `typeof`).
- ⚠️ If Vitest/jsdom adds native `ImageData` in the future, our guard ensures the mock is skipped.
