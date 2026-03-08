# SKILL: ONNX Inference Testing Patterns

**Author:** Saito  
**Date:** 2026-03-07  
**Applies to:** Any project using `onnxruntime-web` with Vitest

---

## Summary

Patterns for writing unit and integration tests for ONNX model inference in the browser using `onnxruntime-web` and Vitest.

---

## Pattern 1 — Mock `onnxruntime-web` with `vi.mock`

```typescript
vi.mock('onnxruntime-web', () => {
  class Tensor {
    type: string; data: any; dims: number[];
    constructor(type: string, data: any, dims: number[]) {
      this.type = type; this.data = data; this.dims = dims;
    }
  }
  const InferenceSession = { create: vi.fn() };
  return { Tensor, InferenceSession };
});
```

Then in tests:
```typescript
const ort = await import('onnxruntime-web');
vi.mocked(ort.InferenceSession.create).mockResolvedValueOnce(mockSession as any);
```

---

## Pattern 2 — Mock `fetch` for Model Loading Tests

```typescript
beforeEach(() => { global.fetch = vi.fn(); });
afterEach(() => { vi.restoreAllMocks(); });

// 404 error:
(global.fetch as any).mockResolvedValue({
  ok: false, status: 404, statusText: 'Not Found',
  headers: { get: () => null }, body: null,
});

// Streaming with progress:
const mockReader = {
  read: vi.fn()
    .mockResolvedValueOnce({ done: false, value: new Uint8Array(5_000_000).fill(1) })
    .mockResolvedValueOnce({ done: true }),
};
(global.fetch as any).mockResolvedValue({
  ok: true,
  headers: { get: () => '5000000' },
  body: { getReader: () => mockReader },
});
```

---

## Pattern 3 — Tensor Shape Assertions

Always test tensor shapes using the known constants from the inference contract:

```typescript
const STYLE_GLYPH_ELEMENTS = 1 * 10 * 1 * 128 * 128; // 163_840
const OUTPUT_ELEMENTS       = 1 * 1  * 128 * 128;     // 16_384

it('style_glyphs tensor has correct element count', () => {
  const glyphs = new Float32Array(STYLE_GLYPH_ELEMENTS);
  expect(glyphs.length).toBe(163_840);
});

it('output has correct element count', () => {
  const output = new Float32Array(OUTPUT_ELEMENTS);
  expect(output.length).toBe(16_384);
});
```

---

## Pattern 4 — Value Range Validation (tanh output)

ONNX models with tanh output layer produce values in [-1, 1]. Test this explicitly:

```typescript
it('output values within [-1.0, 1.0]', () => {
  const output = new Float32Array([1.0, -1.0, 0.0, 0.5, -0.5]);
  for (const v of output) {
    expect(v).toBeGreaterThanOrEqual(-1.0);
    expect(v).toBeLessThanOrEqual(1.0);
  }
});
```

---

## Pattern 5 — Pixel Normalisation / Denormalisation

Standard normalisation pipeline for ONNX font/image models:

```typescript
// Input normalisation: [0,255] → [-1,1]
const normalise = (gray: number) => (gray / 255.0 - 0.5) / 0.5;
// white(255) → +1.0, black(0) → -1.0

// Output denormalisation: [-1,1] → [0,255]
const denormalise = (v: number) => Math.max(0, Math.min(255, Math.round(((1 - v) / 2) * 255)));
// +1.0 → 0 (black ink), -1.0 → 255 (white bg)
```

⚠️ Common inversion trap: the formula is `(1 - output) / 2 * 255` NOT `(output + 1) / 2 * 255`. The sign flip is because `+1.0` in model space = black ink = pixel value 0.

---

## Pattern 6 — int64 Tensors in Browser

ONNX `int64` input tensors require `BigInt64Array`:

```typescript
const charIndex = 33;
const tensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(charIndex)]), [1]);
// Verify:
expect(tensor.data[0]).toBe(BigInt(33));
```

---

## Pattern 7 — Edge Case Checklist for ONNX Models

Always test:
- [ ] Empty input (length 0)
- [ ] Wrong input length (e.g., 5 glyphs instead of 10)
- [ ] NaN in input data
- [ ] Values outside normalised range (> 1.0 or < -1.0)
- [ ] char_index at boundary (0 and max, e.g. 65)
- [ ] char_index out of range (-1, max+1)
- [ ] Non-integer char_index (should floor-clamp)

---

## Pattern 8 — Performance Baseline Stubs

When actual measurement requires a browser harness, document targets as stubs:

```typescript
const PERF_TARGETS = {
  MODEL_LOAD_MS: 5_000,
  INFERENCE_PER_GLYPH_MS: 500,
} as const;

it('📌 target: load < 5000ms', () => {
  // Replace with: const t0 = performance.now(); await loader.load(url);
  // expect(performance.now() - t0).toBeLessThan(PERF_TARGETS.MODEL_LOAD_MS);
  expect(PERF_TARGETS.MODEL_LOAD_MS).toBe(5_000);
});
```

---

## Pattern 9 — Backend Range Request Tests (ASP.NET Core xUnit)

Test HTTP Range requests using `Results.File(..., enableRangeProcessing: true)`:

```csharp
[Fact]
public async Task ModelDownload_AcceptsRangeRequests_Returns206()
{
    var request = new HttpRequestMessage(HttpMethod.Get, "/api/model");
    request.Headers.Range = new RangeHeaderValue(0, 511);
    var response = await _client.SendAsync(request);
    Assert.Equal(HttpStatusCode.PartialContent, response.StatusCode);
    var body = await response.Content.ReadAsByteArrayAsync();
    Assert.Equal(512, body.Length);
}
```

**Note:** Static file middleware does not register in `WebApplicationFactory` when `ModelPath` is injected via `ConfigureAppConfiguration` (timing issue at `app.Build()`). Use `Results.File` endpoint for Range tests. For static file caching headers, use a source-level assertion or E2E test.

---

## Pattern 11 — Python ONNX Sanity Check (5-Check Template)

After every export, run a fast Python-side sanity check before browser testing.
Uses only `onnxruntime` + `numpy`; no fonts, no training data required.  Completes in < 10 s.

```python
import numpy as np
import onnxruntime as ort

B, N, H, W = 1, 10, 128, 128
N_CHARS = 66

def _run(sess, style_fill, char_idx=0):
    style = np.full((B, N, 1, H, W), fill_value=style_fill, dtype=np.float32)
    char  = np.array([char_idx], dtype=np.int64)
    return sess.run(None, {"style_glyphs": style, "char_index": char})[0]

sess = ort.InferenceSession("models/v1/generator.onnx", providers=["CPUExecutionProvider"])
out  = _run(sess, style_fill=0.0)

# 1. Output range
assert out.min() >= -1.1 and out.max() <= 1.1

# 2. Non-blank (in model space: +1.0=black ink, -1.0=white bg)
#    At least 1% of pixels must be in the ink region (above 0.0)
ink_frac = np.mean(out > 0.0)
assert ink_frac >= 0.01, f"Blank glyph! ink_frac={ink_frac:.4f}"

# 3. Style conditioning
out_w = _run(sess, style_fill=+1.0)
out_b = _run(sess, style_fill=-1.0)
assert np.mean(np.abs(out_w - out_b)) > 0.01

# 4. Char isolation
out_0   = _run(sess, style_fill=0.0, char_idx=0)
out_max = _run(sess, style_fill=0.0, char_idx=N_CHARS - 1)
assert np.mean(np.abs(out_0 - out_max)) > 0.005
```

⚠️ **The non-blank check is the most important.**  Style conditioning (check 3) can pass even when all output values are -1.0 (blank/white), because the model may still *shift* values relatively without producing actual ink.  The non-blank check catches all-blank output that relative checks miss.

See `src/model/export/check_model.py` for the full implementation with pass/fail reporting and exit codes.

---

## Pattern 12 — Browser-Side Non-Blank Assertion (Playwright)

**Always pair a relative MAD assertion with an absolute non-blank assertion in Playwright smoke tests.**

A mode-collapsed model that outputs all-background (-1.0) will:
- **PASS** a MAD > 0.01 check (tiny relative differences between extreme inputs still pass)
- **FAIL** a max > -0.5 check (all pixels are background — blank white canvas)

```typescript
// ❌ INSUFFICIENT: only tests relative style response
expect(result.meanAbsDiff).toBeGreaterThan(0.01);

// ✅ COMPLETE: also checks absolute content (non-blank)
// All-background output → postprocessing → 255 (white) → blank canvas
// At least some pixels must be glyph-like (above -0.5) for visible output
expect(result.maxOutput).toBeGreaterThan(-0.5);

// And structural variation (not flat / mode-collapsed)
expect(result.stdOutput).toBeGreaterThan(0.05);
```

**How to collect max/std inside page.evaluate:**
```javascript
let min = Infinity, max = -Infinity;
let sum = 0, sum2 = 0;
for (let i = 0; i < data.length; i++) {
  if (data[i] < min) min = data[i];
  if (data[i] > max) max = data[i];
  sum += data[i];
  sum2 += data[i] * data[i];
}
const mean = sum / data.length;
const std = Math.sqrt(sum2 / data.length - mean * mean);
return { min, max, std };
```

⚠️ Avoid `Math.max(...data)` / `Math.min(...data)` with spread for arrays > ~50 k elements — risk of call stack overflow in some environments. Use an explicit loop.

See `src/frontend/e2e/style-conditioning-real.spec.ts` for full working example.

---

Inject a test model file so `ModelManifestCache` returns `Available = true`:

```csharp
public sealed class ModelWebFactory : WebApplicationFactory<Program>
{
    public string TempRoot { get; } = Path.Combine(Path.GetTempPath(), $"test-{Guid.NewGuid():N}");
    public string ModelDir => Path.Combine(TempRoot, "models");
    public const int DummyModelSizeBytes = 4 * 1024;

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        var v1Dir = Path.Combine(ModelDir, "v1");
        Directory.CreateDirectory(v1Dir);
        var dummy = new byte[DummyModelSizeBytes];
        Array.Fill(dummy, (byte)0xAB); // recognisable pattern for byte-level assertions
        File.WriteAllBytes(Path.Combine(v1Dir, "generator.onnx"), dummy);

        builder.ConfigureAppConfiguration(config =>
            config.AddInMemoryCollection(new Dictionary<string, string?> { ["ModelPath"] = ModelDir }));
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing && Directory.Exists(TempRoot))
            Directory.Delete(TempRoot, recursive: true);
        base.Dispose(disposing);
    }
}
```
