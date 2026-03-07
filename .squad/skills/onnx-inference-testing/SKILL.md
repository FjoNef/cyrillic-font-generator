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

## Pattern 10 — WebApplicationFactory with Temp Model File

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
