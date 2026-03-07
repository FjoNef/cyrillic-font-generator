# Skill: Playwright + ONNX Runtime Web Testing

**Author:** Saito  
**Created:** 2026-03-07  
**Tags:** playwright, onnxruntime-web, wasm, e2e, performance-testing  

## Context

This pattern emerged from Issue #23 in the Cyrillic Font Generator project. The goal was to promote 14 vitest performance stubs to live browser-based assertions against real ONNX inference.

## Pattern: Injecting ort into a Playwright Page

When testing onnxruntime-web inference in Playwright, the cleanest approach is to inject the UMD bundle directly and intercept WASM file requests:

```typescript
// 1. Inject ort UMD bundle into the page
await page.addScriptTag({ 
  content: fs.readFileSync('node_modules/onnxruntime-web/dist/ort.wasm.min.js', 'utf-8') 
});

// 2. Configure ort to load WASM from an interceptable route
await page.evaluate(() => {
  const ort = (window as any).ort;
  ort.env.wasm.wasmPaths = '/ort-wasm-dist/';
  ort.env.wasm.numThreads = 1; // avoids SharedArrayBuffer COOP requirement in CI
});

// 3. Intercept WASM files and serve from node_modules
await page.route('**/ort-wasm-dist/**', async route => {
  const filename = path.basename(new URL(route.request().url()).pathname);
  const filePath = path.join('node_modules/onnxruntime-web/dist', filename);
  if (fs.existsSync(filePath)) {
    await route.fulfill({
      status: 200,
      contentType: filename.endsWith('.wasm') ? 'application/wasm' : 'application/javascript',
      body: fs.readFileSync(filePath),
    });
  } else {
    await route.continue();
  }
});
```

## Pattern: Stub ONNX Model for CI

Generate a minimal ONNX model with the same tensor contract as the production model:

```python
import onnx
from onnx import helper, TensorProto

style_glyphs = helper.make_tensor_value_info('style_glyphs', TensorProto.FLOAT, [1, 10, 1, 128, 128])
char_index = helper.make_tensor_value_info('char_index', TensorProto.INT64, [1])
generated_glyph = helper.make_tensor_value_info('generated_glyph', TensorProto.FLOAT, [1, 1, 128, 128])

# Initializers for Slice op
starts = helper.make_tensor('starts', TensorProto.INT64, [1], [0])
ends = helper.make_tensor('ends', TensorProto.INT64, [1], [1])
axes = helper.make_tensor('axes', TensorProto.INT64, [1], [1])
steps = helper.make_tensor('steps', TensorProto.INT64, [1], [1])
reshape_shape = helper.make_tensor('reshape_shape', TensorProto.INT64, [4], [1, 1, 128, 128])

slice_node = helper.make_node('Slice', ['style_glyphs','starts','ends','axes','steps'], ['sliced'])
reshape_node = helper.make_node('Reshape', ['sliced','reshape_shape'], ['generated_glyph'])

graph = helper.make_graph([slice_node, reshape_node], 'stub',
    [style_glyphs, char_index], [generated_glyph],
    initializer=[starts, ends, axes, steps, reshape_shape])

model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, 'tests/fixtures/stub-generator.onnx')
# Result: ~345 bytes
```

## Pattern: Measuring Inference Time in page.evaluate()

```typescript
const result = await page.evaluate(async (modelArray: number[]) => {
  const ort = (window as any).ort;
  const modelBytes = new Uint8Array(modelArray).buffer;
  
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ['wasm'] });
  const loadMs = performance.now() - t0;
  
  const styleGlyphs = new Float32Array(1 * 10 * 1 * 128 * 128).fill(0.5);
  const styleTensor = new ort.Tensor('float32', styleGlyphs, [1, 10, 1, 128, 128]);
  const indexTensor = new ort.Tensor('int64', BigInt64Array.from([0n]), [1]);
  
  const t1 = performance.now();
  const results = await session.run({ style_glyphs: styleTensor, char_index: indexTensor });
  const inferMs = performance.now() - t1;
  
  const out = results['generated_glyph'] ?? Object.values(results)[0];
  return { loadMs, inferMs, shape: Array.from(out.dims), dtype: out.type };
}, Array.from(fs.readFileSync('tests/fixtures/stub-generator.onnx')));

expect(result.loadMs).toBeLessThan(5_000);
expect(result.inferMs).toBeLessThan(500);
expect(result.shape).toEqual([1, 1, 128, 128]);
```

## Known Gotchas

1. **`import.meta.url` in Playwright tests**: Playwright compiles TypeScript to CJS, so `__dirname` is available as a global. Do NOT use the `fileURLToPath(import.meta.url)` ESM pattern — just use `__dirname` directly.

2. **SharedArrayBuffer for multi-threaded WASM**: Multi-threaded WASM requires `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. Set `ort.env.wasm.numThreads = 1` in CI to avoid this requirement.

3. **UMD bundle required for page injection**: Use `ort.wasm.min.js` (WASM-only UMD bundle), not the ESM variant. The UMD bundle exposes `window.ort` when injected as an inline script.

4. **Serving model bytes to evaluate()**: `page.evaluate()` serializes JavaScript values. Pass model bytes as `Array.from(Buffer)`, not as a Buffer directly (Buffer is not serializable). Reconstruct as `new Uint8Array(modelArray).buffer` inside evaluate.

5. **Firefox WASM JIT**: Firefox takes 5-10× longer than Chromium for WASM compilation on first load. Set generous timeouts (default 30s is usually sufficient).

## Playwright Config for ONNX Tests

```typescript
// playwright.config.ts
export default defineConfig({
  testDir: './e2e',
  use: { baseURL: 'http://localhost:5173' },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
  ],
  webServer: {
    command: 'npm run dev',    // Vite dev server — no build needed
    url: 'http://localhost:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 1 : undefined,
});
```
