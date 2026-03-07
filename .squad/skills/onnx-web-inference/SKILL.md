# Skill: ONNX Runtime Web — Browser Inference Pattern

**Category:** Frontend / AI Runtime  
**Author:** Togusa  
**Last updated:** 2026-03-07

---

## Pattern Summary

Load and run an ONNX model entirely in the browser using `onnxruntime-web`, with:
- Progress-tracked streaming fetch
- WebGL → WASM execution provider fallback
- Non-blocking inference via Web Worker
- Request-ID multiplexing for concurrent calls

---

## File Structure

```
src/inference/
  OnnxInference.ts        — Direct InferenceSession wrapper (used in worker)
  ModelLoader.ts          — Singleton; spawns worker; promise API with requestId tracking
  browserSupport.ts       — Synchronous capability detection (WASM/WebGL/Workers)
  worker/
    inferenceWorker.ts    — Vite module Worker; owns the ONNX session
```

---

## Key Implementation

### Model Loading with Progress

```typescript
// In inferenceWorker.ts (runs inside Web Worker)
const response = await fetch(modelUrl);
const contentLength = Number(response.headers.get('Content-Length') ?? 0);
const reader = response.body!.getReader();
const chunks: Uint8Array[] = [];
let received = 0;

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  chunks.push(value);
  received += value.length;
  if (contentLength > 0) {
    const progress = Math.round((received / contentLength) * 100);
    self.postMessage({ type: 'progress', progress });
  }
}

// Assemble buffer and create session
const buffer = new Uint8Array(received);
let offset = 0;
for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.length; }

session = await ort.InferenceSession.create(buffer.buffer, {
  executionProviders: ['webgl', 'wasm'],
});
self.postMessage({ type: 'loaded' });
```

### Tensor Construction (v1 contract)

```typescript
// style_glyphs: MUST include batch dimension
const styleTensor = new ort.Tensor('float32', styleGlyphs, [1, 10, 1, 128, 128]);
// char_index: int64, BigInt64Array required
const charTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(charIndex)]), [1]);

const results = await session.run({ style_glyphs: styleTensor, char_index: charTensor });
const output = results['generated_glyph'].data as Float32Array;
```

### Pixel Denormalisation (white background, black glyph display)

```typescript
// Model space: +1.0 = glyph ink, -1.0 = background
// Display: +1 → 0 (black), -1 → 255 (white)
const val = Math.max(0, Math.min(255, Math.round(((1 - output[i]) / 2) * 255)));
```

### Normalisation Convention

Training renders **white glyph on black background** → normalised with `mean=0.5, std=0.5`:
- glyph pixel (255) → **+1.0**
- background pixel (0) → **-1.0**

To feed the model from a black-glyph-on-white rendered canvas:
```typescript
const brightness = imageData.data[px * 4] / 255; // [0,1]
result[offset + px] = 1 - brightness * 2;         // white bg → -1, black glyph → +1
```

---

## Browser Support Detection

```typescript
import { detectBrowserSupport } from './browserSupport';

const support = detectBrowserSupport();
if (!support.supported) {
  showError(support.reason);
  return;
}
// Use support.executionProviders for session creation
```

---

## Worker Message Protocol

| Direction | Message |
|---|---|
| Host → Worker | `{ type: 'load', modelUrl }` |
| Host → Worker | `{ type: 'infer', styleGlyphs, charIndex, requestId }` |
| Worker → Host | `{ type: 'progress', progress: 0–100 }` |
| Worker → Host | `{ type: 'loaded' }` |
| Worker → Host | `{ type: 'result', output: Float32Array, requestId }` |
| Worker → Host | `{ type: 'error', message, requestId? }` |

---

## Gotchas

- **Batch dimension**: `style_glyphs` shape is `[1, 10, 1, 128, 128]` — omitting the leading `1` causes a runtime shape error.
- **int64 tensors**: Use `BigInt64Array.from([BigInt(charIndex)])`, not `Int32Array`.
- **WASM MIME type**: Vite/ASP.NET must serve `.wasm` files with `Content-Type: application/wasm` for the WASM backend to load.
- **SharedArrayBuffer**: Multi-threaded WASM requires `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers.
- **Output name**: Use `results['generated_glyph']`; add `?? Object.values(results)[0]` fallback for robustness.
