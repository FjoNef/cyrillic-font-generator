import * as ort from 'onnxruntime-web';

/**
 * Web Worker for ONNX model inference.
 * Runs in background thread, WASM-only backend (WebGL does not support QLinear INT8 ops).
 *
 * WASM files are served from /ort-wasm/ (copied there by scripts/copy-ort-wasm.cjs).
 * Setting wasmPaths explicitly prevents ORT from trying to infer the path from its own
 * script URL, which is unreliable inside a Vite-bundled worker.
 *
 * Message protocol:
 * - Host → Worker:
 *   { type: 'load', modelUrl: string }
 *   { type: 'infer', styleGlyphs: Float32Array, charIndex: number, requestId: string }
 *
 * - Worker → Host:
 *   { type: 'progress', progress: number }
 *   { type: 'loaded' }
 *   { type: 'result', output: Float32Array, requestId: string }
 *   { type: 'error', message: string, requestId?: string }
 */

// ⚠️ Must be set BEFORE any InferenceSession is created.
// ORT 1.20 uses ort-wasm-simd-threaded.mjs (worker shim) + ort-wasm-simd-threaded.wasm
// (binary). Without this, ORT tries to infer the path from its own bundled script URL,
// which is wrong inside a Vite worker chunk and causes silent WASM load failure.
//
// Use absolute URL to prevent Vite 5 from intercepting as a static-analysis import.
// ORT 1.20 dynamically imports runtime .mjs files (e.g., .jsep.mjs for WebGPU probing).
// Vite 5 intercepts relative paths during bundling; absolute origin URLs bypass this.
ort.env.wasm.wasmPaths = `${self.location.origin}/ort-wasm/`;

// Run WASM single-threaded: we are already inside a dedicated worker, so spinning up
// a nested proxy worker is unnecessary overhead. numThreads=1 also avoids the
// SharedArrayBuffer requirement (COOP/COEP headers) that not every dev env satisfies.
ort.env.wasm.numThreads = 1;

// Disable JSEP (WebGPU) variant selection. ORT 1.20 auto-selects ort-wasm-simd-threaded.jsep.mjs
// if it detects WebGPU capability, even when executionProviders: ['wasm'] is specified.
// The JSEP variant has compatibility issues with INT8 quantized models and can produce
// incorrect output. Explicitly disable proxy mode to force ORT to use the standard
// ort-wasm-simd-threaded.{mjs,wasm} variant instead.
ort.env.wasm.proxy = false;

let session: ort.InferenceSession | null = null;

self.onmessage = async (event: MessageEvent) => {
  const { type } = event.data;

  try {
    if (type === 'load') {
      const { modelUrl } = event.data;
      await loadModel(modelUrl);
    } else if (type === 'infer') {
      const { styleGlyphs, charIndex, requestId } = event.data;
      const output = await runInference(styleGlyphs, charIndex);
      self.postMessage({ type: 'result', output, requestId });
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    self.postMessage({ 
      type: 'error', 
      message, 
      requestId: event.data.requestId 
    });
  }
};

async function loadModel(modelUrl: string): Promise<void> {
  const response = await fetch(modelUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch model: ${response.statusText}`);
  }

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

  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }

  // INT8 quantized model requires WASM — WebGL does not support QLinear ops
  // and silently produces all-background output when mixed with WASM fallback.
  session = await ort.InferenceSession.create(buffer.buffer, {
    executionProviders: ['wasm'],
  });

  // DEBUG: log actual model input/output names so we can verify key correctness
  console.debug('[inferenceWorker] session.inputNames:', session.inputNames);
  console.debug('[inferenceWorker] session.outputNames:', session.outputNames);

  self.postMessage({ type: 'loaded' });
}

async function runInference(
  styleGlyphs: Float32Array,
  charIndex: number
): Promise<Float32Array> {
  if (!session) {
    throw new Error('Model not loaded');
  }

  // Input tensors per contract from decisions.md:
  // - style_glyphs: [B, 10, 1, 128, 128] float32
  // - char_index: [B] int64
  // Output: generated_glyph [B, 1, 128, 128] float32, range [-1, 1]

  // Guard: ORT WASM cannot accept SharedArrayBuffer-backed views directly.
  // Copy to a plain ArrayBuffer if needed (mirrors OnnxInference.ts).
  // Note: typeof guard is required because SharedArrayBuffer is undefined in non-cross-origin-isolated
  // contexts (no COOP/COEP headers), and `instanceof` on an undefined constructor throws a ReferenceError.
  const safeStyleGlyphs = typeof SharedArrayBuffer !== 'undefined' && styleGlyphs.buffer instanceof SharedArrayBuffer
    ? new Float32Array(styleGlyphs.buffer.slice(
        styleGlyphs.byteOffset,
        styleGlyphs.byteOffset + styleGlyphs.byteLength,
      ))
    : styleGlyphs;

  const styleTensor = new ort.Tensor('float32', safeStyleGlyphs, [1, 10, 1, 128, 128]);
  const indexTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(charIndex)]), [1]);

  // DEBUG: verify inputs vary between font loads and that char_index is correct
  console.debug('[inferenceWorker] char_index:', charIndex);
  console.debug('[inferenceWorker] style_glyphs first 5 values:', Array.from(safeStyleGlyphs.slice(0, 5)));
  console.debug('[inferenceWorker] style_glyphs tensor shape:', styleTensor.dims);
  console.debug('[inferenceWorker] char_index tensor shape:', indexTensor.dims, 'dtype:', indexTensor.type);

  const feeds: Record<string, ort.Tensor> = {
    style_glyphs: styleTensor,
    char_index: indexTensor,
  };

  const results = await session.run(feeds);

  // Output tensor name should be 'generated_glyph' per contract
  const outputTensor = results['generated_glyph'] ?? results['output'] ?? Object.values(results)[0];
  const outputData = outputTensor.data as Float32Array;

  // DEBUG: verify raw output varies between inferences
  console.debug('[inferenceWorker] outputTensor.name/key resolved:', Object.keys(results)[0]);
  console.debug('[inferenceWorker] outputTensor first 5 raw values:', Array.from(outputData.slice(0, 5)));

  // Blank-output detection: if all sampled pixels are at or near -1.0 (background),
  // the model ran but produced no ink. Common causes: wrong WASM backend (WebGL + INT8
  // silently returns all-background), or stale pre-fix model weights.
  const sampleSize = Math.min(outputData.length, 512);
  let maxVal = -Infinity;
  for (let i = 0; i < sampleSize; i++) maxVal = Math.max(maxVal, outputData[i]);
  if (maxVal <= 0.0) {
    console.warn(
      `[inferenceWorker] ⚠️ Blank output for char_index=${charIndex}: ` +
      `max(first ${sampleSize} px) = ${maxVal.toFixed(4)}. ` +
      'Possible cause: wrong WASM files, WebGL+INT8 fallback, or pre-fix model.',
    );
  }
  console.debug('[inferenceWorker] output max (first 512 px):', maxVal.toFixed(4));

  // ⚠️ Critical: copy output before returning.
  //
  // ORT's WASM backend returns a Float32Array that is a *view* into
  // WebAssembly.Memory. On cross-origin-isolated pages SharedArrayBuffer is
  // used for WASM memory; postMessage does NOT clone SAB views — the receiver
  // gets an alias into the same physical memory. ORT reuses its output buffer
  // between session.run() calls, so without an explicit copy every result stored
  // in App.tsx rawGlyphs would reflect only the *last* inference, making all 66
  // assembled glyphs identical regardless of char_index or style input.
  return new Float32Array(outputData);
}
