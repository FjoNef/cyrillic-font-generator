// ⚠️ Import from 'onnxruntime-web/wasm' — NOT 'onnxruntime-web'.
// The default bundle loads the JSEP (WebGPU) WASM variant which silently produces
// blank output for INT8 QLinear ops. The /wasm sub-path uses the standard non-JSEP
// variant that correctly handles INT8 quantized models.
import * as ort from 'onnxruntime-web/wasm';

/**
 * Thin wrapper around an ONNX Runtime Web InferenceSession.
 *
 * Inference contract (v1, models/v1/generator.onnx):
 *   Inputs:
 *     style_glyphs  [1, 10, 1, 128, 128]  float32  in [-1, 1]
 *                   10 Latin reference glyphs (A B C D E H I O R X),
 *                   rendered white-on-black then normalised: black bg → -1, white glyph → +1
 *     char_index    [1]                   int64    in [0, 65]
 *   Output:
 *     generated_glyph  [1, 1, 128, 128]  float32  in [-1, 1]
 *                      +1 = glyph ink, -1 = background
 *
 * Execution providers: WASM only (WebGL does not support INT8 quantized models).
 */
export class OnnxInference {
  private session: ort.InferenceSession | null = null;

  /**
   * Fetch the model with progress tracking, then create an InferenceSession.
   * Uses WASM backend exclusively — INT8 quantized models require WASM; WebGL
   * does not support QLinear operations and silently produces all-background output.
   *
   * @param modelUrl    URL to the .onnx model file
   * @param onProgress  Optional callback receiving load percentage 0–100
   */
  async loadModel(
    modelUrl: string,
    onProgress?: (pct: number) => void
  ): Promise<void> {
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
      if (contentLength > 0 && onProgress) {
        onProgress(Math.round((received / contentLength) * 100));
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
    this.session = await ort.InferenceSession.create(buffer.buffer, {
      executionProviders: ['wasm'],
    });

    onProgress?.(100);
  }

  /**
   * Run inference for a single Cyrillic glyph.
   *
   * @param styleGlyphs  Flattened [10 × 1 × 128 × 128] = 163840 float32 values in [-1, 1]
   *                     Latin reference glyphs (A B C D E H I O R X)
   * @param charIndex    0–65 Cyrillic character index (0=А … 32=Я, 33=а … 65=я)
   * @returns            128×128 RGBA ImageData for display (white background, black glyph)
   */
  async generateGlyph(
    styleGlyphs: Float32Array,
    charIndex: number
  ): Promise<ImageData> {
    if (!this.session) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    // style_glyphs: [1, 10, 1, 128, 128] — batch dimension required.
    // Guard: if styleGlyphs is backed by a SharedArrayBuffer (SAB), copy it to a plain
    // ArrayBuffer first — ORT's WASM backend cannot accept SAB-backed views directly.
    const safeStyleGlyphs = styleGlyphs.buffer instanceof SharedArrayBuffer
      ? new Float32Array(styleGlyphs.buffer.slice(styleGlyphs.byteOffset, styleGlyphs.byteOffset + styleGlyphs.byteLength))
      : styleGlyphs;
    const styleTensor = new ort.Tensor('float32', safeStyleGlyphs, [1, 10, 1, 128, 128]);
    const indexTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(charIndex)]), [1]);

    const results = await this.session.run({
      style_glyphs: styleTensor,
      char_index: indexTensor,
    });

    // Output tensor name per contract; fall back to first output if renamed
    const outputTensor = results['generated_glyph'] ?? Object.values(results)[0];
    const outputData = outputTensor.data as Float32Array;

    // Denormalise: +1 (glyph ink) → 0 (black pixel), -1 (background) → 255 (white pixel)
    const size = 128;
    const pixels = new Uint8ClampedArray(size * size * 4);
    for (let i = 0; i < size * size; i++) {
      const val = Math.max(0, Math.min(255, Math.round(((1 - outputData[i]) / 2) * 255)));
      pixels[i * 4 + 0] = val; // R
      pixels[i * 4 + 1] = val; // G
      pixels[i * 4 + 2] = val; // B
      pixels[i * 4 + 3] = 255; // A
    }

    return new ImageData(pixels, size, size);
  }

  isReady(): boolean {
    return this.session !== null;
  }
}
