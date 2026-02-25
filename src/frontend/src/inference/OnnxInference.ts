import * as ort from 'onnxruntime-web';

/**
 * Thin wrapper around an ONNX Runtime Web InferenceSession.
 * Tensor shapes are placeholders — Major will provide exact contract.
 */
export class OnnxInference {
  private session: ort.InferenceSession | null = null;

  /**
   * Fetch the model with progress tracking, then create an InferenceSession.
   * @param modelUrl  URL to the .onnx model file (served by Vite or ASP.NET backend)
   * @param onProgress  Optional callback receiving load percentage 0-100
   */
  async loadModel(
    modelUrl: string,
    onProgress?: (pct: number) => void
  ): Promise<void> {
    // TODO: fetch with ReadableStream progress, then pass ArrayBuffer to ort
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

    this.session = await ort.InferenceSession.create(buffer.buffer, {
      executionProviders: ['webgl', 'wasm'],
    });

    onProgress?.(100);
  }

  /**
   * Run inference for a single Cyrillic glyph.
   *
   * @param styleGlyphs  Flattened [10 × 1 × 128 × 128] Float32Array of Latin reference glyphs
   * @param charIndex    0-65: index into CYRILLIC_CHARS (0-32 upper, 33-65 lower)
   * @returns            128×128 ImageData of the generated glyph (grayscale, RGBA)
   *
   * TODO: confirm input/output tensor names and shapes with Major.
   */
  async generateGlyph(
    styleGlyphs: Float32Array,
    charIndex: number
  ): Promise<ImageData> {
    if (!this.session) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    // TODO: replace tensor names / shapes once Major publishes the ONNX contract
    const styleTensor = new ort.Tensor('float32', styleGlyphs, [10, 1, 128, 128]);
    const indexTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(charIndex)]), [1]);

    const feeds: Record<string, ort.Tensor> = {
      style_glyphs: styleTensor,  // TODO: confirm input name with Major
      char_index: indexTensor,    // TODO: confirm input name with Major
    };

    const results = await this.session.run(feeds);

    // TODO: confirm output tensor name with Major
    const outputTensor = results['output'] ?? Object.values(results)[0];
    const outputData = outputTensor.data as Float32Array;

    // Convert single-channel float [-1,1] → RGBA ImageData
    const size = 128;
    const pixels = new Uint8ClampedArray(size * size * 4);
    for (let i = 0; i < size * size; i++) {
      const val = Math.round(((outputData[i] + 1) / 2) * 255); // [-1,1] → [0,255]
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
