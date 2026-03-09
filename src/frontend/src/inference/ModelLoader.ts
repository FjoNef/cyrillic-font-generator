/**
 * ModelLoader — Singleton wrapper around the inference Web Worker.
 * 
 * Provides promise-based API for model loading and inference.
 * Automatically manages worker lifecycle and request IDs for concurrent safety.
 */

type WorkerMessage = 
  | { type: 'progress'; progress: number }
  | { type: 'loaded' }
  | { type: 'result'; output: Float32Array; requestId: string }
  | { type: 'error'; message: string; requestId?: string };

interface PendingRequest {
  resolve: (output: Float32Array) => void;
  reject: (error: Error) => void;
}

export class ModelLoader {
  private worker: Worker | null = null;
  private loadPromise: Promise<void> | null = null;
  private onProgress: ((pct: number) => void) | null = null;
  private pendingRequests = new Map<string, PendingRequest>();
  private requestCounter = 0;

  /**
   * Load the ONNX model from the given URL.
   * Progress updates are reported via onProgress callback.
   * Safe to call multiple times — returns same promise if already loading.
   */
  load(modelUrl: string, onProgress?: (pct: number) => void): Promise<void> {
    if (this.loadPromise) {
      return this.loadPromise;
    }

    this.onProgress = onProgress ?? null;

    this.loadPromise = new Promise<void>((resolve, reject) => {
      // Create worker using Vite's Worker constructor with module syntax
      this.worker = new Worker(
        new URL('./worker/inferenceWorker.ts', import.meta.url),
        { type: 'module' }
      );

      this.worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
        const msg = event.data;

        if (msg.type === 'progress') {
          this.onProgress?.(msg.progress);
        } else if (msg.type === 'loaded') {
          resolve();
        } else if (msg.type === 'result') {
          const pending = this.pendingRequests.get(msg.requestId);
          if (pending) {
            // Defensive copy: ORT's WASM backend may return a Float32Array that
            // views SharedArrayBuffer (WASM memory). postMessage does not clone
            // SAB — the received Float32Array would alias the same WASM memory
            // and would be overwritten by the next inference.  Copying here
            // guarantees an independent buffer regardless of SAB availability.
            pending.resolve(new Float32Array(msg.output));
            this.pendingRequests.delete(msg.requestId);
          }
        } else if (msg.type === 'error') {
          if (msg.requestId) {
            const pending = this.pendingRequests.get(msg.requestId);
            if (pending) {
              pending.reject(new Error(msg.message));
              this.pendingRequests.delete(msg.requestId);
            }
          } else {
            // Load error
            reject(new Error(msg.message));
          }
        }
      };

      this.worker.onerror = (err) => {
        // ErrorEvent has message, filename, lineno, colno
        // Plain Event (when worker script fails to load) has only type
        const details = err instanceof ErrorEvent
          ? `${err.message} (${err.filename}:${err.lineno}:${err.colno})`
          : `[${err.type}] script load failure - check console`;
        console.error('[ModelLoader] Worker onerror:', {
          type: err.type,
          message: (err as ErrorEvent).message,
          filename: (err as ErrorEvent).filename,
          lineno: (err as ErrorEvent).lineno,
          colno: (err as ErrorEvent).colno,
        });
        reject(new Error(`Worker error: ${details}`));
      };

      // Trigger load
      this.worker.postMessage({ type: 'load', modelUrl });
    });

    return this.loadPromise;
  }

  /**
   * Run inference for a single Cyrillic glyph.
   * 
   * @param styleGlyphs Flattened [10, 1, 128, 128] = 163840 floats
   * @param charIndex   0-65 (Russian Cyrillic char index)
   * @returns           128×128 = 16384 floats, range [-1, 1]
   */
  async infer(styleGlyphs: Float32Array, charIndex: number): Promise<Float32Array> {
    if (!this.worker || !this.loadPromise) {
      throw new Error('Model not loaded. Call load() first.');
    }

    await this.loadPromise;

    const requestId = `req-${++this.requestCounter}`;

    return new Promise<Float32Array>((resolve, reject) => {
      this.pendingRequests.set(requestId, { resolve, reject });

      this.worker!.postMessage({
        type: 'infer',
        styleGlyphs,
        charIndex,
        requestId,
      });
    });
  }

  /**
   * Check if the model is loaded and ready for inference.
   */
  isReady(): boolean {
    return this.loadPromise !== null && this.worker !== null;
  }
}

export const modelLoader = new ModelLoader();
