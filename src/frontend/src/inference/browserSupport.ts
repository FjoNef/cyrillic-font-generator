/**
 * Browser capability detection for ONNX Runtime Web inference.
 *
 * Checks for the minimum requirements needed to run the Cyrillic Font Generator:
 *   - WebAssembly (required for WASM backend fallback)
 *   - SharedArrayBuffer (required for multi-threaded WASM)
 *   - Web Workers (required for non-blocking model load & inference)
 *   - WebGL (optional; enables faster inference via GPU)
 *
 * Execution provider selection: WebGL → WASM (4-thread) → WASM (single-thread)
 */

export interface BrowserSupportResult {
  /** Browser meets minimum requirements (WASM + Workers). */
  supported: boolean;
  /** Human-readable reason when supported is false. */
  reason?: string;
  /** WebAssembly is available. */
  hasWasm: boolean;
  /** SharedArrayBuffer is available (enables multi-threaded WASM). */
  hasSharedArrayBuffer: boolean;
  /** Web Workers are available (required for non-blocking inference). */
  hasWorkers: boolean;
  /** WebGL is available (enables fast GPU inference). */
  hasWebGL: boolean;
  /** Recommended onnxruntime-web execution providers in priority order. */
  executionProviders: Array<'webgl' | 'wasm'>;
}

/**
 * Detect browser capabilities and return a support result.
 * This function is synchronous — call it once during app initialisation.
 */
export function detectBrowserSupport(): BrowserSupportResult {
  const hasWasm =
    typeof WebAssembly === 'object' && typeof WebAssembly.instantiate === 'function';

  const hasWorkers = typeof Worker !== 'undefined';

  const hasSharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined';

  let hasWebGL = false;
  try {
    const canvas = document.createElement('canvas');
    hasWebGL =
      canvas.getContext('webgl') !== null || canvas.getContext('webgl2') !== null;
  } catch {
    hasWebGL = false;
  }

  const supported = hasWasm && hasWorkers;
  const reason = !hasWasm
    ? 'WebAssembly is not supported in this browser. Please use a modern browser (Chrome 57+, Firefox 53+, Safari 11+).'
    : !hasWorkers
    ? 'Web Workers are not supported in this browser. Inference cannot run off the main thread.'
    : undefined;

  const executionProviders: Array<'webgl' | 'wasm'> = hasWebGL
    ? ['webgl', 'wasm']
    : ['wasm'];

  return {
    supported,
    reason,
    hasWasm,
    hasSharedArrayBuffer,
    hasWorkers,
    hasWebGL,
    executionProviders,
  };
}
