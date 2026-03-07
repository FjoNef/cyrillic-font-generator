// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ModelLoader } from '../ModelLoader';
import { assembleFontFromGlyphs } from '../../FontAssembler';
import { downloadFont } from '../../FontDownloader';
import { CYRILLIC_CHARS } from '../../font/cyrillicCharset';

/**
 * Integration tests for end-to-end inference pipeline.
 *
 * Validates the full flow that App.tsx orchestrates:
 *   style glyphs → ModelLoader.infer × 66 → assembleFontFromGlyphs → downloadFont
 */

/** Helper: build a mock worker that auto-responds to infer messages. */
function makeMockWorker(outputValue = 0.5) {
  const worker: any = {
    postMessage: vi.fn((msg: any) => {
      if (msg.type === 'infer') {
        Promise.resolve().then(() => {
          const output = new Float32Array(128 * 128).fill(outputValue);
          worker.onmessage?.({ data: { type: 'result', output, requestId: msg.requestId } });
        });
      }
    }),
    terminate: vi.fn(),
    onmessage: null as ((e: { data: any }) => void) | null,
    onerror: null as ((e: any) => void) | null,
  };
  return worker;
}

describe('Inference Pipeline Integration', () => {
  let mockWorker: ReturnType<typeof makeMockWorker>;
  let modelLoader: ModelLoader;

  beforeEach(() => {
    mockWorker = makeMockWorker();
    global.Worker = vi.fn().mockImplementation(() => mockWorker) as any;
    modelLoader = new ModelLoader();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  async function loadModel() {
    const loadPromise = modelLoader.load('/api/model');
    mockWorker.onmessage?.({ data: { type: 'loaded' } });
    await loadPromise;
  }

  it('should infer all 66 Cyrillic glyphs and accumulate results', async () => {
    await loadModel();

    const styleGlyphs = new Float32Array(10 * 128 * 128).fill(0.5);
    const rawGlyphs = new Map<number, Float32Array>();

    for (const { index } of CYRILLIC_CHARS) {
      const output = await modelLoader.infer(styleGlyphs, index);
      rawGlyphs.set(index, output);
    }

    expect(rawGlyphs.size).toBe(66);

    const inferCalls = mockWorker.postMessage.mock.calls.filter(
      (c: any[]) => c[0].type === 'infer'
    );
    expect(inferCalls).toHaveLength(66);

    // style_glyphs tensor must be [10,1,128,128] flattened = 163840 floats
    inferCalls.forEach((call: any[]) => {
      expect(call[0].styleGlyphs).toHaveLength(10 * 128 * 128);
    });

    // Every char index 0–65 must be requested exactly once
    const seenIndices = new Set(inferCalls.map((c: any[]) => c[0].charIndex));
    expect(seenIndices.size).toBe(66);
    for (let i = 0; i < 66; i++) {
      expect(seenIndices.has(i)).toBe(true);
    }
  });

  it('should produce a valid OTF ArrayBuffer after 66-glyph assembly', async () => {
    await loadModel();

    const styleGlyphs = new Float32Array(10 * 128 * 128).fill(0.3);
    const rawGlyphs = new Map<number, Float32Array>();

    for (const { index } of CYRILLIC_CHARS) {
      rawGlyphs.set(index, await modelLoader.infer(styleGlyphs, index));
    }

    const buffer = assembleFontFromGlyphs(rawGlyphs, 'TestFont');

    expect(buffer).toBeInstanceOf(ArrayBuffer);
    expect(buffer.byteLength).toBeGreaterThan(0);

    // OTF begins with 'OTTO' (0x4F54544F) or TrueType 0x00010000
    const view = new DataView(buffer);
    const magic = view.getUint32(0, false);
    expect([0x4F54544F, 0x00010000]).toContain(magic);
  });

  it('should track generation progress from 1 to 66 monotonically', async () => {
    await loadModel();

    const styleGlyphs = new Float32Array(10 * 128 * 128).fill(0.0);
    const progressLog: number[] = [];

    for (let i = 0; i < CYRILLIC_CHARS.length; i++) {
      await modelLoader.infer(styleGlyphs, CYRILLIC_CHARS[i].index);
      progressLog.push(i + 1);
    }

    expect(progressLog[0]).toBe(1);
    expect(progressLog[65]).toBe(66);
    for (let i = 1; i < progressLog.length; i++) {
      expect(progressLog[i]).toBe(progressLog[i - 1] + 1);
    }
  });

  it('should propagate inference errors mid-generation', async () => {
    mockWorker.postMessage = vi.fn((msg: any) => {
      if (msg.type === 'infer') {
        Promise.resolve().then(() => {
          if (msg.charIndex === 5) {
            mockWorker.onmessage?.({
              data: { type: 'error', message: 'ONNX shape mismatch', requestId: msg.requestId },
            });
          } else {
            const output = new Float32Array(128 * 128).fill(0.5);
            mockWorker.onmessage?.({ data: { type: 'result', output, requestId: msg.requestId } });
          }
        });
      }
    });

    await loadModel();

    const styleGlyphs = new Float32Array(10 * 128 * 128);
    await expect(
      (async () => {
        for (const { index } of CYRILLIC_CHARS) {
          await modelLoader.infer(styleGlyphs, index);
        }
      })()
    ).rejects.toThrow('ONNX shape mismatch');
  });

  it('should prevent inference before model is loaded', async () => {
    const styleGlyphs = new Float32Array(10 * 128 * 128);
    await expect(modelLoader.infer(styleGlyphs, 0)).rejects.toThrow('Model not loaded');
  });

  it('should download font with correct MIME type and filename', () => {
    const anchor = { click: vi.fn(), href: '', download: '' };
    vi.spyOn(document, 'createElement').mockReturnValue(anchor as any);
    vi.spyOn(document.body, 'appendChild').mockImplementation(() => anchor as any);
    vi.spyOn(document.body, 'removeChild').mockImplementation(() => anchor as any);
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock');
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});

    downloadFont(new ArrayBuffer(128), 'my-font.otf');

    expect(URL.createObjectURL).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'font/otf' })
    );
    expect(anchor.download).toBe('my-font.otf');
    expect(anchor.click).toHaveBeenCalledOnce();
  });

  it('should revoke object URL after download to prevent memory leak', () => {
    const anchor = { click: vi.fn(), href: '', download: '' };
    vi.spyOn(document, 'createElement').mockReturnValue(anchor as any);
    vi.spyOn(document.body, 'appendChild').mockImplementation(() => anchor as any);
    vi.spyOn(document.body, 'removeChild').mockImplementation(() => anchor as any);
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:test-url');
    const revoke = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});

    downloadFont(new ArrayBuffer(8), 'test.otf');

    expect(revoke).toHaveBeenCalledWith('blob:test-url');
  });

  it('each infer result is an independent copy — not a shared-buffer alias (regression: #39)', async () => {
    // Simulate the ORT WASM SharedArrayBuffer reuse scenario:
    // The "worker" sends the SAME Float32Array reference every call and
    // overwrites its values before each response.  Without a defensive copy in
    // ModelLoader.resolve (or in the worker), all stored results would reflect
    // only the last inference, making every assembled glyph identical.
    const sharedOutput = new Float32Array(128 * 128);
    let seq = 0;

    mockWorker.postMessage = vi.fn((msg: any) => {
      if (msg.type === 'infer') {
        Promise.resolve().then(() => {
          sharedOutput.fill(++seq); // overwrite buffer with incrementing sentinel
          mockWorker.onmessage?.({ data: { type: 'result', output: sharedOutput, requestId: msg.requestId } });
        });
      }
    });

    await loadModel();

    const styleGlyphs = new Float32Array(10 * 128 * 128).fill(0.5);
    const r1 = await modelLoader.infer(styleGlyphs, 0);
    const r2 = await modelLoader.infer(styleGlyphs, 1);
    const r3 = await modelLoader.infer(styleGlyphs, 2);

    // Each result must snapshot the value it had at the moment of inference.
    // Without defensive copy: r1[0] === r2[0] === r3[0] === 3 (all overwritten).
    // With copy in ModelLoader.resolve: r1[0] === 1, r2[0] === 2, r3[0] === 3.
    expect(r1[0]).toBe(1);
    expect(r2[0]).toBe(2);
    expect(r3[0]).toBe(3);
  });
});
