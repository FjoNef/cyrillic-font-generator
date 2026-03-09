// @vitest-environment jsdom

/**
 * Style Conditioning Tests
 *
 * Verifies that OnnxInference passes the style_glyphs tensor correctly to the
 * ONNX session, and that different style inputs reach the model as distinct
 * tensor values.
 *
 * ⚠️ BUG DETECTION: If style_glyphs is hardcoded, dropped, or replaced with a
 *    constant, the two session.run() calls will receive identical tensor data —
 *    and the assertions below will FAIL.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// ── ONNX runtime mock ─────────────────────────────────────────────────────────

vi.mock('onnxruntime-web/wasm', () => {
  class Tensor {
    type: string;
    data: any;
    dims: number[];
    constructor(type: string, data: any, dims: number[]) {
      this.type = type;
      this.data = data;
      this.dims = dims;
    }
  }

  const InferenceSession = {
    create: vi.fn(),
  };

  return { Tensor, InferenceSession };
});

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Builds a style glyphs buffer filled with a given value. */
function makeStyleGlyphs(fillValue: number): Float32Array {
  return new Float32Array(10 * 1 * 128 * 128).fill(fillValue);
}

/**
 * Creates a minimal mock ONNX session whose run() records every call and
 * returns a plausible generated_glyph output.
 */
function makeMockSession() {
  const runCalls: Array<Record<string, any>> = [];

  const mockRun = vi.fn().mockImplementation(async (feeds: Record<string, any>) => {
    runCalls.push({ ...feeds });
    return {
      generated_glyph: {
        data: new Float32Array(1 * 1 * 128 * 128).fill(0.0),
        dims: [1, 1, 128, 128],
        type: 'float32',
      },
    };
  });

  return {
    session: { run: mockRun, inputNames: ['style_glyphs', 'char_index'], outputNames: ['generated_glyph'] },
    runCalls,
    mockRun,
  };
}

/** Loads the mocked OnnxInference with an injected session via fake fetch. */
async function buildLoadedInference(sessionMock: ReturnType<typeof makeMockSession>['session']) {
  const ort = await import('onnxruntime-web/wasm');

  // Stub out fetch so loadModel can stream a buffer.
  const fakeChunk = new Uint8Array(16).fill(1);
  const mockReader = {
    read: vi.fn()
      .mockResolvedValueOnce({ done: false, value: fakeChunk })
      .mockResolvedValueOnce({ done: true }),
  };
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    headers: { get: () => String(fakeChunk.length) },
    body: { getReader: () => mockReader },
  });

  vi.mocked(ort.InferenceSession.create).mockResolvedValueOnce(sessionMock as any);

  const { OnnxInference } = await import('../OnnxInference');
  const inference = new OnnxInference();
  await inference.loadModel('/mock/model.onnx');
  return inference;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('Style Conditioning — session receives style_glyphs', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('session.run is called with a tensor named "style_glyphs"', async () => {
    const { session, mockRun } = makeMockSession();
    const inference = await buildLoadedInference(session);

    const styleGlyphs = makeStyleGlyphs(0.5);
    await inference.generateGlyph(styleGlyphs, 0);

    expect(mockRun).toHaveBeenCalledOnce();
    const feeds = mockRun.mock.calls[0][0];
    expect(feeds).toHaveProperty('style_glyphs');
  });

  it('style_glyphs tensor has shape [1, 10, 1, 128, 128]', async () => {
    const { session, mockRun } = makeMockSession();
    const inference = await buildLoadedInference(session);

    await inference.generateGlyph(makeStyleGlyphs(0.0), 0);

    const feeds = mockRun.mock.calls[0][0];
    expect(feeds.style_glyphs.dims).toEqual([1, 10, 1, 128, 128]);
  });

  it('style_glyphs tensor type is float32', async () => {
    const { session, mockRun } = makeMockSession();
    const inference = await buildLoadedInference(session);

    await inference.generateGlyph(makeStyleGlyphs(0.0), 0);

    const feeds = mockRun.mock.calls[0][0];
    expect(feeds.style_glyphs.type).toBe('float32');
  });

  it('different style inputs reach the session as different tensor data', async () => {
    const { session, mockRun } = makeMockSession();
    const inference = await buildLoadedInference(session);

    // Call 1: style from font A (all 0.8 — strong ink)
    await inference.generateGlyph(makeStyleGlyphs(0.8), 0);
    // Call 2: style from font B (all -0.8 — strong background)
    await inference.generateGlyph(makeStyleGlyphs(-0.8), 0);

    expect(mockRun).toHaveBeenCalledTimes(2);

    const feedsA = mockRun.mock.calls[0][0];
    const feedsB = mockRun.mock.calls[1][0];

    const dataA = feedsA.style_glyphs.data as Float32Array;
    const dataB = feedsB.style_glyphs.data as Float32Array;

    // ⚠️ KEY BUG DETECTION: if style is ignored (hardcoded), these will be equal.
    const areIdentical = dataA.every((v: number, i: number) => v === dataB[i]);
    expect(areIdentical).toBe(false);
  });

  it('style_glyphs tensor data preserves the values passed by the caller', async () => {
    const { session, mockRun } = makeMockSession();
    const inference = await buildLoadedInference(session);

    const inputGlyphs = makeStyleGlyphs(0.42);
    await inference.generateGlyph(inputGlyphs, 0);

    const feeds = mockRun.mock.calls[0][0];
    const sentData = feeds.style_glyphs.data as Float32Array;

    // Every element must match the original input exactly.
    for (let i = 0; i < inputGlyphs.length; i++) {
      expect(sentData[i]).toBe(inputGlyphs[i]);
    }
  });

  it('session also receives char_index alongside style_glyphs', async () => {
    const { session, mockRun } = makeMockSession();
    const inference = await buildLoadedInference(session);

    await inference.generateGlyph(makeStyleGlyphs(0.0), 33);

    const feeds = mockRun.mock.calls[0][0];
    expect(feeds).toHaveProperty('char_index');
    expect(feeds.char_index.data[0]).toBe(BigInt(33));
  });
});
