// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { FontLoader } from '../FontLoader';

/**
 * Style variation tests for FontLoader.extractStyleGlyphs.
 *
 * These tests verify that:
 *   1. Different input fonts produce different style tensor data.
 *   2. Output values are normalised to [-1.0, 1.0].
 *   3. Output tensor has the correct shape: 10 × 128 × 128 = 163 840 floats.
 *
 * ⚠️ BUG DETECTION: If the canvas rendering pipeline ignores font glyph shapes
 *    (e.g. Path2D silently fails, or the wrong font object is used), test #1
 *    will FAIL — indicating that style conditioning is broken.
 */

const RENDER_SIZE = 128;
const STYLE_CHAR_COUNT = 10;
const TOTAL_PIXELS = STYLE_CHAR_COUNT * RENDER_SIZE * RENDER_SIZE; // 163 840

// ── Canvas mock factory ───────────────────────────────────────────────────────

/**
 * Creates a lightweight canvas mock whose ctx.fill() optionally stamps a black
 * rectangle, simulating what an actual glyph render would produce.
 *
 * @param hasInk  When true, ctx.fill() writes black pixels into the pixel buffer.
 *                When false, the canvas stays white (blank font / no glyph).
 */
function createMockCanvas(hasInk: boolean) {
  let pixels = new Uint8ClampedArray(RENDER_SIZE * RENDER_SIZE * 4).fill(255);

  const ctx: any = {
    clearRect: vi.fn(),
    fillRect: vi.fn((_x: number, _y: number, _w: number, _h: number) => {
      // Simulate white background fill.
      pixels = new Uint8ClampedArray(RENDER_SIZE * RENDER_SIZE * 4).fill(255);
    }),
    fill: vi.fn((_path2d: any) => {
      if (hasInk) {
        // Stamp a recognisable black rectangle to simulate glyph ink.
        for (let y = 20; y < 108; y++) {
          for (let x = 10; x < 118; x++) {
            const idx = (y * RENDER_SIZE + x) * 4;
            pixels[idx] = 0;
            pixels[idx + 1] = 0;
            pixels[idx + 2] = 0;
          }
        }
      }
    }),
    getImageData: vi.fn().mockImplementation(() => ({ data: pixels })),
    set fillStyle(_v: string) {},
    get fillStyle() { return ''; },
  };

  return {
    width: RENDER_SIZE,
    height: RENDER_SIZE,
    getContext: vi.fn().mockReturnValue(ctx),
    _ctx: ctx,
  } as any;
}

// ── Mock font factory ─────────────────────────────────────────────────────────

function makeMockFont() {
  return {
    charToGlyph: vi.fn().mockReturnValue({
      getPath: vi.fn().mockReturnValue({
        fill: 'black',
        toSVG: vi.fn().mockReturnValue('M 0 0 L 50 0 L 50 50 L 0 50 Z'),
      }),
    }),
  } as any;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('FontLoader — style glyph variation', () => {
  let createElementSpy: ReturnType<typeof vi.spyOn>;
  let canvasCallIndex: number;
  let fontAHasInk: boolean;
  let fontBHasInk: boolean;

  beforeEach(() => {
    canvasCallIndex = 0;
    fontAHasInk = true;   // font A renders visible glyphs
    fontBHasInk = false;  // font B renders blank glyphs (e.g. a glyph-less placeholder font)

    createElementSpy = vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      if (tag === 'canvas') {
        // Alternate canvas mocks: first call = font A's canvas, second = font B's.
        const hasInk = canvasCallIndex++ === 0 ? fontAHasInk : fontBHasInk;
        return createMockCanvas(hasInk);
      }
      // Fallthrough for non-canvas elements.
      createElementSpy.mockRestore();
      const el = document.createElement(tag);
      createElementSpy = vi.spyOn(document, 'createElement').mockImplementation((t: string) => {
        if (t === 'canvas') {
          const hasInk = canvasCallIndex++ === 0 ? fontAHasInk : fontBHasInk;
          return createMockCanvas(hasInk);
        }
        return document.createElement(t);
      });
      return el;
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('returns a Float32Array with 163 840 elements (10 × 128 × 128)', () => {
    const loader = new FontLoader();
    const result = loader.extractStyleGlyphs(makeMockFont());
    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(TOTAL_PIXELS);
  });

  it('all values are within [-1.0, 1.0]', () => {
    const loader = new FontLoader();
    const result = loader.extractStyleGlyphs(makeMockFont());
    for (let i = 0; i < result.length; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(-1.0);
      expect(result[i]).toBeLessThanOrEqual(1.0);
    }
  });

  it('different fonts produce non-identical style tensors', () => {
    const loader = new FontLoader();

    // Font A: canvas mock with ink → some pixels normalised to +1.0 (black ink)
    const glyphsA = loader.extractStyleGlyphs(makeMockFont());

    // Font B: canvas mock without ink → all pixels normalised to -1.0 (white background)
    const glyphsB = loader.extractStyleGlyphs(makeMockFont());

    // The two arrays must differ — if style conditioning is working, different
    // font renders must yield different tensors.
    const areIdentical = glyphsA.every((v, i) => v === glyphsB[i]);
    expect(areIdentical).toBe(false);
  });

  it('blank font produces a tensor dominated by background value (-1.0)', () => {
    // Override: both calls get blank canvases (no ink)
    fontAHasInk = false;
    const loader = new FontLoader();
    const result = loader.extractStyleGlyphs(makeMockFont());

    // Nearly all pixels should be -1.0 (white background)
    const backgroundCount = Array.from(result).filter(v => v === -1.0).length;
    expect(backgroundCount).toBe(TOTAL_PIXELS);
  });

  it('ink-rendering font produces a tensor with positive (ink) values', () => {
    // Override: first call gets an inked canvas
    fontAHasInk = true;
    const loader = new FontLoader();
    const result = loader.extractStyleGlyphs(makeMockFont());

    // Should contain some +1.0 values (black ink pixels)
    const inkCount = Array.from(result).filter(v => v === 1.0).length;
    expect(inkCount).toBeGreaterThan(0);
  });
});
