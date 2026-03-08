// @vitest-environment jsdom

/**
 * fontPipeline.test.ts — Spec tests for the font assembly pipeline.
 *
 * Covers three modules Togusa is building:
 *   - GlyphVectorizer: Float32Array[16384] (128×128, [-1,1]) → opentype.Path (scanline rects)
 *   - FontAssembler:   66 Cyrillic glyph images → OTF ArrayBuffer via opentype.js
 *   - FontDownloader:  ArrayBuffer → browser file download
 *
 * Font metrics (per decisions.md, LOCKED):
 *   1000 UPM, ascender 800, descender -200, advance width 600
 *
 * Tensor convention (per decisions.md):
 *   +1.0 = black ink, -1.0 = white background
 *   Threshold: value > 0 → ink, value ≤ 0 → white
 *
 * These tests are the spec.  They will fail until Togusa provides the implementations.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as opentype from 'opentype.js';
import { vectorizeGlyph } from './GlyphVectorizer';
import { assembleFontFromGlyphs } from './FontAssembler';
import { downloadFont } from './FontDownloader';

/** PathCommand subtypes that carry X/Y coordinates (M and L). */
type XYCommand = opentype.PathCommand & { x: number; y: number };

// ─── shared constants ──────────────────────────────────────────────────────
const ASCENDER = 800;
const DESCENDER = -200;
const ADVANCE_WIDTH = 600;
const PIXELS = 128; // glyph image side length

// ─── helper factories ──────────────────────────────────────────────────────

/** All-white image: every pixel at -1.0 (background). */
function whiteImage(): Float32Array {
  return new Float32Array(PIXELS * PIXELS).fill(-1.0);
}

/** All-black image: every pixel at +1.0 (ink). */
function blackImage(): Float32Array {
  return new Float32Array(PIXELS * PIXELS).fill(1.0);
}

/** Image with a single pixel set to `value`, everything else white. */
function pixelAt(row: number, col: number, value: number = 1.0): Float32Array {
  const data = whiteImage();
  data[row * PIXELS + col] = value;
  return data;
}

/** Image with every pixel in `row` set to ink (+1.0). */
function filledRow(row: number): Float32Array {
  const data = whiteImage();
  for (let col = 0; col < PIXELS; col++) {
    data[row * PIXELS + col] = 1.0;
  }
  return data;
}

/** Image with ink pixels in `row` from `colStart` (inclusive) to `colEnd` (exclusive). */
function rowRun(row: number, colStart: number, colEnd: number): Float32Array {
  const data = whiteImage();
  for (let col = colStart; col < colEnd; col++) {
    data[row * PIXELS + col] = 1.0;
  }
  return data;
}

/** Build a Map of 66 Cyrillic glyph images (all white) for FontAssembler tests. */
function makeGlyphImages(fillValue = -1.0): Map<number, Float32Array> {
  const map = new Map<number, Float32Array>();
  for (let i = 0; i < 66; i++) {
    map.set(i, new Float32Array(16384).fill(fillValue));
  }
  return map;
}

// ─── GlyphVectorizer ──────────────────────────────────────────────────────

describe('GlyphVectorizer', () => {
  // ── test 1 ────────────────────────────────────────────────────────────────
  it('1. all-white input (all -1.0) → empty path with zero path commands', () => {
    const path = vectorizeGlyph(whiteImage());
    // No ink pixels → no rectangles → no moveTo/lineTo/close commands
    expect(path.commands.length).toBe(0);
  });

  // ── test 2 ────────────────────────────────────────────────────────────────
  it('2. all-black input (all +1.0) → non-empty path (full-coverage rectangles)', () => {
    const path = vectorizeGlyph(blackImage());
    // Every row has a full-width run → 128 rectangles × 5 commands = 640 commands
    // At minimum there must be more than zero commands
    expect(path.commands.length).toBeGreaterThan(0);
  });

  // ── test 3 ────────────────────────────────────────────────────────────────
  it('3. single horizontal run → exactly one rectangle at correct X range', () => {
    // Row 64, columns 10–19 (10-pixel wide run)
    const path = vectorizeGlyph(rowRun(64, 10, 20));

    // One rectangle = moveTo + 3×lineTo + close = 5 commands
    expect(path.commands.length).toBe(5);

    // X start = col 10 scaled to [0, ADVANCE_WIDTH]
    const xScale = ADVANCE_WIDTH / PIXELS; // 600/128 ≈ 4.6875
    const commands = path.commands as opentype.PathCommand[];
    const moveCmd = commands.find(c => c.type === 'M');
    expect(moveCmd).toBeDefined();
    expect((moveCmd as XYCommand).x).toBeCloseTo(10 * xScale, 1);
  });

  // ── test 4 ────────────────────────────────────────────────────────────────
  it('4. Y-axis flip: row 0 rectangle top ≈ ascender (800), row 127 rectangle bottom = descender (-200)', () => {
    // Y mapping (per font coordinate convention, Y increases upward):
    //   rect top    = ascender − row × yScale
    //   rect bottom = ascender − (row + 1) × yScale
    //   where yScale = (ascender − descender) / 128 = 1000/128

    // Row 0 — the top edge of the first-row rectangle should equal the ascender
    const topPath = vectorizeGlyph(pixelAt(0, 0));
    const topYValues = (topPath.commands as opentype.PathCommand[])
      .filter((c): c is XYCommand => c.type === 'M' || c.type === 'L')
      .map((c: XYCommand) => c.y);
    expect(Math.max(...topYValues)).toBeCloseTo(ASCENDER, 0);

    // Row 127 — the bottom edge of the last-row rectangle should equal the descender
    const bottomPath = vectorizeGlyph(pixelAt(127, 0));
    const bottomYValues = (bottomPath.commands as opentype.PathCommand[])
      .filter((c): c is XYCommand => c.type === 'M' || c.type === 'L')
      .map((c: XYCommand) => c.y);
    expect(Math.min(...bottomYValues)).toBeCloseTo(DESCENDER, 0);
  });

  // ── test 5 ────────────────────────────────────────────────────────────────
  it('5. X scaling: full 128-column row maps to x range [0, 600]', () => {
    const path = vectorizeGlyph(filledRow(64));
    const xValues = (path.commands as opentype.PathCommand[])
      .filter((c): c is XYCommand => c.type === 'M' || c.type === 'L')
      .map((c: XYCommand) => c.x);
    expect(Math.min(...xValues)).toBeCloseTo(0, 0);
    expect(Math.max(...xValues)).toBeCloseTo(ADVANCE_WIDTH, 0);
  });

  // ── test 6 ────────────────────────────────────────────────────────────────
  it('6. threshold: pixel at exactly 0.0 → white (no ink); pixel at +0.001 → ink', () => {
    // Threshold rule: value > 0 → ink, value ≤ 0 → white background
    const atThreshold = vectorizeGlyph(pixelAt(64, 64, 0.0));
    expect(atThreshold.commands.length).toBe(0);

    const justAboveThreshold = vectorizeGlyph(pixelAt(64, 64, 0.001));
    expect(justAboveThreshold.commands.length).toBeGreaterThan(0);
  });
});

// ─── FontAssembler ───────────────────────────────────────────────────────

describe('FontAssembler', () => {
  // ── test 7 ────────────────────────────────────────────────────────────────
  it('7. returns an ArrayBuffer with non-zero byte length', () => {
    const buffer = assembleFontFromGlyphs(makeGlyphImages(), null, 'TestFont');
    expect(buffer).toBeInstanceOf(ArrayBuffer);
    expect(buffer.byteLength).toBeGreaterThan(0);
  });

  // ── test 8 ────────────────────────────────────────────────────────────────
  it('8. font contains exactly 66 Cyrillic glyphs + .notdef (67 total)', () => {
    const buffer = assembleFontFromGlyphs(makeGlyphImages(), null, 'TestFont');
    const font = opentype.parse(buffer);

    // 66 Cyrillic + 1 .notdef = 67 glyphs
    expect(font.glyphs.length).toBe(67);
    // Glyph at slot 0 must be .notdef
    expect(font.glyphs.get(0).name).toBe('.notdef');
  });

  // ── test 9 ────────────────────────────────────────────────────────────────
  it('9. first Cyrillic glyph (index 0) maps to Unicode А (U+0410)', () => {
    const buffer = assembleFontFromGlyphs(makeGlyphImages(), null, 'TestFont');
    const font = opentype.parse(buffer);

    // The cmap must resolve А to a glyph slot other than .notdef (slot 0)
    const glyphIndex = font.charToGlyphIndex('А');
    expect(glyphIndex).toBeGreaterThan(0);

    // The glyph at that slot must have the correct Unicode value
    const glyph = font.glyphs.get(glyphIndex);
    expect(glyph.unicode).toBe(0x0410);
  });

  // ── test 10 ───────────────────────────────────────────────────────────────
  it('10. Cyrillic glyph at index 6 maps to Ё (U+0401)', () => {
    // Per decisions.md: uppercase ordering is А(0), Б(1), В(2), Г(3), Д(4), Е(5), Ё(6), Ж(7)…
    // Ё is inserted at alphabetical position 6 within the 66-glyph set.
    const buffer = assembleFontFromGlyphs(makeGlyphImages(), null, 'TestFont');
    const font = opentype.parse(buffer);

    // cmap must include Ё
    const yoGlyphIndex = font.charToGlyphIndex('Ё');
    expect(yoGlyphIndex).toBeGreaterThan(0);

    // The glyph at that slot must carry unicode U+0401
    const yoGlyph = font.glyphs.get(yoGlyphIndex);
    expect(yoGlyph.unicode).toBe(0x0401);

    // Ё must sit at Cyrillic position 6 in the glyph array (slot 7 accounting for .notdef)
    expect(yoGlyphIndex).toBe(7);
  });

  // ── test 11 ───────────────────────────────────────────────────────────────
  it('11. familyName is stored in the font name table', () => {
    const buffer = assembleFontFromGlyphs(makeGlyphImages(), null, 'MyGeneratedCyrillicFont');
    const font = opentype.parse(buffer);

    expect(font.names.fontFamily.en).toBe('MyGeneratedCyrillicFont');
  });

  // ── test 12 ───────────────────────────────────────────────────────────────
  it('12. empty glyphImages map → .notdef-only font still returns valid ArrayBuffer', () => {
    // If no Cyrillic images are provided the assembler must still produce a
    // syntactically valid OTF with at least the .notdef glyph.
    const buffer = assembleFontFromGlyphs(new Map(), null, 'EmptyFont');
    expect(buffer).toBeInstanceOf(ArrayBuffer);
    expect(buffer.byteLength).toBeGreaterThan(0);

    const font = opentype.parse(buffer);
    expect(font.glyphs.get(0).name).toBe('.notdef');
  });

  // ── test 13 ───────────────────────────────────────────────────────────────
  it('13. merged font contains both Latin glyphs from uploaded font and Cyrillic glyphs', () => {
    // Create a simple source font with Latin glyphs
    const latinGlyphs: opentype.Glyph[] = [
      new opentype.Glyph({ name: '.notdef', unicode: undefined as unknown as number, advanceWidth: 500, path: new opentype.Path() }),
      new opentype.Glyph({ name: 'A', unicode: 0x0041, advanceWidth: 600, path: new opentype.Path() }),
      new opentype.Glyph({ name: 'B', unicode: 0x0042, advanceWidth: 600, path: new opentype.Path() }),
    ];
    const sourceFont = new opentype.Font({
      familyName: 'TestLatinFont',
      styleName: 'Regular',
      unitsPerEm: 1000,
      ascender: 800,
      descender: -200,
      glyphs: latinGlyphs,
    });
    const sourceFontBuffer = sourceFont.toArrayBuffer();

    // Merge with Cyrillic glyphs
    const buffer = assembleFontFromGlyphs(makeGlyphImages(), sourceFontBuffer, 'TestFont');
    const mergedFont = opentype.parse(buffer);

    // Verify Latin glyphs are present
    const aGlyphIndex = mergedFont.charToGlyphIndex('A');
    expect(aGlyphIndex).toBeGreaterThan(0);
    const aGlyph = mergedFont.glyphs.get(aGlyphIndex);
    expect(aGlyph.unicode).toBe(0x0041);

    const bGlyphIndex = mergedFont.charToGlyphIndex('B');
    expect(bGlyphIndex).toBeGreaterThan(0);
    const bGlyph = mergedFont.glyphs.get(bGlyphIndex);
    expect(bGlyph.unicode).toBe(0x0042);

    // Verify Cyrillic glyphs are present
    const cyrillicAIndex = mergedFont.charToGlyphIndex('А');
    expect(cyrillicAIndex).toBeGreaterThan(0);
    const cyrillicAGlyph = mergedFont.glyphs.get(cyrillicAIndex);
    expect(cyrillicAGlyph.unicode).toBe(0x0410);

    // Total glyphs: .notdef + 2 Latin + 66 Cyrillic = 69
    expect(mergedFont.glyphs.length).toBe(69);
  });

  // ── test 14 ───────────────────────────────────────────────────────────────
  it('14. merged font family name ends with " Cyrillic"', () => {
    // Create source font with a known family name
    const latinGlyphs: opentype.Glyph[] = [
      new opentype.Glyph({ name: '.notdef', unicode: undefined as unknown as number, advanceWidth: 500, path: new opentype.Path() }),
    ];
    const sourceFont = new opentype.Font({
      familyName: 'MyCustomFont',
      styleName: 'Regular',
      unitsPerEm: 1000,
      ascender: 800,
      descender: -200,
      glyphs: latinGlyphs,
    });
    const sourceFontBuffer = sourceFont.toArrayBuffer();

    // Merge with Cyrillic glyphs
    const buffer = assembleFontFromGlyphs(makeGlyphImages(), sourceFontBuffer, 'TestFont');
    const mergedFont = opentype.parse(buffer);

    // Family name should be "MyCustomFont Cyrillic"
    expect(mergedFont.names.fontFamily.en).toBe('MyCustomFont Cyrillic');
  });

  // ── test 15 ───────────────────────────────────────────────────────────────
  it('15. merged font replaces existing Cyrillic glyphs in uploaded font', () => {
    // Create source font with existing Cyrillic glyph
    const glyphs: opentype.Glyph[] = [
      new opentype.Glyph({ name: '.notdef', unicode: undefined as unknown as number, advanceWidth: 500, path: new opentype.Path() }),
      new opentype.Glyph({ name: 'A', unicode: 0x0041, advanceWidth: 600, path: new opentype.Path() }),
      new opentype.Glyph({ name: 'uni0410', unicode: 0x0410, advanceWidth: 600, path: new opentype.Path() }), // Old Cyrillic А
    ];
    const sourceFont = new opentype.Font({
      familyName: 'TestFont',
      styleName: 'Regular',
      unitsPerEm: 1000,
      ascender: 800,
      descender: -200,
      glyphs,
    });
    const sourceFontBuffer = sourceFont.toArrayBuffer();

    // Merge - this should replace the existing Cyrillic glyph
    const buffer = assembleFontFromGlyphs(makeGlyphImages(), sourceFontBuffer, 'TestFont');
    const mergedFont = opentype.parse(buffer);

    // Should have .notdef + 1 Latin + 66 Cyrillic = 68 glyphs (not 69)
    expect(mergedFont.glyphs.length).toBe(68);

    // Latin A should still be present
    const aGlyphIndex = mergedFont.charToGlyphIndex('A');
    expect(aGlyphIndex).toBeGreaterThan(0);

    // Cyrillic А should be present (from AI generation, not the old one)
    const cyrillicAIndex = mergedFont.charToGlyphIndex('А');
    expect(cyrillicAIndex).toBeGreaterThan(0);
  });
});

// ─── FontDownloader ──────────────────────────────────────────────────────

describe('FontDownloader', () => {
  let capturedBlob: Blob | undefined;
  let anchorClickSpy: ReturnType<typeof vi.fn>;
  let createdAnchor: HTMLAnchorElement | undefined;

  beforeEach(() => {
    capturedBlob = undefined;
    createdAnchor = undefined;
    anchorClickSpy = vi.fn();

    // Mock URL.createObjectURL / revokeObjectURL (not implemented in jsdom)
    vi.spyOn(URL, 'createObjectURL').mockImplementation((obj: Blob | MediaSource | File) => {
      capturedBlob = obj as Blob;
      return 'blob:http://localhost/mock-font-uuid';
    });
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => { /* no-op */ });

    // Intercept document.createElement to capture the anchor and spy on click
    const originalCreate = document.createElement.bind(document);
    vi.spyOn(document, 'createElement').mockImplementation((tagName: string) => {
      const el = originalCreate(tagName);
      if (tagName === 'a') {
        el.click = anchorClickSpy;
        createdAnchor = el as HTMLAnchorElement;
      }
      return el;
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // ── test 13 ───────────────────────────────────────────────────────────────
  it('13. creates a Blob URL and revokes it after triggering the download', () => {
    downloadFont(new ArrayBuffer(256), 'my-font.otf');

    expect(URL.createObjectURL).toHaveBeenCalledOnce();
    // The URL must be revoked (to release the object URL memory)
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:http://localhost/mock-font-uuid');
  });

  // ── test 14 ───────────────────────────────────────────────────────────────
  it('14. anchor element is clicked and carries the correct download filename + href', () => {
    downloadFont(new ArrayBuffer(256), 'cyrillic-font.otf');

    expect(anchorClickSpy).toHaveBeenCalledOnce();
    expect(createdAnchor).toBeDefined();
    expect(createdAnchor!.download).toBe('cyrillic-font.otf');
    expect(createdAnchor!.href).toBe('blob:http://localhost/mock-font-uuid');
  });

  // ── test 15 ───────────────────────────────────────────────────────────────
  it('15. Blob is created with MIME type font/otf', () => {
    downloadFont(new ArrayBuffer(256), 'test.otf');

    expect(capturedBlob).toBeDefined();
    expect(capturedBlob!.type).toBe('font/otf');
  });
});
