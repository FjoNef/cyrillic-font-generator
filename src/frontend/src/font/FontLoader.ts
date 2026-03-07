import * as opentype from 'opentype.js';
import { CYRILLIC_CHARS } from './cyrillicCharset';

/** Latin reference glyphs used for style conditioning (10 characters). */
const STYLE_CHARS = ['A', 'B', 'C', 'D', 'E', 'H', 'I', 'O', 'R', 'X'] as const;

const RENDER_SIZE = 128;
const RENDER_PADDING = 8;

export class FontLoader {
  /**
   * Parse an OTF/TTF/WOFF2 ArrayBuffer into an opentype.Font.
   */
  async loadFont(buffer: ArrayBuffer): Promise<opentype.Font> {
    return opentype.parse(buffer);
  }

  /**
   * Render 10 Latin reference glyphs (A B C D E H I O R X) to 128×128 grayscale canvases.
   * Returns a flattened Float32Array of shape [10, 1, 128, 128] normalized to [-1, 1].
   * (1 = black ink, -1 = white background, matching pix2pix convention)
   */
  extractStyleGlyphs(font: opentype.Font): Float32Array {
    const total = STYLE_CHARS.length * RENDER_SIZE * RENDER_SIZE;
    const result = new Float32Array(total);

    const canvas = document.createElement('canvas');
    canvas.width = RENDER_SIZE;
    canvas.height = RENDER_SIZE;
    const ctx = canvas.getContext('2d')!;

    for (let i = 0; i < STYLE_CHARS.length; i++) {
      ctx.clearRect(0, 0, RENDER_SIZE, RENDER_SIZE);
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, RENDER_SIZE, RENDER_SIZE);

      const glyph = font.charToGlyph(STYLE_CHARS[i]);
      const baseline = RENDER_SIZE - RENDER_PADDING;

      const path = glyph.getPath(RENDER_PADDING, baseline, RENDER_SIZE - RENDER_PADDING * 2);
      path.fill = 'black';

      const path2d = new Path2D(path.toSVG(2));
      ctx.fillStyle = 'black';
      ctx.fill(path2d);

      const imageData = ctx.getImageData(0, 0, RENDER_SIZE, RENDER_SIZE);
      const offset = i * RENDER_SIZE * RENDER_SIZE;

      for (let px = 0; px < RENDER_SIZE * RENDER_SIZE; px++) {
        // Use red channel (grayscale image), convert [0,255] → [-1, 1]
        // White (255) → -1 (background), Black (0) → 1 (ink)
        const brightness = imageData.data[px * 4] / 255;
        result[offset + px] = 1 - brightness * 2; // invert: white bg → -1
      }
    }

    return result;
  }

  /**
   * Assemble a new OTF font with generated Cyrillic glyphs.
   * 
   * @param _styleGlyphs    The 10 Latin style glyphs as Float32Array [10, 1, 128, 128] (unused, kept for API compatibility)
   * @param inferFn         Async function (charIndex: number) => Float32Array output [128*128]
   * @returns               ArrayBuffer of the assembled .otf file
   */
  async assembleCyrillicFont(
    _styleGlyphs: Float32Array,
    inferFn: (charIndex: number) => Promise<Float32Array>
  ): Promise<ArrayBuffer> {
    // Generate all 66 Cyrillic glyphs
    const glyphImages = new Map<string, Float32Array>();

    for (const { char, index } of CYRILLIC_CHARS) {
      const output = await inferFn(index);
      glyphImages.set(char, output);
    }

    // Create a new font with basic metrics
    const unitsPerEm = 1000;
    const ascender = 800;
    const descender = -200;
    const advanceWidth = 600;

    const notdefGlyph = new opentype.Glyph({
      name: '.notdef',
      unicode: undefined,
      advanceWidth: advanceWidth,
      path: new opentype.Path(),
    });

    const glyphs: opentype.Glyph[] = [notdefGlyph];

    // Add Cyrillic glyphs
    for (const { char, unicode } of CYRILLIC_CHARS) {
      const imageData = glyphImages.get(char)!;
      const path = this.vectorizeGlyph(imageData, unitsPerEm);

      const glyph = new opentype.Glyph({
        name: `uni${unicode.toString(16).toUpperCase().padStart(4, '0')}`,
        unicode: unicode,
        advanceWidth: advanceWidth,
        path: path,
      });

      glyphs.push(glyph);
    }

    const font = new opentype.Font({
      familyName: 'Generated Cyrillic',
      styleName: 'Regular',
      unitsPerEm: unitsPerEm,
      ascender: ascender,
      descender: descender,
      glyphs: glyphs,
    });

    return font.toArrayBuffer();
  }

  /**
   * Vectorize a 128×128 float32 glyph image to an opentype.Path.
   * Uses threshold-based contour extraction.
   * 
   * @param data  Float32Array [128*128], values in [-1, 1] (1=ink, -1=background)
   * @param upm   Target units per em for scaling
   * @returns     opentype.Path in font coordinate space
   */
  private vectorizeGlyph(data: Float32Array, upm: number): opentype.Path {
    const size = 128;
    const threshold = 0.0; // Values > 0 are considered ink

    // Binarize to boolean array
    const binary = new Array(size * size);
    for (let i = 0; i < size * size; i++) {
      binary[i] = data[i] > threshold;
    }

    const path = new opentype.Path();
    const scale = upm / size;

    // Simple scanline-based path generation
    // For each row, find runs of ink pixels and draw rectangles
    for (let y = 0; y < size; y++) {
      let startX = -1;
      for (let x = 0; x <= size; x++) {
        const isInk = x < size && binary[y * size + x];
        
        if (isInk && startX === -1) {
          startX = x;
        } else if (!isInk && startX !== -1) {
          // Draw horizontal segment from startX to x-1
          const x0 = startX * scale;
          const x1 = x * scale;
          const y0 = (size - y - 1) * scale; // Flip Y for font coords
          const y1 = (size - y) * scale;

          path.moveTo(x0, y0);
          path.lineTo(x1, y0);
          path.lineTo(x1, y1);
          path.lineTo(x0, y1);
          path.close();

          startX = -1;
        }
      }
    }

    return path;
  }
}
