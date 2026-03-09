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
      
      if (i === 0) {
        console.debug(`[FontLoader] Glyph '${STYLE_CHARS[i]}': hasContours=${glyph.path?.commands?.length || 0} commands`);
      }
      
      // Calculate fontSize to fit glyph (including descenders) in canvas
      const availableHeight = RENDER_SIZE - RENDER_PADDING * 2;
      const fontHeight = font.ascender - font.descender; // total height in font units
      const fontSize = (availableHeight / fontHeight) * font.unitsPerEm;
      const scale = fontSize / font.unitsPerEm;
      const ascender = font.ascender * scale;
      const baseline = RENDER_PADDING + ascender;

      const path = glyph.getPath(RENDER_PADDING, baseline, fontSize);
      
      // Manually draw path using canvas 2D API (Path2D has issues in this environment)
      ctx.beginPath();
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const cmd of path.commands) {
        if (cmd.type === 'M') {
          ctx.moveTo(cmd.x, cmd.y);
          minX = Math.min(minX, cmd.x); maxX = Math.max(maxX, cmd.x);
          minY = Math.min(minY, cmd.y); maxY = Math.max(maxY, cmd.y);
        } else if (cmd.type === 'L') {
          ctx.lineTo(cmd.x, cmd.y);
          minX = Math.min(minX, cmd.x); maxX = Math.max(maxX, cmd.x);
          minY = Math.min(minY, cmd.y); maxY = Math.max(maxY, cmd.y);
        } else if (cmd.type === 'C') {
          ctx.bezierCurveTo(cmd.x1, cmd.y1, cmd.x2, cmd.y2, cmd.x, cmd.y);
          minX = Math.min(minX, cmd.x, cmd.x1, cmd.x2); maxX = Math.max(maxX, cmd.x, cmd.x1, cmd.x2);
          minY = Math.min(minY, cmd.y, cmd.y1, cmd.y2); maxY = Math.max(maxY, cmd.y, cmd.y1, cmd.y2);
        } else if (cmd.type === 'Q') {
          ctx.quadraticCurveTo(cmd.x1, cmd.y1, cmd.x, cmd.y);
          minX = Math.min(minX, cmd.x, cmd.x1); maxX = Math.max(maxX, cmd.x, cmd.x1);
          minY = Math.min(minY, cmd.y, cmd.y1); maxY = Math.max(maxY, cmd.y, cmd.y1);
        } else if (cmd.type === 'Z') {
          ctx.closePath();
        }
      }
      ctx.fillStyle = 'black';
      ctx.fill();

      if (i === 0) {
        console.debug(`[FontLoader] Glyph '${STYLE_CHARS[i]}': bbox [${minX.toFixed(1)}, ${minY.toFixed(1)}] to [${maxX.toFixed(1)}, ${maxY.toFixed(1)}], canvas: [0,0] to [${RENDER_SIZE},${RENDER_SIZE}]`);
      }

      const imageData = ctx.getImageData(0, 0, RENDER_SIZE, RENDER_SIZE);
      const offset = i * RENDER_SIZE * RENDER_SIZE;

      for (let px = 0; px < RENDER_SIZE * RENDER_SIZE; px++) {
        // Use red channel (grayscale image), convert [0,255] → [-1, 1]
        // White (255) → -1 (background), Black (0) → 1 (ink)
        const brightness = imageData.data[px * 4] / 255;
        result[offset + px] = 1 - brightness * 2; // invert: white bg → -1
      }
    }
    
    // DEBUG: Verify style extraction produced meaningful data
    const sampleValues = Array.from(result.slice(0, 20));
    const minVal = Math.min(...sampleValues);
    const maxVal = Math.max(...sampleValues);
    const lastGlyph = font.charToGlyph(STYLE_CHARS[STYLE_CHARS.length - 1]);
    console.debug(`[FontLoader] DEBUG: Last glyph commands=${lastGlyph.path?.commands?.length || 0}`);
    console.debug(`[FontLoader] Extracted style glyphs: ${STYLE_CHARS.join('')}. Sample values (first 20): min=${minVal.toFixed(3)}, max=${maxVal.toFixed(3)}`);

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
