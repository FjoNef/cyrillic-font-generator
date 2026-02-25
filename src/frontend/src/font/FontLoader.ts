import * as opentype from 'opentype.js';

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
      const unitsPerEm = font.unitsPerEm;
      const scale = (RENDER_SIZE - RENDER_PADDING * 2) / unitsPerEm;
      const baseline = RENDER_SIZE - RENDER_PADDING;

      const path = glyph.getPath(RENDER_PADDING, baseline, RENDER_SIZE - RENDER_PADDING * 2);
      path.fill = 'black';

      const path2d = new Path2D(path.toSVG()!);
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
   * Assemble a new OTF font by injecting vectorized Cyrillic glyphs into the source font.
   *
   * @param sourceFont     The user's original font (provides metrics, Latin glyphs, metadata)
   * @param generatedGlyphs  Map of Cyrillic char → 128×128 ImageData from model inference
   * @returns              ArrayBuffer of the assembled .otf file, ready for download
   *
   * TODO: implement potrace vectorization (ImageData → SVG path) and glyph injection.
   *       Steps:
   *       1. For each entry in generatedGlyphs, binarize ImageData
   *       2. Run potrace (JS port) to get SVG path string
   *       3. Scale SVG path from 128px → font units (sourceFont.unitsPerEm)
   *       4. Create opentype.Glyph with path + advance width from sourceFont metrics
   *       5. Clone sourceFont glyphs array, append/replace Cyrillic glyphs
   *       6. Serialize with font.download() or font.arrayBuffer()
   */
  assembleCyrillicFont(
    _sourceFont: opentype.Font,
    _generatedGlyphs: Map<string, ImageData>
  ): ArrayBuffer {
    throw new Error('TODO: assembleCyrillicFont not yet implemented — waiting for potrace integration');
  }
}
