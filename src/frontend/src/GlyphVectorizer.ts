import * as opentype from 'opentype.js';

const IMG_SIZE = 128;

// Default font metrics (1000 UPM, ascender 800, descender -200, advance width 600)
const DEFAULT_UPM = 1000;
const DEFAULT_ASCENDER = 800;
const DEFAULT_DESCENDER = -200;
const DEFAULT_ADVANCE_WIDTH = 600;

/**
 * Vectorize a 128×128 Float32Array glyph (raw model output, range [-1,1]) into an opentype.Path.
 *
 * Ink convention: pixel value > 0 → black ink, else white background.
 * Coordinate mapping:
 *   - X: column 0 → 0 units, column 128 → advance width (scaled to target UPM)
 *   - Y: row 0 = top of image = ascender; row 128 = descender
 *       yTop of row r = ascender - r * yScale
 *       yBottom of row r = ascender - (r+1) * yScale
 *
 * Algorithm: scanline rectangles — for each row, find consecutive runs of ink pixels
 * and emit one closed rectangle path per run.
 *
 * @param data Raw model output Float32Array [128*128], range [-1,1]
 * @param targetUpm Target font units per em (default: 1000). Used to scale the output coordinates.
 * @returns opentype.Path with coordinates scaled to targetUpm
 */
export function vectorizeGlyph(data: Float32Array, targetUpm: number = DEFAULT_UPM): opentype.Path {
  const path = new opentype.Path();

  // Scale factor from default 1000 UPM to target UPM
  const scale = targetUpm / DEFAULT_UPM;
  
  // Font metrics scaled to target UPM
  const ascender = DEFAULT_ASCENDER * scale;
  const descender = DEFAULT_DESCENDER * scale;
  const advanceWidth = DEFAULT_ADVANCE_WIDTH * scale;

  // Per-pixel scale factors
  const xScale = advanceWidth / IMG_SIZE;
  const yScale = (ascender - descender) / IMG_SIZE;

  for (let row = 0; row < IMG_SIZE; row++) {
    const yTop    = ascender - row       * yScale;
    const yBottom = ascender - (row + 1) * yScale;

    let runStart = -1;
    for (let col = 0; col <= IMG_SIZE; col++) {
      const isInk = col < IMG_SIZE && data[row * IMG_SIZE + col] > 0;

      if (isInk && runStart === -1) {
        runStart = col;
      } else if (!isInk && runStart !== -1) {
        const xLeft  = runStart * xScale;
        const xRight = col      * xScale;

        // CW rectangle (matching existing codebase convention)
        path.moveTo(xLeft,  yBottom);
        path.lineTo(xRight, yBottom);
        path.lineTo(xRight, yTop);
        path.lineTo(xLeft,  yTop);
        path.close();

        runStart = -1;
      }
    }
  }

  return path;
}
