import * as opentype from 'opentype.js';

const IMG_SIZE = 128;

// Fixed font metrics (1000 UPM, ascender 800, descender -200, advance width 600)
const ASCENDER = 800;
const DESCENDER = -200;
const ADVANCE_WIDTH = 600;

// Per-pixel scale factors
const X_SCALE = ADVANCE_WIDTH / IMG_SIZE;             // 600/128 ≈ 4.6875
const Y_SCALE = (ASCENDER - DESCENDER) / IMG_SIZE;    // 1000/128 ≈ 7.8125

/**
 * Vectorize a 128×128 Float32Array glyph (raw model output, range [-1,1]) into an opentype.Path.
 *
 * Ink convention: pixel value > 0 → black ink, else white background.
 * Coordinate mapping:
 *   - X: column 0 → 0 units, column 128 → 600 units (advance width)
 *   - Y: row 0 = top of image = ascender (800); row 128 = descender (-200)
 *       yTop of row r = 800 - r * (1000/128)
 *       yBottom of row r = 800 - (r+1) * (1000/128)
 *
 * Algorithm: scanline rectangles — for each row, find consecutive runs of ink pixels
 * and emit one closed rectangle path per run.
 */
export function vectorizeGlyph(data: Float32Array): opentype.Path {
  const path = new opentype.Path();

  for (let row = 0; row < IMG_SIZE; row++) {
    const yTop    = ASCENDER - row       * Y_SCALE;
    const yBottom = ASCENDER - (row + 1) * Y_SCALE;

    let runStart = -1;
    for (let col = 0; col <= IMG_SIZE; col++) {
      const isInk = col < IMG_SIZE && data[row * IMG_SIZE + col] > 0;

      if (isInk && runStart === -1) {
        runStart = col;
      } else if (!isInk && runStart !== -1) {
        const xLeft  = runStart * X_SCALE;
        const xRight = col      * X_SCALE;

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
