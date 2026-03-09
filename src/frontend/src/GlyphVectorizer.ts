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
 * Ink convention (RAW model output space, NOT display-pixel space):
 *   +1.0 = black ink (foreground) → detected as ink (value > 0)
 *   -1.0 = white background       → skipped (value ≤ 0)
 *
 * Note: the display postprocessing formula ((1-output)/2)*255 maps +1→0 (dark) and -1→255
 * (bright), so in *display* space ink appears as LOW pixel values.  The vectorizer
 * intentionally operates on the raw [-1,1] tensor BEFORE that transformation, so the
 * correct threshold is `value > 0`, not `value < threshold`.
 *
 * Coordinate mapping (y-up font space, CCW contours = filled in CFF):
 *   - X: column 0 → 0 units, column 128 → advance width (scaled to target UPM)
 *   - Y: row 0 = top of image = ascender; row 128 = descender
 *       yTop of row r = ascender - r * yScale
 *       yBottom of row r = ascender - (r+1) * yScale
 *
 * Algorithm: scanline rectangles — for each row, find consecutive runs of ink pixels
 * and emit one closed CCW rectangle path per run.
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

  // DEBUG: Sample output values to detect all-background output
  const sampleSize = Math.min(data.length, 100);
  let minVal = Infinity;
  let maxVal = -Infinity;
  let inkPixelCount = 0;
  
  for (let i = 0; i < sampleSize; i++) {
    minVal = Math.min(minVal, data[i]);
    maxVal = Math.max(maxVal, data[i]);
  }
  
  for (let i = 0; i < data.length; i++) {
    if (data[i] > 0) inkPixelCount++;
  }

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

        // CCW rectangle in y-up font space → filled outer contour in CFF/OTF
        path.moveTo(xLeft,  yBottom);
        path.lineTo(xRight, yBottom);
        path.lineTo(xRight, yTop);
        path.lineTo(xLeft,  yTop);
        path.close();

        runStart = -1;
      }
    }
  }

  if (path.commands.length === 0) {
    console.warn(
      `[GlyphVectorizer] vectorizeGlyph produced an empty path (0 commands). ` +
      `Data stats: min=${minVal.toFixed(4)}, max=${maxVal.toFixed(4)}, ink pixels (>0): ${inkPixelCount}/${data.length}. ` +
      `Possible causes: all model output values ≤ 0 (all-background output), or wrong ` +
      `data space (display values 0-255 passed instead of raw [-1,1]). ` +
      `Check that data is raw model output where +1.0 = ink, -1.0 = background.`
    );
  } else {
    console.debug(
      `[GlyphVectorizer] Generated ${path.commands.length} path commands. ` +
      `Data stats: min=${minVal.toFixed(4)}, max=${maxVal.toFixed(4)}, ink pixels: ${inkPixelCount}/${data.length}`
    );
  }

  return path;
}
