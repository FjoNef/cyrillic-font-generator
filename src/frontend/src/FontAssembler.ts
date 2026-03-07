import * as opentype from 'opentype.js';
import { CYRILLIC_CHARS } from './font/cyrillicCharset';
import { vectorizeGlyph } from './GlyphVectorizer';

const UPM       = 1000;
const ASCENDER  = 800;
const DESCENDER = -200;
const ADVANCE_WIDTH = 600;

const OFL_LICENSE_TEXT =
  'This Font Software is licensed under the SIL Open Font License, Version 1.1. ' +
  'This license is available with a FAQ at: https://openfontlicense.org';

const OFL_LICENSE_URL = 'https://openfontlicense.org';

/**
 * Assembles a complete OTF font from pre-generated glyph images.
 *
 * @param glyphImages  Map from model glyph index (0-65) → raw model output Float32Array [128*128], range [-1,1]
 * @param familyName   Font family name for the nameTable
 * @returns            Binary OTF data as ArrayBuffer
 */
export function assembleFontFromGlyphs(
  glyphImages: Map<number, Float32Array>,
  familyName: string
): ArrayBuffer {
  // .notdef glyph (blank, required first glyph in every OpenType font)
  const notdefGlyph = new opentype.Glyph({
    name: '.notdef',
    unicode: undefined as unknown as number,
    advanceWidth: ADVANCE_WIDTH,
    path: new opentype.Path(),
  });

  const glyphs: opentype.Glyph[] = [notdefGlyph];

  // Add one glyph per Cyrillic character, in charset order
  for (const { index, unicode } of CYRILLIC_CHARS) {
    const imageData = glyphImages.get(index);
    const glyphPath = imageData
      ? vectorizeGlyph(imageData)
      : new opentype.Path(); // fallback blank if missing

    const glyph = new opentype.Glyph({
      name: `uni${unicode.toString(16).toUpperCase().padStart(4, '0')}`,
      unicode,
      advanceWidth: ADVANCE_WIDTH,
      path: glyphPath,
    });

    glyphs.push(glyph);
  }

  const font = new opentype.Font({
    familyName,
    styleName: 'Regular',
    unitsPerEm: UPM,
    ascender: ASCENDER,
    descender: DESCENDER,
    glyphs,
  });

  // Attach OFL license metadata to the name table
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const names = font.names as any;
  names.license    = { en: OFL_LICENSE_TEXT };
  names.licenseURL = { en: OFL_LICENSE_URL };

  return font.toArrayBuffer();
}
