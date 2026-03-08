import * as opentype from 'opentype.js';
import { CYRILLIC_CHARS } from './font/cyrillicCharset';
import { vectorizeGlyph } from './GlyphVectorizer';

const OFL_LICENSE_TEXT =
  'This Font Software is licensed under the SIL Open Font License, Version 1.1. ' +
  'This license is available with a FAQ at: https://openfontlicense.org';

const OFL_LICENSE_URL = 'https://openfontlicense.org';

// Cyrillic Unicode range: 0x0400-0x04FF
const CYRILLIC_UNICODE_MIN = 0x0400;
const CYRILLIC_UNICODE_MAX = 0x04FF;

/**
 * Assembles a complete OTF font by merging AI-generated Cyrillic glyphs into the uploaded font.
 *
 * @param glyphImages   Map from model glyph index (0-65) → raw model output Float32Array [128*128], range [-1,1]
 * @param uploadedFont  Original uploaded font buffer to merge with
 * @param baseFamilyName Base font family name (usually the uploaded font filename)
 * @returns             Binary OTF data as ArrayBuffer with merged glyphs
 */
export function assembleFontFromGlyphs(
  glyphImages: Map<number, Float32Array>,
  uploadedFont: ArrayBuffer | null,
  baseFamilyName: string
): ArrayBuffer {
  // If no uploaded font provided, create a standalone Cyrillic-only font with default metrics
  if (!uploadedFont) {
    return createStandaloneCyrillicFont(glyphImages, baseFamilyName);
  }

  // Parse the uploaded font
  const sourceFont = opentype.parse(uploadedFont);

  // Extract font metrics from uploaded font
  const upm = sourceFont.unitsPerEm;
  const ascender = sourceFont.ascender;
  const descender = sourceFont.descender;

  // Scale advance width proportionally to uploaded font's UPM
  // Our vectorizer outputs in 1000 UPM space with 600 advance width
  const cyrillicAdvanceWidth = Math.round(600 * upm / 1000);

  // Build the output font family name
  const existingFamilyName = 
    sourceFont.names.fontFamily?.en || 
    sourceFont.names.fullName?.en || 
    baseFamilyName;
  const newFamilyName = `${existingFamilyName} Cyrillic`;

  // Build output glyph list
  const glyphs: opentype.Glyph[] = [];

  // 1. Add .notdef glyph (required first glyph)
  const notdefGlyph = new opentype.Glyph({
    name: '.notdef',
    unicode: undefined as unknown as number,
    advanceWidth: sourceFont.glyphs.get(0)?.advanceWidth || cyrillicAdvanceWidth,
    path: sourceFont.glyphs.get(0)?.path || new opentype.Path(),
  });
  glyphs.push(notdefGlyph);

  // 2. Copy all glyphs from uploaded font (except .notdef and Cyrillic range)
  const numGlyphs = sourceFont.numGlyphs;
  for (let i = 1; i < numGlyphs; i++) {
    const glyph = sourceFont.glyphs.get(i);
    
    // Skip glyphs without unicode (special glyphs like ligatures)
    if (glyph.unicode === undefined) {
      continue;
    }

    // Skip Cyrillic range - we'll add our generated ones instead
    if (glyph.unicode >= CYRILLIC_UNICODE_MIN && glyph.unicode <= CYRILLIC_UNICODE_MAX) {
      continue;
    }

    // Copy the glyph
    glyphs.push(new opentype.Glyph({
      name: glyph.name ?? undefined,
      unicode: glyph.unicode,
      advanceWidth: glyph.advanceWidth,
      path: glyph.path,
    }));
  }

  // 3. Add AI-generated Cyrillic glyphs
  for (const { index, unicode } of CYRILLIC_CHARS) {
    const imageData = glyphImages.get(index);
    const glyphPath = imageData
      ? vectorizeGlyph(imageData, upm)
      : new opentype.Path(); // fallback blank if missing

    const glyph = new opentype.Glyph({
      name: `uni${unicode.toString(16).toUpperCase().padStart(4, '0')}`,
      unicode,
      advanceWidth: cyrillicAdvanceWidth,
      path: glyphPath,
    });

    glyphs.push(glyph);
  }

  // Create merged font
  const font = new opentype.Font({
    familyName: newFamilyName,
    styleName: sourceFont.names.fontSubfamily?.en || 'Regular',
    unitsPerEm: upm,
    ascender,
    descender,
    glyphs,
  });

  // Attach OFL license metadata to the name table
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const names = font.names as any;
  names.license    = { en: OFL_LICENSE_TEXT };
  names.licenseURL = { en: OFL_LICENSE_URL };

  return font.toArrayBuffer();
}

/**
 * Creates a standalone Cyrillic-only font with default metrics (fallback path).
 */
function createStandaloneCyrillicFont(
  glyphImages: Map<number, Float32Array>,
  familyName: string
): ArrayBuffer {
  const UPM = 1000;
  const ASCENDER = 800;
  const DESCENDER = -200;
  const ADVANCE_WIDTH = 600;

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
      ? vectorizeGlyph(imageData, UPM)
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
