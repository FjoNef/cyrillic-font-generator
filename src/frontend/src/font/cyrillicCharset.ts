/**
 * Russian Cyrillic character set.
 * 66 characters: 33 uppercase (А-Я + Ё) at indices 0-32,
 *                33 lowercase (а-я + ё) at indices 33-65.
 *
 * Unicode block: U+0410–U+044F (А-я) + U+0401 (Ё) + U+0451 (ё)
 * Uppercase А-Я: U+0410–U+042F (32 chars), Ё: U+0401
 * Lowercase а-я: U+0430–U+044F (32 chars), ё: U+0451
 */

export interface CyrillicChar {
  char: string;
  unicode: number;
  index: number; // 0-65
  isUppercase: boolean;
}

// Build uppercase А-Я (U+0410–U+042F), then Ё (U+0401)
const upperBase: CyrillicChar[] = Array.from({ length: 32 }, (_, i) => ({
  char: String.fromCodePoint(0x0410 + i),
  unicode: 0x0410 + i,
  index: i,
  isUppercase: true,
}));
// Ё is U+0401 — insert at alphabetic position (after Е = index 5)
// For model indexing we keep it last in uppercase block (index 32)
upperBase.push({
  char: 'Ё',
  unicode: 0x0401,
  index: 32,
  isUppercase: true,
});

// Build lowercase а-я (U+0430–U+044F), then ё (U+0451)
const lowerBase: CyrillicChar[] = Array.from({ length: 32 }, (_, i) => ({
  char: String.fromCodePoint(0x0430 + i),
  unicode: 0x0430 + i,
  index: 33 + i,
  isUppercase: false,
}));
lowerBase.push({
  char: 'ё',
  unicode: 0x0451,
  index: 65,
  isUppercase: false,
});

/** All 66 Russian Cyrillic characters, indices 0-65. */
export const CYRILLIC_CHARS: CyrillicChar[] = [...upperBase, ...lowerBase];

/** Map from character string → model index (0-65). */
export const CHAR_INDEX_MAP: Map<string, number> = new Map(
  CYRILLIC_CHARS.map(({ char, index }) => [char, index])
);

/** Map from Unicode code point → model index. */
export const UNICODE_INDEX_MAP: Map<number, number> = new Map(
  CYRILLIC_CHARS.map(({ unicode, index }) => [unicode, index])
);
