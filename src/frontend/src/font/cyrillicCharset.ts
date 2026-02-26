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

// Build uppercase block (indices 0-32, 33 chars):
//   А-Е (U+0410–U+0415) at indices 0-5
//   Ё   (U+0401)        at index 6  ← LOCKED tensor contract
//   Ж-Я (U+0416–U+042F) at indices 7-32
const upperBase: CyrillicChar[] = [];
// А(0)–Е(5): U+0410–U+0415
for (let i = 0; i < 6; i++) {
  upperBase.push({ char: String.fromCodePoint(0x0410 + i), unicode: 0x0410 + i, index: i, isUppercase: true });
}
// Ё at index 6
upperBase.push({ char: 'Ё', unicode: 0x0401, index: 6, isUppercase: true });
// Ж(7)–Я(32): U+0416–U+042F
for (let i = 0; i < 26; i++) {
  upperBase.push({ char: String.fromCodePoint(0x0416 + i), unicode: 0x0416 + i, index: 7 + i, isUppercase: true });
}

// Build lowercase block (indices 33-65, 33 chars):
//   а-е (U+0430–U+0435) at indices 33-38
//   ё   (U+0451)        at index 39  ← LOCKED tensor contract
//   ж-я (U+0436–U+044F) at indices 40-65
const lowerBase: CyrillicChar[] = [];
// а(33)–е(38): U+0430–U+0435
for (let i = 0; i < 6; i++) {
  lowerBase.push({ char: String.fromCodePoint(0x0430 + i), unicode: 0x0430 + i, index: 33 + i, isUppercase: false });
}
// ё at index 39
lowerBase.push({ char: 'ё', unicode: 0x0451, index: 39, isUppercase: false });
// ж(40)–я(65): U+0436–U+044F
for (let i = 0; i < 26; i++) {
  lowerBase.push({ char: String.fromCodePoint(0x0436 + i), unicode: 0x0436 + i, index: 40 + i, isUppercase: false });
}

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
