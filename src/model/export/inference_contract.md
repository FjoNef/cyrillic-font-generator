# Inference Contract — Cyrillic Font Generator ONNX Model

**Version:** v1  
**For:** Togusa (Frontend / ONNX Runtime Web integration)  
**Produced by:** Major (AI/ML Engineer)

---

## Overview

The browser receives a single ONNX file (`generator.onnx`) that encodes both the
StyleEncoder and the UNetGenerator in one graph.  Given a set of rendered Latin
glyphs from the user's uploaded font and a target Cyrillic character index, the
model outputs a 128×128 grayscale glyph image.

---

## Model File

| Property | Value |
|---|---|
| File | `models/v1/generator.onnx` |
| ONNX opset | 17 |
| Quantization | Dynamic INT8 weights |
| Estimated size | ~12–18 MB (uncompressed); ~8–12 MB gzip |
| Size limit | < 20 MB compressed |

---

## Inputs

### `style_glyphs`

| Property | Value |
|---|---|
| Name | `style_glyphs` |
| Shape | `[B, 10, 1, 128, 128]` |
| Dtype | `float32` |
| Value range | `[-1.0, 1.0]` |

`B` = batch size (use `B=1` for single-glyph inference in the browser).  
`10` = number of Latin reference glyphs (fixed; always supply exactly 10).  
`1` = grayscale channel.  
`128×128` = glyph image resolution.

### `char_index`

| Property | Value |
|---|---|
| Name | `char_index` |
| Shape | `[B]` |
| Dtype | `int64` |
| Value range | `0–65` |

---

## Output

### `generated_glyph`

| Property | Value |
|---|---|
| Name | `generated_glyph` |
| Shape | `[B, 1, 128, 128]` |
| Dtype | `float32` |
| Value range | `[-1.0, 1.0]` |

---

## Character Index Mapping

```
Index  Char    Index  Char
  0     А        33     а
  1     Б        34     б
  2     В        35     в
  3     Г        36     г
  4     Д        37     д
  5     Е        38     е
  6     Ё        39     ё
  7     Ж        40     ж
  8     З        41     з
  9     И        42     и
 10     Й        43     й
 11     К        44     к
 12     Л        45     л
 13     М        46     м
 14     Н        47     н
 15     О        48     о
 16     П        49     п
 17     Р        50     р
 18     С        51     с
 19     Т        52     т
 20     У        53     у
 21     Ф        54     ф
 22     Х        55     х
 23     Ц        56     ц
 24     Ч        57     ч
 25     Ш        58     ш
 26     Щ        59     щ
 27     Ъ        60     ъ
 28     Ы        61     ы
 29     Ь        62     ь
 30     Э        63     э
 31     Ю        64     ю
 32     Я        65     я
```

To generate all 66 glyphs for a font, iterate `char_index` 0–65 with a fixed
`style_glyphs` tensor.

---

## Latin Style Characters (required)

The 10 Latin characters to render from the user's font, **in this order**:

```
Index 0  → "A"
Index 1  → "B"
Index 2  → "H"
Index 3  → "O"
Index 4  → "g"
Index 5  → "n"
Index 6  → "o"
Index 7  → "p"
Index 8  → "s"
Index 9  → "x"
```

These specific characters were selected for maximum structural diversity
(enclosed counters, diagonals, ascenders, descenders, curved strokes).

---

## Preprocessing Pipeline (font file → model input)

```
User uploads font file (.ttf / .otf / .woff2)
          │
          ▼
1. Parse font with opentype.js (browser) or fonttools (Node/server).
   - If WOFF2: decompress to raw TTF bytes first.

2. For each of the 10 Latin chars ["A","B","H","O","g","n","o","p","s","x"]:
   a. Render glyph to an offscreen <canvas> at 128×128 px.
      - Fill background with white (#FFFFFF).
      - Draw glyph centred, scaled to fill ~80% of the canvas, in black (#000000).
   b. Extract ImageData (RGBA), convert to single-channel (grayscale):
        gray = 0.299*R + 0.587*G + 0.114*B   (or just use R channel for black/white)
   c. Normalise pixel values:
        pixel_norm = (pixel_gray / 255.0 - 0.5) / 0.5
        → result is in [-1.0, 1.0]
   d. Store as Float32Array of length 128*128.

3. Stack 10 glyph arrays into style_glyphs tensor:
   shape: [1, 10, 1, 128, 128]
   layout (C-order): [batch=0][glyph_i][channel=0][row][col]

4. Set char_index tensor:
   shape: [1]   dtype: Int64   value: (lookup table above)
```

---

## Postprocessing Pipeline (model output → canvas pixel data)

```
generated_glyph : [1, 1, 128, 128]  float32  values in [-1, 1]
          │
          ▼
1. Denormalise: pixel_255 = (pixel_norm + 1.0) / 2.0 * 255.0
   Clamp to [0, 255], convert to Uint8.

2. The result is a grayscale 128×128 image (white background, black glyph).
   Shape: [128, 128]  Uint8.

3. Write to <canvas> ImageData (RGBA):
     R = G = B = pixel_255
     A = 255

4. Optional: scale canvas to desired display size (CSS or drawImage).

5. Vectorisation (for final font assembly):
   - Pass the Uint8 bitmap to potrace / imagetracerjs in the browser.
   - Receive SVG path data.
   - Use opentype.js to encode the path as an OpenType glyph contour.
```

---

## Expected Inference Time

| Hardware | Backend | Time per glyph |
|---|---|---|
| Mid-range laptop GPU (GTX 1060 / RX 580) | WebGL | ~15–30 ms |
| Modern CPU (i7 / Ryzen 7, 8-core) | WASM (4 threads) | ~80–150 ms |
| Low-end CPU (mobile, single-core) | WASM (1 thread) | ~300–600 ms |

To generate a complete 66-glyph font set:
- WebGL: ~1–2 seconds
- WASM 4-thread: ~5–10 seconds
- WASM 1-thread: ~20–40 seconds

**Recommendation:** Use the WebGL backend when available, fall back to WASM.
Run inference in a Web Worker to avoid blocking the UI thread.

---

## ONNX Runtime Web Integration Sketch

```typescript
import * as ort from 'onnxruntime-web';

// Load model once, cache the session.
const session = await ort.InferenceSession.create('/models/v1/generator.onnx', {
  executionProviders: ['webgl', 'wasm'],
});

async function generateGlyph(
  styleGlyphs: Float32Array,   // length: 10 * 128 * 128 (pre-normalised)
  charIndex: number,
): Promise<Float32Array> {
  const styleTensor = new ort.Tensor('float32', styleGlyphs, [1, 10, 1, 128, 128]);
  const charTensor  = new ort.Tensor('int64', BigInt64Array.from([BigInt(charIndex)]), [1]);

  const results = await session.run({
    style_glyphs: styleTensor,
    char_index:   charTensor,
  });

  return results['generated_glyph'].data as Float32Array;
}
```

---

## Versioning

| Version | File | Notes |
|---|---|---|
| v1 | `models/v1/generator.onnx` | MVP, 128×128, 66 Russian Cyrillic chars |

Future versions will increment the directory (`v2/`, `v3/`) and update this document.

---

*Last updated: 2026-02-25 — Major (AI/ML Engineer)*
