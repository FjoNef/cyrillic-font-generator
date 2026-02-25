# Cyrillic Font Generator — ML Model

Conditional GAN (pix2pix-style) that generates Cyrillic glyphs matching a target font's visual style.

## Architecture

```
User font file
    │
    ▼
StyleEncoder (CNN)
  Input:  N Latin reference glyphs  [N, 1, 128, 128]
  Output: style embedding           [256]
    │
    ▼ (concat with char embedding)
UNetGenerator
  Input:  style embedding + char index embedding  [512, 8, 8]
  Output: generated Cyrillic glyph                [1, 128, 128]
```

**StyleEncoder** — Small CNN that encodes N Latin reference glyphs (default: 10) into a 256-dim style vector. Captures stroke weight, serif style, slant, and contrast.

**UNetGenerator** — U-Net with skip connections. Takes the style embedding (tiled spatially) concatenated with a character embedding and decodes a 128×128 grayscale glyph image.

**PatchDiscriminator** — 70×70 PatchGAN discriminator. Operates on (style_glyph, generated_glyph) pairs.

## Scope

- **Target charset:** Russian Cyrillic only — 33 characters
  - Uppercase: А Б В Г Д Е Ё Ж З И Й К Л М Н О П Р С Т У Ф Х Ц Ч Ш Щ Ъ Ы Ь Э Ю Я
  - Lowercase: а б в г д е ё ж з и й к л м н о п р с т у ф х ц ч ш щ ъ ы ь э ю я
- **Character indices:** 0–32 (uppercase А=0 → Я=32), mirrored for lowercase (а=33 → я=65)
  - See `export/inference_contract.md` for full index table

## Training Data

Google Fonts fonts that have **both Latin and Cyrillic** coverage, OFL-licensed.

Download with:
```bash
python src/model/data/download_fonts.py
```

Fonts are saved to `data/fonts/` (gitignored). Expect ~400 fonts after download.

## Setup

```bash
cd src/model
pip install -r requirements.txt
```

## Training

```bash
python train/train.py --config configs/train_config.yaml
```

Checkpoints are saved every 10 epochs to `models/checkpoints/`.

Training on a single GPU (RTX 3080 or better) takes roughly 4–6 hours for 200 epochs with a dataset of 400 fonts.

## ONNX Export

```bash
python export/export_onnx.py --checkpoint models/checkpoints/epoch_200.pth --output models/v1/generator.onnx
```

The export script:
1. Loads the trained checkpoint
2. Combines StyleEncoder + UNetGenerator into a single ONNX graph
3. Applies float16 quantization
4. Validates with `onnxruntime`
5. Saves to `models/v1/generator.onnx` (target: < 20MB compressed)

## Inference API Contract

See `export/inference_contract.md` for the exact tensor shapes, dtypes, and preprocessing steps that the browser client must implement.

## File Structure

```
src/model/
├── README.md                  ← This file
├── requirements.txt           ← Python dependencies
├── configs/
│   └── train_config.yaml      ← Hyperparameters
├── data/
│   ├── dataset.py             ← PyTorch Dataset
│   └── download_fonts.py      ← Google Fonts downloader
├── train/
│   ├── model.py               ← Model definitions
│   └── train.py               ← Training loop
└── export/
    ├── export_onnx.py         ← ONNX export script
    └── inference_contract.md  ← API contract for browser client
```
