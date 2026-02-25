# models/ — Trained Model Files

This directory contains exported ONNX model files.

Large model files (`.onnx`, `.pth`) are **gitignored** and are not committed
to the repository.

---

## Pre-trained Model

| Version | File | Size | Download |
|---|---|---|---|
| v1 (MVP) | `v1/generator.onnx` | ~15 MB | [Download generator.onnx v1 — ~15MB](TODO) |

> **TODO:** Replace the download link above with the actual hosted URL
> once the model has been trained and uploaded (e.g. GitHub Releases, Azure Blob Storage).

---

## Versioning Scheme

```
models/
├── v1/
│   └── generator.onnx       ← MVP model (128×128, 66 Russian Cyrillic chars)
├── v2/
│   └── generator.onnx       ← Future: higher quality / resolution
├── checkpoints/
│   ├── epoch_0010.pth
│   ├── epoch_0020.pth
│   └── …
└── samples/
    └── epoch_0200/          ← Generated sample images logged during training
        ├── А_roboto.png
        └── …
```

- Each `vN/` directory is a **complete, standalone inference artefact**.
- Increment the version when the model architecture or training data changes.
- `checkpoints/` holds PyTorch `.pth` files from the training loop (not for inference).
- `samples/` holds PNG previews generated during training for quality monitoring.

---

## How to Export a New Model

```bash
python src/model/export/export_onnx.py \
    --checkpoint models/checkpoints/epoch_0200.pth \
    --output     models/v1/generator.onnx
```

See `src/model/export/inference_contract.md` for the ONNX input/output specification.
