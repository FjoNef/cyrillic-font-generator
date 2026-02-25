# models/v1/ — Production Model Files

This directory contains the exported ONNX model file for browser inference.

## ⚠️ Model Not Yet Trained

The `generator.onnx` model file **does not exist yet**. Follow these steps to train and export it:

### 1. Train the Model

```bash
cd models/train

# Install dependencies
pip install -r requirements.txt

# Train with real fonts (recommended)
python train.py \
    --data_dir ../../data/fonts \
    --batch_size 16 \
    --num_epochs 200 \
    --checkpoint_dir ../checkpoints

# OR train with synthetic data (for testing)
python train.py --synthetic --num_epochs 50
```

Training will save checkpoints to `models/checkpoints/epoch_NNNN.pth`.

### 2. Export to ONNX

After training completes, export the best checkpoint:

```bash
cd models/train

python export.py \
    --checkpoint ../checkpoints/epoch_0200.pth \
    --output ../v1/generator.onnx
```

This will create `models/v1/generator.onnx` (~15-20 MB after INT8 quantization).

### 3. Verify in Browser

Once `generator.onnx` exists in this directory:
1. The .NET backend will serve it from this location
2. The React frontend will load it via ONNX Runtime Web
3. Users can upload a font and generate Cyrillic glyphs

## Model Specification

The ONNX model must implement this exact contract:

### Inputs

- **`style_glyphs`**: `[B, 10, 1, 128, 128]` float32
  - 10 Latin reference characters rendered at 128×128 grayscale
  - Characters: A, B, C, D, E, H, I, O, R, X
  - Normalized to [-1, 1] where **+1.0 = black ink, -1.0 = white**

- **`char_index`**: `[B]` int64
  - Character index 0–65
  - 0–32: uppercase А–Я (Ё at index 6)
  - 33–65: lowercase а–я (ё at index 39)

### Output

- **`generated_glyph`**: `[B, 1, 128, 128]` float32
  - Generated Cyrillic glyph at 128×128 grayscale
  - Range [-1, 1] where **+1.0 = black ink, -1.0 = white**

## Expected Model Size

| Format | Size | Notes |
|--------|------|-------|
| PyTorch checkpoint (.pth) | ~50-70 MB | Training artifact, not for production |
| ONNX (float32) | ~40-50 MB | Unquantized |
| **ONNX (INT8 quantized)** | **~15-20 MB** | **Production target** ✅ |

## Frontend Integration

The frontend expects the model at this exact path:

```
models/v1/generator.onnx
```

The .NET backend serves it with:
- **URL**: `/api/model/v1/generator.onnx`
- **Immutable cache headers** for browser caching
- **Range request support** for progressive loading

## Browser Inference

- **ONNX Runtime Web** loads the model in the browser
- **Preferred backend**: WebGL (GPU acceleration)
- **Fallback backend**: WASM (4-thread CPU)
- **Expected inference time**: ~15-30ms per glyph (WebGL) or ~80-150ms (WASM)
- **Inference runs in Web Worker** to avoid UI blocking

## Development

During development, if `generator.onnx` does not exist:
- The .NET backend returns HTTP 404
- The React frontend shows "Model not found" error
- Users cannot generate glyphs until the model is trained and placed here

## See Also

- [Training README](../train/README.md) — Full training pipeline documentation
- [Inference Contract](../../src/frontend/src/inference/worker/inferenceWorker.ts) — Frontend integration code
- [Decisions Log](../../.squad/decisions.md) — Model architecture and contract decisions