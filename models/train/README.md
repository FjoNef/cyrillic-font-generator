# Training Pipeline for Cyrillic Font Generator cGAN

This directory contains the complete PyTorch training pipeline for the Cyrillic font generation conditional GAN model.

## Architecture

**Conditional GAN (pix2pix-style)** with:
- **StyleEncoder**: Encodes 10 Latin reference glyphs → fixed-size style vector (256-dim)
- **UNetGenerator**: Generates 128×128 Cyrillic glyph conditioned on character index + style vector
- **PatchDiscriminator**: 70×70 PatchGAN discriminator for adversarial training

## Files

- `model.py` — PyTorch model architecture (StyleEncoder, UNetGenerator, PatchDiscriminator)
- `train.py` — Training script with adversarial + L1 loss
- `dataset.py` — Dataset loader for Google Fonts (TTF/OTF files)
- `export.py` — ONNX export script with INT8 quantization
- `requirements.txt` — Python dependencies

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create data directory and download fonts
mkdir -p ../../data/fonts
# Download Google Fonts with Latin+Cyrillic coverage
# Place .ttf/.otf files in ../../data/fonts/
```

## Training

### With Real Fonts

```bash
python train.py \
    --data_dir ../../data/fonts \
    --batch_size 16 \
    --num_epochs 200 \
    --checkpoint_dir ../checkpoints \
    --sample_dir ../samples \
    --log_dir ../logs
```

### With Synthetic Data (for testing)

```bash
python train.py \
    --synthetic \
    --batch_size 16 \
    --num_epochs 50
```

### Resume from Checkpoint

```bash
python train.py \
    --data_dir ../../data/fonts \
    --resume ../checkpoints/epoch_0100.pth \
    --num_epochs 200
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `../../data/fonts` | Directory with TTF/OTF font files |
| `--batch_size` | `16` | Training batch size |
| `--num_epochs` | `200` | Number of training epochs |
| `--lr_g` | `0.0002` | Generator learning rate |
| `--lr_d` | `0.0002` | Discriminator learning rate |
| `--lambda_l1` | `100.0` | L1 loss weight (pixel reconstruction) |
| `--num_workers` | `4` | DataLoader worker threads |
| `--checkpoint_dir` | `../checkpoints` | Where to save `.pth` checkpoints |
| `--sample_dir` | `../samples` | Where to save sample outputs |
| `--log_dir` | `../logs` | TensorBoard log directory |

## Monitoring Training

```bash
# Start TensorBoard
tensorboard --logdir ../logs

# View at http://localhost:6006
```

## Export to ONNX

After training, export the best checkpoint to ONNX:

```bash
python export.py \
    --checkpoint ../checkpoints/epoch_0200.pth \
    --output ../v1/generator.onnx \
    --opset 17
```

This will:
1. Load the PyTorch checkpoint
2. Export to ONNX (opset 17)
3. Apply INT8 dynamic quantization (~50% size reduction)
4. Save to `models/v1/generator.onnx`

### Export Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | (required) | Path to `.pth` checkpoint |
| `--output` | `../v1/generator.onnx` | Where to save ONNX model |
| `--opset` | `17` | ONNX opset version |
| `--no-quantize` | `False` | Disable INT8 quantization |

## Model Contract (LOCKED)

The exported ONNX model must match this exact contract for frontend integration:

### Inputs
- **`style_glyphs`**: `[B, 10, 1, 128, 128]` float32
  - 10 Latin reference characters: A, B, C, D, E, H, I, O, R, X
  - Grayscale normalized to [-1, 1] where **+1.0 = black ink, -1.0 = white background**

- **`char_index`**: `[B]` int64
  - Character index 0–65:
    - 0–32: uppercase А–Я (with Ё at index 6)
    - 33–65: lowercase а–я (with ё at index 39)

### Output
- **`generated_glyph`**: `[B, 1, 128, 128]` float32
  - Grayscale in range [-1, 1] where **+1.0 = black ink, -1.0 = white background**

## Training Data

### Recommended: Google Fonts

Download fonts with both Latin and Cyrillic coverage from [Google Fonts](https://fonts.google.com/):

1. Filter fonts by language: Latin + Cyrillic
2. Download ~100-400 fonts
3. Extract `.ttf` files to `../../data/fonts/`

### Dataset Structure

The `FontDataset` class expects:
```
data/fonts/
├── font1.ttf
├── font2.otf
├── font3.ttf
└── ...
```

Each font file should contain:
- Latin characters: A, B, C, D, E, H, I, O, R, X (for style references)
- Russian Cyrillic: А-Я, а-я, Ё, ё (66 total characters)

### Data Augmentation

The dataset applies random augmentations during training:
- Font size variation: ±12pt
- Rotation: ±5 degrees
- Scale: 0.9–1.1×

## Expected Training Time

On a modern GPU (RTX 3080 or better):
- **200 epochs**: ~4-8 hours (depends on dataset size)
- **Checkpoint saved every 10 epochs**
- **Samples generated every 5 epochs**

## Model Size

- **PyTorch checkpoint (.pth)**: ~50-70 MB
- **ONNX (float32)**: ~40-50 MB
- **ONNX (INT8 quantized)**: ~15-20 MB ✅ Target for browser delivery

## Troubleshooting

### "No fonts found in data directory"
- Ensure `.ttf` or `.otf` files exist in the specified `--data_dir`
- Use `--synthetic` flag to test training pipeline without real fonts

### "CUDA out of memory"
- Reduce `--batch_size` (try 8 or 4)
- Reduce image size (requires code changes, not recommended)

### "Model outputs all white/black"
- Check discriminator/generator balance via TensorBoard
- Try reducing `--lambda_l1` (e.g., 50.0 instead of 100.0)
- Ensure training data is properly normalized [-1, 1]

### Export fails
- Ensure you're using PyTorch 2.0+ with ONNX opset 17 support
- Try `--no-quantize` if quantization fails

## Next Steps

1. **Train the model**: `python train.py --data_dir ../../data/fonts --num_epochs 200`
2. **Monitor progress**: `tensorboard --logdir ../logs`
3. **Export to ONNX**: `python export.py --checkpoint ../checkpoints/epoch_0200.pth`
4. **Verify in browser**: Place `generator.onnx` in `models/v1/` and test in the frontend

## License

Training code and model architecture: MIT License  
Generated fonts: OFL (Open Font License) as per project requirements
