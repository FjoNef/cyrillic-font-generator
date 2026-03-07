# Training Guide — RTX 3070Ti Laptop GPU

This guide covers running training on an **NVIDIA RTX 3070Ti Laptop GPU** (8 GB VRAM,
6144 CUDA cores) and documents all performance optimizations applied to `train/train.py`.

---

## Prerequisites

```bash
cd src/model
pip install -r requirements.txt

# Download fonts (requires internet)
python data/download_fonts.py
```

---

## Running Training

```bash
# Standard training with config file
python train/train.py --config configs/train_config.yaml

# Synthetic data — no fonts required (pipeline validation / CI)
python train/train.py --synthetic --batch_size 16 --num_epochs 10

# Resume from checkpoint
python train/train.py --config configs/train_config.yaml --resume models/checkpoints/epoch_0050.pth

# Override batch size (tune for your VRAM)
python train/train.py --config configs/train_config.yaml --batch_size 48
```

---

## GPU Power Mode (Laptop)

The RTX 3070Ti Laptop GPU has a lower TDP than the desktop variant. For maximum
sustained clock speeds during training, set the system to **High Performance** mode
before starting.

### Windows
1. Open **Control Panel → Power Options** (or search "Power plan" in Start).
2. Select **High Performance** (or **Ultimate Performance** if visible).
3. In NVIDIA Control Panel → **Manage 3D Settings → Power management mode** → set to
   **Prefer Maximum Performance**.

### Linux
```bash
# Set CPU governor to performance
sudo cpupower frequency-set -g performance

# Set NVIDIA persistence mode and maximum GPU clocks
sudo nvidia-smi --persistence-mode=1
sudo nvidia-smi -pl <TDP_WATTS>   # e.g. 115 for RTX 3070Ti Laptop

# Or via nvidia-settings
nvidia-settings -a "[gpu:0]/GPUPowerMizerMode=1"
```

---

## Monitoring GPU Utilisation

Use `nvidia-smi dmon` to continuously sample GPU metrics during training:

```bash
# Sample GPU utilisation, memory, temperature every second
nvidia-smi dmon -s u -d 1

# More detailed: utilisation + memory + power + clocks
nvidia-smi dmon -s ucmp -d 1
```

Columns: `sm` = shader utilisation (%), `mem` = memory controller (%), `enc`/`dec` = encode/decode.

Target: `sm` ≥ 90 % during forward/backward passes. If you see low utilisation, the
DataLoader is likely bottlenecking — increase `num_workers`.

---

## Benchmarking Epoch Time

The training loop prints a timing line after every epoch:

```
Epoch 0001 completed in 47.3s
```

To compare before/after a change, capture a few epoch times under identical conditions
(same data, same batch size, same GPU power mode). Use the first epoch as warm-up and
average epochs 2–5.

---

## Performance Optimizations Applied

### 1. Automatic Mixed Precision (AMP / FP16)

**File:** `train/train.py`

```python
try:
    from torch.amp import GradScaler, autocast as _autocast_fn  # PyTorch ≥ 2.0
    def autocast(enabled=True): return _autocast_fn("cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # PyTorch < 2.0

use_amp = device.type == "cuda"
scaler_g = GradScaler(enabled=use_amp)
scaler_d = GradScaler(enabled=use_amp)

# In the training loop:
with autocast(enabled=use_amp):
    loss_d = ...
scaler_d.scale(loss_d).backward()
scaler_d.step(opt_d)
scaler_d.update()
```

- **Why:** The RTX 3070Ti has 256 Tensor Cores that operate at 2× throughput in FP16
  vs FP32. AMP automatically casts eligible ops (Conv, MatMul) to FP16 while keeping
  numerically sensitive ops (loss scaling, BatchNorm) in FP32.
- **Expected gain:** 1.5–2× faster on Tensor Core workloads. Also halves activation
  memory usage, allowing larger batch sizes.
- **GradScaler:** Prevents FP16 underflow. Two separate scalers are used (one per
  optimizer) since GAN training has two independent backward passes.

### 2. cuDNN Benchmark Mode

**File:** `train/train.py`

```python
torch.backends.cudnn.benchmark = True
```

- **Why:** Tells cuDNN to benchmark and cache the fastest convolution algorithm for the
  fixed input shapes used in this model (all convolutions operate on known spatial sizes).
  Adds ~30 s overhead on epoch 1 but pays off over 200 epochs.
- **Note:** Only beneficial when input shapes are consistent across batches. Our model
  has fixed 128×128 spatial dims, so this is always appropriate.

### 3. DataLoader — `pin_memory` + `persistent_workers`

**File:** `train/train.py`

```python
num_workers = min(4, os.cpu_count() or 1)
train_loader = DataLoader(
    train_ds,
    batch_size=train_cfg["batch_size"],
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),  # Pinned memory only when CUDA is available
    persistent_workers=(num_workers > 0),  # Workers stay alive between epochs; requires num_workers > 0
)
```

- **`pin_memory`:** Enabled only when training on CUDA. Allocates CPU tensors in
  page-locked memory so CUDA DMA can transfer them to GPU asynchronously, overlapping
  with the GPU compute of the previous batch. Disabled on CPU to avoid unnecessary overhead.
- **`persistent_workers`:** Set to `num_workers > 0` — prevents Python DataLoader workers
  from being destroyed and respawned at the end of each epoch. On Windows this is especially
  impactful because process creation is expensive. Guarded against `num_workers=0` to avoid
  a `ValueError` at DataLoader construction.
- **`num_workers`:** Capped at `min(4, cpu_count)`. The RTX 3070Ti Laptop is paired
  with an 8–16 core CPU — 4 workers is typically optimal. If loading is still a
  bottleneck, try increasing to 6–8.

### 4. Batch Size Tuning

**File:** `configs/train_config.yaml`

```yaml
batch_size: 32  # RTX 3070Ti Laptop (8GB VRAM): try 16–64
```

The dominant memory cost per sample is `style_glyphs` of shape `[B, 10, 1, 128, 128]`
≈ 655 KB per sample at FP32 (≈ 327 KB at FP16 with AMP). For `B=32` that is ~21 MB
for style glyphs alone, plus activations and model parameters (~85 MB at FP32/nf=32).

| Batch size | Approx. VRAM | Notes                          |
|-----------|--------------|-------------------------------|
| 16        | ~2 GB        | Safe baseline                  |
| 32        | ~4 GB        | Default — good balance         |
| 48        | ~5.5 GB      | Try first if AMP is enabled    |
| 64        | ~7 GB        | Maximum recommended for 8 GB   |

Override via CLI: `--batch_size 48`

---

## Expected Performance (RTX 3070Ti Laptop, B=32)

| Configuration                  | Approx. epoch time (1000 samples) |
|--------------------------------|----------------------------------|
| FP32, no benchmark, num_workers=0 | ~120 s                        |
| FP32 + benchmark + workers=4  | ~80 s                           |
| AMP + benchmark + workers=4   | ~45–60 s                         |

Actual times depend on font count (dataset size), disk speed, and power limit.
Always measure on your own hardware after setting High Performance power mode.
