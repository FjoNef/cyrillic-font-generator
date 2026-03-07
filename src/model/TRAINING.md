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

---

## Performance Tuning

> Profiled 2026-03-08 on RTX 3070 Laptop GPU (8 GB VRAM) · PyTorch 2.10.0+cu128 · 16-core CPU.
> All timings use `torch.cuda.synchronize()` + `time.perf_counter()` on the **synthetic** 1000-sample dataset
> (32 batch size = 31.25 batches/epoch) unless noted.  Epoch 1 is warm-up; epoch 2 is the measured value.

### Baseline

| Metric | Value |
|---|---|
| Config | B=32, num_workers=4, prefetch_factor=2, AMP on, cudnn.benchmark on |
| Epoch 1 time | **16.71 s** (includes cuDNN algorithm auto-benchmark, one-time cost) |
| Epoch 2 time | **5.47 s** ← measured baseline |
| VRAM allocated | 0.38 GB |
| VRAM reserved | 1.86 GB |
| AMP scaler scale | G=16384, D=32768 (>>1 → FP16 is active and stable) |

Phase breakdown (both epochs):

| Phase | Time | Share |
|---|---|---|
| G_backward | 11.04 s | 58.1% ← **primary bottleneck** |
| D_forward | 3.88 s | 20.4% |
| G_forward | 2.73 s | 14.4% |
| D_backward | 1.21 s | 6.4% |
| data_transfer | 0.13 s | 0.7% |

**The generator backward pass is the bottleneck** — 58% of compute time. Data loading is negligible (0.7%) because the DataLoader pipeline overlaps with GPU compute.

---

### Strategy 1 — `num_workers` sweep

| Config | Epoch 2 time | Notes |
|---|---|---|
| num_workers=0 | 6.24 s | No async prefetch, CPU blocks GPU briefly |
| num_workers=2 | 5.47 s | Same as baseline |
| **num_workers=4** | **5.47 s** | ✓ Optimal — matches baseline |
| num_workers=6 | 5.55 s | No improvement, slight overhead |
| num_workers=8 | 5.55 s | No improvement |

**Finding:** num_workers=4 is optimal. Workers beyond 4 add process management overhead without
improving throughput. This confirms the current default.

---

### Strategy 2 — `batch_size` sweep

| Config | Epoch 2 time | VRAM reserved | Notes |
|---|---|---|---|
| batch_size=32 | 5.47 s | 1.86 GB | Baseline |
| **batch_size=64** | **5.45 s** | 3.14 GB | Marginally faster, better GPU utilisation |
| batch_size=128 | 5.48 s | 6.14 GB | Similar to B=32 at small dataset |

**Finding:** Larger batches do not significantly reduce per-epoch time (total work is the same).
B=64 is marginally fastest and uses VRAM more efficiently without risk of OOM. Updated default.

For the **real 130k-sample dataset**, batch size affects:
- B=32: 4071 batches/epoch × ~175 ms/batch ≈ 712 s GPU compute
- B=64: 2036 batches/epoch × ~349 ms/batch ≈ 710 s GPU compute
- Epoch time is dominated by GPU compute regardless of batch size.

---

### Strategy 3 — `prefetch_factor`

| Config | Epoch 2 time |
|---|---|
| prefetch_factor=2 | 5.47 s |
| prefetch_factor=4 | 5.60 s |

**Finding:** prefetch_factor=2 is slightly faster. Higher prefetch uses more CPU memory and adds
minor overhead without benefit when data loading is not the bottleneck. Keep at 2.

---

### Strategy 4 — `torch.compile(mode="reduce-overhead")`

**Result: FAILED** — Triton is not installed. `torch.compile` on Windows requires
[triton-lang/triton](https://github.com/triton-lang/triton) which does not have an official
Windows release. The code wraps the compile call in `try/except` with graceful eager fallback,
but the failure occurs during the first forward pass, not at compile time.

```
torch._inductor.exc.TritonMissing: Cannot find a working triton installation.
```

**Workaround for Linux:** Install `triton` (`pip install triton`) and then:

```python
try:
    gen = torch.compile(gen, mode="reduce-overhead")
    se = torch.compile(se, mode="reduce-overhead")
    disc = torch.compile(disc, mode="reduce-overhead")
except Exception:
    pass  # graceful fallback to eager mode
```

Expected speedup on Linux/WSL2: ~15–25% on G_backward (the primary bottleneck).

---

### Real Dataset Profiling (1974 fonts × 66 chars = 130,284 samples)

Data loading speed with on-the-fly PIL rendering:

| num_workers | ms/sample | Full-epoch data-loading time |
|---|---|---|
| 0 | 6.91 ms | ~900 s (15 min) |
| 4 | 2.45 ms | ~319 s (5.3 min) |

GPU compute is ~175 ms/batch at B=32 → **712 s (11.9 min) per epoch**.

With `num_workers=4`, data loading (78 ms/batch) is **hidden behind** GPU compute
(175 ms/batch). The pipeline is **GPU-compute-bound**, not data-bound.

**On-the-fly rendering bottleneck:** Each `__getitem__` renders 10 style glyphs + 1 Cyrillic
target = 11 `PIL ImageFont.truetype()` calls. For 130,284 samples/epoch that is **~1.43 million
font render calls per epoch**.

---

### Strategy 5 — Cached `.pt` Dataset

**Implementation:** `data/build_cache.py` + `CachedFontDataset` (in `data/dataset.py`).

Pre-render all fonts to per-font `.pt` files. Each file stores `style_glyphs [10, 1, 128, 128]`
and `target_glyphs [66, 1, 128, 128]` as float32 tensors. `__getitem__` loads the .pt file
(memoised via `functools.lru_cache(maxsize=256)`) and indexes into the pre-loaded tensors.

```bash
# Build cache (one-time, ~10–15 min for 1974 fonts)
cd src/model
python data/build_cache.py --fonts_dir ../../data/fonts --output ../../data/fonts_cache

# Estimated cache size:
#   float32: 1974 × 5 MB ≈ 9.9 GB
#   uint8 (--uint8 flag): 1974 × 1.25 MB ≈ 2.5 GB
```

Enable in `configs/train_config.yaml`:

```yaml
data:
  fonts_cache_dir: "../../data/fonts_cache"   # uncomment to use cached dataset
```

**Expected impact:** Reduces per-sample CPU overhead from ~6.91 ms (w=0 rendering) to ~0.01 ms
(tensor index after first-load). Since `num_workers=4` already makes data loading latent, the
practical speedup for full-dataset training is minimal (~5%). Primary benefit: reduced CPU
utilization frees thermal headroom for GPU boost clock on laptop hardware.

---

### Recommended Configuration (Winning Setup)

```yaml
training:
  batch_size: 64           # Marginally faster than 32; 3.1 GB VRAM
  # DataLoader (in train.py):
  num_workers: 4           # Optimal for 16-core laptop CPU
  prefetch_factor: 2       # Better than 4 when compute-bound
  persistent_workers: true # (automatically set when num_workers > 0)
  pin_memory: true         # (automatically set when CUDA is available)
```

**Achieved epoch time (1000 synthetic samples):** **5.45 s** — well under the 60 s target.

**Limiting factor for real-data training:** GPU compute is the hard wall at ~710 s/epoch
(12 min) for the full 1974-font dataset. This is dominated by G_backward (58% of GPU time —
the UNet decoder + feature-matching loss backward pass). Removing this bottleneck requires
either:
1. `torch.compile` (available on Linux with Triton installed; ~20% gain expected)
2. A smaller training set per "fast epoch" (e.g., sample 400 random fonts per epoch → ~3 min)
3. Reducing model depth (not recommended — would hurt quality)

| Config | Synthetic epoch (1000 samples) | vs 60 s target |
|---|---|---|
| FP32, w=0, no benchmark (old estimate) | ~120 s | ✗ |
| FP32 + benchmark + w=4 | ~80 s | ✗ |
| AMP + benchmark + w=4 (old estimate) | ~45–60 s | borderline |
| **AMP + benchmark + w=4 (measured)** | **5.47 s** | **✓ 11× under target** |
| **AMP + benchmark + w=4 + B=64 (recommended)** | **5.45 s** | **✓ 11× under target** |
