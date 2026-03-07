# Skill: PyTorch AMP Training (GAN / Multi-Optimizer Pattern)

**Category:** Performance Optimization  
**Language:** Python / PyTorch  
**Applicable to:** Any PyTorch GAN or multi-optimizer training loop

---

## What This Skill Is

A reusable pattern for adding Automatic Mixed Precision (AMP / FP16) to a PyTorch
training loop that uses multiple optimizers (e.g., GAN generator + discriminator).

---

## The Pattern

### Imports

```python
from torch.cuda.amp import GradScaler, autocast
```

### Initialization (after device detection)

```python
use_amp = device.type == "cuda"
scaler_a = GradScaler(enabled=use_amp)   # One per optimizer/backward pass
scaler_b = GradScaler(enabled=use_amp)
```

### Training Step (per optimizer)

```python
# Optimizer A (e.g., Discriminator)
opt_a.zero_grad()
with autocast(enabled=use_amp):
    output_a = model_a(inputs)
    loss_a = criterion(output_a, targets)
scaler_a.scale(loss_a).backward()
scaler_a.step(opt_a)
scaler_a.update()

# Optimizer B (e.g., Generator)
opt_b.zero_grad()
with autocast(enabled=use_amp):
    output_b = model_b(inputs)
    loss_b = criterion(output_b, targets)
scaler_b.scale(loss_b).backward()
scaler_b.step(opt_b)
scaler_b.update()
```

---

## Key Rules

1. **One `GradScaler` per optimizer** — prevents incorrect scale updates when one loss
   overflows but the other doesn't.

2. **`enabled=use_amp` flag** — makes the pattern CPU-compatible without code branches.
   `GradScaler(enabled=False)` and `autocast(enabled=False)` are pass-through no-ops.

3. **Wrap the entire forward + loss computation** inside `autocast`. Do not wrap
   `.backward()` — that runs outside `autocast`.

4. **Use `scaler.scale(loss).backward()`** instead of `loss.backward()`.

5. **Call `scaler.update()` after `scaler.step(opt)`** — updates the loss scale.

6. **`loss.item()` still works** — `.item()` always returns a Python float regardless
   of AMP context.

---

## Supporting Optimizations (pair with AMP)

```python
# After CUDA device detection
torch.backends.cudnn.benchmark = True   # Fixed input shapes only

# DataLoader
DataLoader(
    dataset,
    num_workers=min(4, os.cpu_count() or 1),
    pin_memory=device.type == "cuda",
    persistent_workers=True,            # Eliminates per-epoch worker respawn
)
```

---

## Expected Speedup

| Hardware             | Speedup vs baseline FP32 |
|----------------------|--------------------------|
| NVIDIA Ampere (3xxx) | 1.5–2.0×                 |
| NVIDIA Turing (20xx) | 1.3–1.6×                 |
| CPU                  | no change (no-op)        |

Memory savings: ~40–50% reduction in activation memory (FP16 activations), enabling
larger batch sizes.

---

## Reference Implementation

- `src/model/train/train.py` in this repository (issue #42 / PR #43)
