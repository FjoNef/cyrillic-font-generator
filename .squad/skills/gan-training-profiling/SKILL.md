# Skill: GAN Training Speed Profiling on CUDA

## Summary

How to profile a PyTorch GAN training loop, identify the true bottleneck,
and apply targeted optimizations. Based on profiling the Cyrillic Font
Generator cGAN on RTX 3070 Laptop GPU.

---

## Phase-Level Profiling Pattern

Use `torch.cuda.synchronize()` before each timing boundary to get accurate
wall-clock measurements that reflect actual GPU completion, not just CPU launch.

```python
import time
import torch

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# Timer accumulator (call sync() before start/stop)
class PhaseTimer:
    def __init__(self):
        self._acc: dict[str, float] = {}
        self._t0 = None
        self._phase = None

    def start(self, phase: str):
        sync()
        self._t0 = time.perf_counter()
        self._phase = phase

    def stop(self):
        sync()
        dt = time.perf_counter() - self._t0
        self._acc[self._phase] = self._acc.get(self._phase, 0.0) + dt

    def report(self) -> dict[str, float]:
        return dict(self._acc)
```

Wrap each phase in the training loop:

```python
timer = PhaseTimer()

for batch in loader:
    timer.start("data_transfer")
    x = x.to(device, non_blocking=True)
    timer.stop()

    timer.start("D_forward")
    # ... D forward + loss
    timer.stop()

    timer.start("D_backward")
    scaler_d.scale(loss_d).backward()
    scaler_d.step(opt_d); scaler_d.update()
    timer.stop()

    # same for G_forward, G_backward ...
```

---

## Key Invariant: Compute vs Data Bottleneck

With `num_workers > 0`, the DataLoader prefetches asynchronously:

- If `data_loading_ms < GPU_compute_ms` per batch: **compute-bound** — no data loading optimization helps
- If `data_loading_ms > GPU_compute_ms`: **data-bound** — add workers, caching, or reduce augmentation

**Check:** `data_transfer` phase < 5% of total time → you're compute-bound.

---

## AMP Health Check

```python
# After epoch, verify AMP is active:
print(f"scaler_g scale: {scaler_g.get_scale():.0f}")  # should be >> 1 (e.g. 16384)
# If scale == 1.0 or keeps decreasing → AMP is disabled or overflow is occurring
```

---

## Bottleneck Priority for GANs

Typical GAN compute profile on UNet-based architecture:

1. **G_backward** (~55–60%): UNet decoder is deep → long backward graph. Feature-matching loss adds another discriminator forward pass inside the G step.
2. **D_forward** (~20–25%): PatchGAN runs twice per G step (for feature matching) → double the D forward cost.
3. **G_forward** (~12–15%): Forward pass is cheaper than backward.
4. **D_backward** (~6–8%): Cheaper than G because discriminator is shallower.
5. **Data transfer** (~1%): Negligible with `pin_memory=True` + `num_workers > 0`.

**To attack G_backward:**
- `torch.compile(mode="reduce-overhead")` — requires Triton (Linux only, ~20% gain)
- Larger batch size — amortizes per-batch overhead but total compute is the same
- Lighter generator architecture (quality tradeoff — not always acceptable)

---

## num_workers Tuning Rule

On an 8–16 core CPU with DataLoader rendering glyph images:
- `num_workers = min(4, cpu_count())` is consistently optimal
- Beyond 4: process management overhead starts to dominate
- For in-memory / synthetic datasets: 0–2 workers sufficient

```python
num_workers = min(4, os.cpu_count() or 1)
persistent_workers = (num_workers > 0)   # guard required by PyTorch
```

---

## Pre-Rendered Cache for Font/Image Datasets

When `__getitem__` renders images with PIL:

```python
import functools
import torch

@functools.lru_cache(maxsize=256)
def _load_pt(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)

class CachedDataset(Dataset):
    def __getitem__(self, idx):
        cache_path, sub_idx = self._samples[idx]
        data = _load_pt(cache_path)       # cached in worker memory after first load
        return data["tensors"][sub_idx]   # O(1) index
```

`lru_cache(maxsize=256)` memoises per-worker-process. With `num_workers=4`, each worker
caches up to 256 files independently. Effective when the same "group file" (e.g., per-font .pt)
is accessed multiple times within an epoch.

**Build script pattern:**
```python
# Pre-render → save per-group .pt files
for group_id, items in groups.items():
    tensors = [render(item) for item in items]
    torch.save({"tensors": torch.stack(tensors)}, f"{output}/{group_id}.pt")
```

---

## Epoch 1 Warmup

`torch.backends.cudnn.benchmark = True` causes cuDNN to benchmark convolution algorithms
on the first forward pass through each unique input shape. This adds ~11–16 s to epoch 1
on a model with many conv layers. Epochs 2+ run at full speed. This is expected and unavoidable
with benchmark mode on. Document it as "warm-up epoch" in output.

---

## torch.compile Caveats (Windows)

`torch.compile` requires Triton for the `inductor` backend. Triton is Linux-only.
On Windows, `torch.compile()` itself succeeds but the first forward pass raises
`TritonMissing`. Always wrap in try/except around the **entire** training, not just the compile call:

```python
compiled = False
try:
    model = torch.compile(model, mode="reduce-overhead")
    # Warm-up forward pass to trigger compile error early (not silently):
    with torch.no_grad():
        _ = model(dummy_input)
    compiled = True
except Exception as e:
    print(f"torch.compile unavailable: {e}. Using eager mode.")
    model = original_model   # revert
```
