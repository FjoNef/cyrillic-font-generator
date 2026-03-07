# Decision: AMP Strategy & DataLoader Config for GPU Training

**Date:** 2026-03-08  
**Author:** Major  
**Issue:** #42 — perf(training): optimize training pipeline for NVIDIA RTX 3070Ti Laptop GPU  
**PR:** #43

---

## Decisions Made

### 1. Two GradScalers (one per optimizer) for GAN AMP training

**Decision:** Use separate `GradScaler` instances (`scaler_g`, `scaler_d`) — one for the generator optimizer, one for the discriminator optimizer.

**Rationale:** GAN training has two fully independent backward passes per iteration. A single shared scaler would require careful ordering of `.step()` and `.update()` calls and can cause incorrect scale updates if one loss overflows but the other doesn't. Two scalers make the independence explicit and are the canonical pattern for multi-optimizer AMP training.

**Implementation:**
```python
use_amp = device.type == "cuda"
scaler_g = GradScaler(enabled=use_amp)
scaler_d = GradScaler(enabled=use_amp)
```

### 2. `GradScaler(enabled=use_amp)` — CPU-compatible AMP

**Decision:** Always instantiate `GradScaler` with `enabled=use_amp` flag (False on CPU).

**Rationale:** Makes the training script run correctly on CPU (no CUDA) without code branches. `GradScaler(enabled=False)` is a pass-through no-op.

### 3. DataLoader `persistent_workers=True` — always on (when num_workers > 0)

**Decision:** `persistent_workers=True` unconditionally alongside `num_workers=min(4, cpu_count)`.

**Rationale:** On Windows, Python subprocess creation is expensive. Keeping workers alive between epochs eliminates per-epoch respawn latency (~1–3 s on Windows per epoch). No downside for training workloads with fixed datasets.

### 4. `torch.backends.cudnn.benchmark = True` — always on when CUDA present

**Decision:** Set `cudnn.benchmark = True` unconditionally after detecting CUDA.

**Rationale:** All convolutions in this model operate on fixed spatial dims (128×128). cuDNN benchmark mode auto-selects the fastest algorithm on first run and caches it. The ~30 s profiling cost on epoch 1 is justified over 200-epoch training runs. Would need to be disabled if input shapes varied between batches (they don't here).

### 5. Default batch_size stays at 32

**Decision:** Do not change the default batch_size. Add a comment documenting the VRAM envelope (16–64 safe range for 8GB).

**Rationale:** 32 is a well-tested default. With AMP enabled, 48–64 becomes viable, but we leave this as a user tuning decision documented in both TRAINING.md and train_config.yaml. Forcing a higher default could cause OOM on constrained systems.

---

## Impact on Other Agents

- **Togusa:** No impact — inference pipeline unchanged.
- **Batou:** No impact — model delivery format unchanged.
- **Saito:** All 9 existing tests pass. AMP doesn't require new unit tests (the training step contract is the same).
