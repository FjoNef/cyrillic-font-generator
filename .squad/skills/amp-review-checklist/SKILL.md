# Skill: AMP (Automatic Mixed Precision) Code Review Checklist

**Category:** QA / Code Review  
**Language:** Python / PyTorch  
**Applicable to:** Any PR adding AMP (`GradScaler` + `autocast`) to a PyTorch training loop

---

## When to Use This Checklist

Use this whenever reviewing a PR that:
- Adds `from torch.cuda.amp import GradScaler, autocast` (or the newer `from torch.amp import ...`)
- Modifies a training loop to use `scaler.scale(loss).backward()`
- Adds `autocast` context managers around forward passes

---

## Checklist

### 1. GradScaler Instances

- [ ] **One `GradScaler` per optimizer** — if the training loop has N independent backward passes (e.g., generator + discriminator in a GAN), there must be N separate `GradScaler` instances.
- [ ] **`enabled=use_amp` flag** — GradScaler should be constructed with `enabled=use_amp` (or equivalent) so it is a no-op on CPU. Never create a GradScaler that is always active.
- [ ] **`scaler.update()` called after every `scaler.step(opt)`** — both must be called within each optimizer's step, in that order.

### 2. `autocast` Scoping

- [ ] **Wraps forward pass + loss computation** — the `with autocast(enabled=use_amp):` block must contain the model forward call(s) and loss calculation(s).
- [ ] **Does NOT wrap `backward()`** — `scaler.scale(loss).backward()` must be OUTSIDE the `autocast` block.
- [ ] **`enabled` parameter matches the GradScaler's `enabled`** — typically both are gated on `device.type == "cuda"`.

### 3. Gradient Clipping

- [ ] **If gradient clipping is present**: `scaler.unscale_(optimizer)` MUST be called before `torch.nn.utils.clip_grad_norm_()`. Without unscaling, gradients are in scaled space and clipping will be incorrect.
- [ ] If no gradient clipping is present, this check is N/A.

### 4. CPU/non-CUDA Compatibility

- [ ] `use_amp = device.type == "cuda"` (or similar) — AMP must be disabled when running on CPU or non-CUDA devices.
- [ ] `GradScaler(enabled=False)` and `autocast(enabled=False)` are safe no-ops — verify the code relies on this rather than having separate code paths.
- [ ] `torch.backends.cudnn.benchmark = True` is only set inside `if torch.cuda.is_available():` — must not execute on CPU.

### 5. ONNX Export / Inference Isolation

- [ ] **Export script uses CPU** — `export_onnx.py` (or equivalent) must explicitly set `device = torch.device("cpu")` and run under `torch.no_grad()`.
- [ ] **No `autocast` in export path** — verify the export script does not import or use `autocast`.
- [ ] **Models loaded from checkpoint fresh** — `model.eval()` before export; no training state bleeding into export.

### 6. DataLoader `persistent_workers`

- [ ] **`persistent_workers=True` requires `num_workers > 0`** — if `persistent_workers` is hardcoded to `True`, the `num_workers` expression must guarantee > 0. Best practice: `persistent_workers=num_workers > 0`.
- [ ] Common pattern: `num_workers = min(4, os.cpu_count() or 1)` (always ≥ 1), then `persistent_workers=num_workers > 0`.

### 7. Import Path

- [ ] **Prefer `from torch.amp import GradScaler, autocast`** over `from torch.cuda.amp import ...` — the `torch.cuda.amp` path is deprecated in PyTorch 2.0+ and emits `FutureWarning` in 2.4+.
- [ ] When using the new path, `autocast` requires `device_type='cuda'` argument: `with autocast(device_type='cuda', enabled=use_amp):`.

### 8. Test Coverage

- [ ] **AMP smoke test exists** — at least one test that runs a single training step with `autocast` + `GradScaler` active (or with `enabled=False` for CPU CI) and asserts losses are finite (`not torch.isnan(loss)`, `not torch.isinf(loss)`).
- [ ] **Existing regression tests still pass** — AMP changes must not break any pre-existing tests.
- [ ] Optional: test that `scaler.get_scale()` > 0 after a valid step (confirms no immediate inf/nan in forward pass caused scale collapse).

### 9. Documentation

- [ ] **TRAINING.md / README updated** if training performance changes significantly.
- [ ] Code snippets in docs match actual implementation (e.g., conditional `pin_memory`, import path).

---

## Common Bugs to Look For

| Bug | Symptom | Fix |
|-----|---------|-----|
| `backward()` inside `autocast` | No immediate error, but numerical instability | Move `.backward()` outside `autocast` block |
| Single `GradScaler` for two optimizers | Scale updates are incorrect; one backward's overflow masks the other | One scaler per optimizer |
| `persistent_workers=True` with `num_workers=0` | `ValueError` at DataLoader construction | `persistent_workers=num_workers > 0` |
| `scaler.unscale_()` not called before grad clip | Gradient clipping operates on scaled gradients; effective clip threshold is wrong | Call `unscale_()` before clip |
| AMP active during ONNX export | Export may fail or produce FP16-typed ONNX nodes | Use `torch.no_grad()` + CPU device in export; no `autocast` |
| `GradScaler` without `enabled=use_amp` | Fails on CPU with `RuntimeError: No CUDA device` | Always use `enabled=use_amp` |

---

## Reference

- First applied in this repo: PR #43 (issue #42) — `src/model/train/train.py`
- PyTorch AMP docs: https://pytorch.org/docs/stable/amp.html
