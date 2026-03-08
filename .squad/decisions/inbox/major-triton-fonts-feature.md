# torch.compile + num_fonts Configuration

**Date:** 2026-03-08  
**By:** Major (AI/ML Engineer)  
**Status:** IMPLEMENTED  
**Issue:** #46  
**PR:** #47  

---

## Context

After Triton installation on Windows, torch.compile is now functional (previously failed in issue #42). This opened the opportunity to benchmark torch.compile performance and determine if it should be enabled by default for training.

Additionally, the user requested a configurable font count option to control dataset size for quick experiments.

---

## Decision 1: torch.compile Default Configuration

**Choice:** Set `use_compile: false` by default in `train_config.yaml`.

**Rationale:**

1. **Marginal benefit:** Benchmark shows 1.08× speedup (7.9% improvement: 6.25s → 5.79s/epoch on 1000-sample synthetic at B=64). This is below the 10% threshold for "significant" speedup.

2. **Significant first-epoch overhead:** Compilation takes ~108s on first epoch (vs. 16.91s baseline). For short training runs (e.g., 10-epoch experiments), this overhead is not amortized.

3. **Conservative default:** Users can enable `use_compile: true` manually for long training runs (200+ epochs) where the amortized cost is negligible (+0.54s/epoch average over 200 epochs).

4. **Graceful fallback:** The implementation wraps torch.compile in try/except with fallback to eager mode if compilation fails (e.g., on CPU or if Triton is unavailable). This ensures training always works.

**Implementation:**

```yaml
training:
  use_compile: false  # torch.compile support (requires PyTorch 2.0+ and Triton on CUDA)
```

```python
# In train.py
use_compile = train_cfg.get("use_compile", False)
if use_compile:
    if device.type != "cuda":
        print("[!] torch.compile requires CUDA on Windows. Skipping.")
        use_compile = False
    else:
        try:
            generator = torch.compile(generator)
            discriminator = torch.compile(discriminator)
            print("✓ Compilation successful")
        except Exception as e:
            print(f"[!] torch.compile failed: {e}. Falling back to eager mode.")
            use_compile = False
```

**Documentation:** Updated `TRAINING.md` Strategy 4 section with benchmark results and recommendation.

---

## Decision 2: num_fonts Configuration Level

**Choice:** `num_fonts` limits the number of **font families** (not glyphs, not style references).

**Rationale:**

1. **Semantic clarity:** When training font-style transfer models, the most meaningful variable is "how many different font families did the model see?". This directly maps to generalization capability.

2. **Consistent sample count per font:** Each font family produces exactly 66 training samples (one per Cyrillic character). Setting `num_fonts=10` gives 660 samples total. Setting `num_fonts=100` gives 6,600 samples. This linearity is intuitive for users.

3. **Style reference count stays constant:** The 10 style glyphs (A, B, C, D, E, H, I, O, R, X) are per-sample metadata, not a configurable axis. Changing this would require retraining from scratch with a different architecture.

4. **Cache-friendly:** When using `CachedFontDataset`, limiting `num_fonts` directly maps to limiting how many `.pt` cache files are loaded. This is efficient and simple to implement.

**Implementation:**

```yaml
data:
  num_fonts: 10  # Optional: limit to first N fonts (sorted alphabetically). Default: null (all fonts)
```

```python
# In dataset.py
class CyrillicFontDataset(Dataset):
    def __init__(
        self,
        fonts_dir: str | Path,
        num_fonts: int | None = None,
        ...
    ):
        all_font_paths = [str(p) for p in self.fonts_dir.rglob("*.?tf") if _font_has_coverage(...)]
        if num_fonts is not None and num_fonts > 0:
            all_font_paths = sorted(all_font_paths)[:num_fonts]
        self.font_paths = all_font_paths
```

**Example use cases:**
- **Quick architecture test:** `num_fonts: 10` → 660 samples, ~3s/epoch
- **Overfitting test:** `num_fonts: 50` → 3,300 samples, train to convergence to verify model capacity
- **Medium-scale experiment:** `num_fonts: 200` → 13,200 samples, ~1 min/epoch on RTX 3070

---

## Implementation

**Files changed:**
- `src/model/configs/train_config.yaml` — added `use_compile` and `num_fonts` config options
- `src/model/train/train.py` — torch.compile integration + wire `num_fonts` to dataset construction
- `src/model/data/dataset.py` — added `num_fonts` parameter to `CyrillicFontDataset` and `CachedFontDataset`
- `src/model/TRAINING.md` — updated torch.compile benchmark section
- `src/model/train/benchmark_compile.py` — automated benchmark script

**Benchmark script:** `train/benchmark_compile.py` automates the before/after comparison. Runs 3 epochs each with and without compile, reports median epoch time and speedup percentage.

---

## Alternatives Considered

### torch.compile: Enable by default

**Rejected:** The 108s first-epoch overhead hurts developer experience for short training runs. Many users will run 10–50 epoch experiments during architecture exploration. A 108s upfront cost is not acceptable for a <10% speedup.

### num_fonts: Limit style glyphs instead

**Rejected:** Style glyphs are architectural metadata (10 Latin characters: A, B, C, D, E, H, I, O, R, X). Changing this requires retraining with a different input shape. Not suitable for quick dataset experiments.

### num_fonts: Limit Cyrillic characters

**Rejected:** This would break the training contract. The model is trained to predict 66 Cyrillic characters (Russian uppercase + lowercase). Subsampling the Cyrillic alphabet would require changing the model's final char_embedding layer and would not generalize to the full alphabet at inference time.

---

## Benchmark Details

**Hardware:** RTX 3070 Laptop GPU (8GB VRAM, 40 SMs), 16-core CPU  
**Software:** PyTorch 2.10.0+cu128, Python 3.14.3, Windows 11  
**Dataset:** 1000 synthetic samples, batch_size=64, AMP enabled, cudnn.benchmark enabled  

| Config | Epoch 1 | Epoch 2 | Epoch 3 | Median | Notes |
|--------|---------|---------|---------|--------|-------|
| Baseline | 16.91s | 6.25s | 6.18s | **6.25s** | Includes cudnn warm-up |
| torch.compile | 124.07s | 5.62s | 5.79s | **5.79s** | Includes Triton compilation |

**Speedup:** 1.08× (7.9% improvement)  
**Compilation overhead:** 107.16s (124.07s - 16.91s baseline warm-up)  
**Amortized cost (200 epochs):** +0.54s/epoch average  
**Break-even point:** ~157 epochs (where amortized overhead equals cumulative speedup gain)

---

## Related

- **Issue #42:** Training speed profiling — identified torch.compile as unavailable on Windows (Triton missing)
- **PR #45:** Training speed optimization — baseline 5.47s/epoch achieved without torch.compile
- **Issue #46:** torch.compile + num_fonts feature request (this decision)
- **PR #47:** Implementation (torch.compile + num_fonts)

---

## Learnings

1. **torch.compile on Windows requires CUDA:** The CPU backend requires a C++ compiler (cl.exe from MSVC) which is often not available. Always test with CUDA tensors on Windows.

2. **Compilation overhead is significant:** For GANs with ~20M parameters, first-epoch compilation can take 100+ seconds. This must be documented and opt-in, not a surprise.

3. **Font-level limiting is most intuitive:** When experimenting with dataset size, users think in terms of "how many fonts" not "how many samples" or "how many style glyphs per sample".

4. **Sub-10% speedups are marginal:** Speedups below 10% are often within noise margin and not worth imposing on all users. Make them opt-in.
