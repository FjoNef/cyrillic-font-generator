# Decision: Restore Training Speed Optimization to dev

**Date:** 2026-03-08  
**Author:** Major (AI/ML Engineer)  
**Context:** Revert commit `1d8ec45` (which reverted `aa89456`) was a mistake. fjodo requested restoration.

## Decision

Reverted the revert commit (`1d8ec45`) on `dev` via `git revert 1d8ec45 --no-edit`, restoring:
- `src/model/data/build_cache.py` — pre-render fonts to .pt cache files
- `src/model/data/dataset.py` — `CachedFontDataset` class
- `src/model/configs/train_config.yaml` — `fonts_cache_dir` config key
- `src/model/train/profile_real_data.py` — real data profiling script
- `src/model/train/profile_training.py` — training loop profiler
- `src/model/train/train.py` — `CachedFontDataset` wiring
- `src/model/TRAINING.md` — full profiling documentation

## Conflict Resolution

PR #47 (`c67a03b`) was already merged when the revert was done. It had stripped `CachedFontDataset` references during its own conflict resolution. The revert-of-revert produced conflicts in:
- `TRAINING.md`: Kept Performance Tuning section (optimization); updated Strategy 4 with PR #47's real torch.compile benchmarks (it works now)
- `train_config.yaml`: Kept both `num_fonts` comment (from PR #47) AND `fonts_cache_dir` comment (from optimization)

## Follow-up Fix

`CachedFontDataset` lacked the `num_fonts` parameter that PR #47 had wired into `CyrillicFontDataset`. Added `num_fonts` to `CachedFontDataset.__init__` and wired it through `train.py`. Re-enabled the previously-skipped `test_cached_dataset_num_fonts_limit` test.

## Final State

- 22 tests pass, 1 skipped (compile on CPU — expected)
- dev HEAD: `d2519bc`
- All optimization code restored + all PR #47 features intact
