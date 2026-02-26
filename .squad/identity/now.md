---
updated_at: 2026-02-26T14:55:39Z
focus_area: CI now fully green — all 11 steps passing. Next: full training run on GPU
active_issues: []
---

# Now

**Last updated:** 2026-02-26  
**Focus:** CI fully green — ready for training phase

## Current state
- Branch: dev (aligned with origin/dev)
- PR #13: fix/togusa-ci-ts-errors — ✅ MERGED
- PR #14: fix/togusa-ci-test-failures — ✅ MERGED
- PR #15: fix/togusa-opentype-vitest-interop — ✅ MERGED
- **CI Pipeline:** ✅ All 11 steps pass (build frontend, type check, vitest, dotnet restore/build/test)
- **Frontend Tests:** 41/41 passing across 5 suites
- **Backend Tests:** ✅ All passing

## Next up
1. **Major** — Full training run: `python src/model/train/train.py --config src/model/configs/train_config.yaml --num_epochs 200` (~4–8h on GPU)
2. **Major** — ONNX export to `models/v1/generator.onnx` after training completes
3. **Saito** — End-to-end smoke test: upload font → generate Cyrillic → download OTF

## Blocker
None. CI is green. Waiting on compute time for full training run.
