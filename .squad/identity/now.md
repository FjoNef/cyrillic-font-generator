---
updated_at: 2026-02-26T15:23:02Z
focus_area: CUDA PyTorch installed, smoke-tested, training command staged. Ready to launch on user signal.
active_issues: []
---

# Now

**Last updated:** 2026-02-26  
**Focus:** GPU environment production-ready. Full training staged, awaiting launch signal.

## Current state
- Branch: dev (aligned with origin/dev)
- PR #13: fix/togusa-ci-ts-errors — ✅ MERGED
- PR #14: fix/togusa-ci-test-failures — ✅ MERGED
- PR #15: fix/togusa-opentype-vitest-interop — ✅ MERGED
- **CI Pipeline:** ✅ All 11 steps pass (build frontend, type check, vitest, dotnet restore/build/test)
- **Frontend Tests:** 41/41 passing across 5 suites
- **Backend Tests:** ✅ All passing
- **PyTorch:** ✅ torch 2.10.0+cu128 — CUDA: True — Device: NVIDIA GeForce RTX 3070 Laptop GPU
- **Smoke test:** ✅ 2 epochs ran clean on CUDA (~3.3 it/s on synthetic data)
- **Training command:** ✅ Staged and ready (see below)

## Next up
1. **User/Major** — Launch full training run (ready to execute):
   ```powershell
   cd C:/Users/fjodo/RiderProjects/cyrillic-font-generator/src/model
   Start-Process python -ArgumentList "train/train.py --config configs/train_config.yaml --num_epochs 200" -RedirectStandardOutput "..\..\models\training.log" -RedirectStandardError "..\..\models\training_err.log" -NoNewWindow -PassThru | Select-Object Id
   ```
   **Est. runtime:** 4–8 hours on RTX 3070 Ti  
   **Log files:** `models/training.log`, `models/training_err.log`

2. **Major** — ONNX export to `models/v1/generator.onnx` after training completes
3. **Saito** — End-to-end smoke test: upload font → generate Cyrillic → download OTF

## Blocker
None. GPU environment fully ready. Training launch deferred (decision in decisions.md).
