---
updated_at: 2026-02-25T23:58Z
focus_area: Training pipeline — ready for full run
active_issues: []
---

# Now

**Last updated:** 2026-02-25  
**Focus:** Training pipeline — ready for full run

## Current state
- Branch: dev (aligned with origin/dev, all ML fixes merged via PR #10)
- PR #10: feat/major-training-pipeline-fixes — MERGED ✅

## Next up
1. **Major** — Fix synthetic checkpoint path in train.py: `../../models/checkpoints/` → `models/checkpoints/` (minor, non-blocking)
2. **Major** — Full training run: `python src/model/train/train.py --config src/model/configs/train_config.yaml --num_epochs 200` (~4–8h on GPU)
3. **Major** — ONNX export to `models/v1/generator.onnx` after training completes
4. **Saito** — End-to-end smoke test: upload font → generate Cyrillic → download OTF

## Blocker
None. Pipeline is green. Waiting on compute time for full training run.
