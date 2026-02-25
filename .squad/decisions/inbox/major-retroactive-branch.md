### 2026-02-25T180000: Retroactive branch created for ML pipeline fixes

**By:** Major (AI/ML Engineer) — requested by FjoNef

**What:** Created `feat/major-training-pipeline-fixes` retroactively to fix branching policy violation. Reset dev back to f07d86a (origin/dev). All ML training pipeline work now lives on the feature branch with PR #10 open for Saito review.

**Why:** Branching policy violation — Major's fixes (3 commits: 8f83be0, da3d162, 102db9b) landed directly on dev instead of a feature branch per .squad/decisions.md branching policy. Corrected via retroactive branching approach:
1. Branch creation from current HEAD captured all work
2. Stage cleanup committed remaining changes (models/logs/ TensorBoard events)
3. Dev reset to origin/dev removed misplaced commits
4. Feature branch pushed to origin with PR opened

**Status:** PR #10 awaiting Saito review before merge to dev.
