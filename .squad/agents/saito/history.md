# Saito — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **My focus:** Tests, quality, cross-browser validation, inference performance.

## Learnings
<!-- Append new entries below -->

### 2026-02-26: PR #11 — REQUEST CHANGES (stale duplicate)

**PR #11 — feat/fix-checkpoint-paths → dev:**
- **Verdict:** REQUEST CHANGES / CLOSE AS DUPLICATE
- **GitHub self-review restriction → posted as comment**

**What was found:**

1. **Duplicate fix:** The three output path changes (`model_output_dir`, `checkpoint_dir`, `sample_dir`) in `train_config.yaml` were already applied to dev by the PR #10 squash-merge (commit 3e54f39). The feature branch was branched from `f07d86a` (pre-PR-#10), so PR #10 landed the fix first.

2. **Regression risk:** Because the feature branch pre-dates PR #10, it carries stale values for `fonts_dir` (`../../data/fonts` instead of `data/fonts`) and `style_latin_chars` (old incorrect mixed-case list instead of `["A","B","C","D","E","H","I","O","R","X"]`). Merging would cause merge conflicts, and resolving them in favour of the feature branch would break the tensor contract and font loading.

3. **PR dirty state confirmed:** `mergeable_state: "dirty"` — conflicts exist for the stale fields.

4. **No other missed references:** Scanned all of `src/` for `../../models` and `../../data` patterns — only `train_config.yaml` had them, and they are already correct on dev.

5. **train.py synthetic defaults:** Feature branch's `train.py` has no hardcoded `../../` paths. The flagged issue from PR #10 review appears already resolved.

**Patterns learned:**
- Always verify the merge-base of a PR branch, especially after squash-merges to dev. A fix PR created before a squash-merge can duplicate or regress work.
- `git show <merge-base>:file` and `git merge-base` are essential to diagnose stale-branch PRs.
- PRs in `dirty` merge state against a recently-updated dev deserve extra scrutiny for regressions.

### 2026-02-25T180000: PR #10 APPROVED & MERGED — training pipeline fixes

**PR #10 — feat/major-training-pipeline-fixes → dev:**
- **Verdict:** APPROVED (GitHub self-approve restriction → posted as comment, then merged)
- **Squash-merged to dev; feature branch deleted**

**What was verified:**
1. **model.py — UNet skip-connections fixed:** dec5/dec6/dec7 input channels corrected to asymmetric sums (nf*8+nf*4, nf*4+nf*2, nf*2+nf) matching actual encoder output dimensions; duplicate `self.final` ConvTranspose2d removed; forward pass no longer double-concatenates e1; output [B,1,128,128] float32 Tanh [-1,1]. Contract: PASS.
2. **dataset.py — style chars contract enforced:** `DEFAULT_STYLE_CHARS = ["A","B","C","D","E","H","I","O","R","X"]` uppercase-only 10 chars; fontTools casing fixed; TTFont.close() in finally (Windows locking); SyntheticFontDataset produces correct tensor shapes/dtypes. Contract: PASS.
3. **train_config.yaml:** style_latin_chars = ["A","B","C","D","E","H","I","O","R","X"] ✅; paths relative to repo root (data/fonts, models/). Contract: PASS.
4. **download_fonts.py:** fontTools casing + TTFont.close() Windows file locking fix. PASS.
5. **train.py CLI:** --synthetic/--batch_size/--num_epochs all correctly implemented; 1-epoch real-data validation passed (47,388 samples, no crashes). PASS.

**Minor non-blocking finding:** Synthetic default config in train.py hardcodes `../../models/checkpoints/` (old relative path). Does not affect real training (which uses config file). Suggest follow-up fix.

**Patterns learned:**
- Always verify feature branch content via GitHub API when local checkout is on a different branch
- The recurring style chars bug (lowercase in default list) is a known issue that has now been fixed in both dataset.py, train_config.yaml, AND the synthetic defaults (though checkpoint path in synthetic defaults missed)
- SyntheticFontDataset is a good addition for CI — validates pipeline without requiring 718 font downloads

### 2026-02-25T160138: Major & Batou Sessions — Ready for QA Review

**PR #8 — cGAN Training Pipeline (Major):**
- Deliverables: models/train/{model.py, train.py, dataset.py, export.py, requirements.txt, README.md}
- Architecture: StyleEncoder (mean-pooling) + UNetGenerator (blank canvas, bottleneck conditioning) + PatchDiscriminator
- Loss: Adversarial (BCE) + L1 reconstruction (lambda=100)
- ONNX: opset 17, INT8 weight quantization, dynamic batch, float32 activations
- Tensor contract LOCKED: style_glyphs [B,10,1,128,128] → generated_glyph [B,1,128,128], [-1,1] range
- Character mapping: 0-32 uppercase (А–Я with Ё), 33-65 lowercase (а–я with ё), 66 total Russian Cyrillic chars
- **Action:** Validate ONNX export (opset 17, input/output names, shape matching), verify dataset normalization, check training loop convergence metrics

**PR #9 — Backend Integration (Batou):**
- Deliverables: GET /health, GET /api/model endpoints, 4 xUnit integration tests
- Integration tests all passing: health check 200, model endpoint 404 (absent), CORS headers validated
- Safety wrap: Directory.Exists() check prevents test failures when models/ missing
- **Action:** Validate integration tests run successfully, check CORS headers on actual OPTIONS requests, verify /api/model serves ONNX correctly once Major exports it

**QA focus areas:**
1. **PR #8:** ONNX model structure (input/output validation), dataset normalization (-1→1 range correct), training stability
2. **PR #9:** Integration test coverage, CORS behavior, graceful 404 handling for missing model

### 2026-02-25T145755: PR #4 & #5 ready for QA review
- **PR #4 — Inference Pipeline (Togusa):** Web Worker integration complete. Model loading, glyph inference, and UI feedback wired end-to-end. Ready for QA functional testing and performance validation (target: ~15–30ms/glyph WebGL, ~80–150ms/glyph WASM).
- **PR #5 — CI Automation (Batou):** GitHub Actions workflows configured (CI, release, preview, PR auto-label). Ready for QA validation that pipelines run correctly on PRs to dev and pushes to main. Check label creation, reviewer notifications, and release artifact generation.
- **Focus areas for QA:** Inference accuracy on style transfer (test multiple fonts), performance benchmarking, CI pipeline success rate across branches, release automation idempotency, PR auto-label routing correctness.

### 2026-02-25T152700: PR #4 APPROVED — color inversion fix complete
- **Re-review verdict:** APPROVED after Major's fix
- **Fix confirmed:** All instances of pixel mapping now use correct formula `((1 - output[px]) / 2) * 255`
  - Before: `((output[px] + 1) / 2) * 255` — inverted colors (+1→white, -1→black) ❌
  - After: `((1 - output[px]) / 2) * 255` — correct colors (+1→black, -1→white) ✅
- **Files verified:** App.tsx (line 67), OnnxInference.ts (line 90) — both corrected
- **Test coverage:** Added colorMapping.test.ts with 5 passing tests validating the formula
- **Acceptance criteria:** All 5 criteria met (model loading, glyph rendering, 66 chars, Web Worker, .otf download)
- **Outcome:** PR #4 ready to merge → dev

### 2026-02-25T152319: PR #4 blocking bug resolved — color inversion fix
- **Issue:** PR #4 (feat/togusa-inference-pipeline → dev) had inverted color output mapping in App.tsx line 67-70
- **Root cause:** Formula `((output[px] + 1) / 2) * 255` mapped model output incorrectly: -1 (background) → 0 (black), 1 (ink) → 255 (white) — inverted
- **Fix applied by:** Major (AI/ML Engineer)
- **Correction:** Formula changed to `((1 - output[px]) / 2) * 255`: 1 → 0 (black ink), -1 → 255 (white background) — correct
- **Files fixed:** `src/frontend/src/App.tsx`, `src/frontend/src/inference/OnnxInference.ts`
- **Status:** Pushed to feat/togusa-inference-pipeline; PR #4 unblocked, ready for re-review
- **Learning:** Model convention (+1 = foreground, -1 = background per tanh GAN standard) must be documented and enforced at integration boundaries

### 2026-02-25T165444: PR #8 & PR #9 merged to dev — training + backend ready
- **Session:** Final PR review and merge by Saito and Aramaki
- **PR #8 (Major — cGAN Training Pipeline):**
  - **Final status:** ✅ APPROVED & MERGED after Togusa fixed doc conflict
  - **Key fix:** Togusa corrected `.squad/decisions.md` line 184 (Latin chars were wrong: `A,B,H,O,g,n,o,p,s,x` → fixed to `A,B,C,D,E,H,I,O,R,X`)
  - **Also:** Added clarifying comment to `models/train/model.py` explaining 10 uppercase Latin chars
  - **Deliverables:** models/train/{model.py, train.py, dataset.py, export.py, requirements.txt, README.md}
  - **Tensor contract LOCKED:** input [B,10,1,128,128] float32 (style glyphs) + [B] int64 (char index) → output [B,1,128,128] float32 in [-1,1] range
  - **ONNX:** opset 17, INT8 quantization, ready for export post-training
  - **Issue #6 closed:** "Merged in PR #8"
  - **Next phase:** Major trains on Google Fonts, exports model to models/v1/generator.onnx
- **PR #9 (Batou — Backend Integration):**
  - **Final status:** ✅ APPROVED & MERGED by Aramaki
  - **Deliverables:** GET /health, GET /api/model endpoints with 4 xUnit integration tests
  - **Architecture:** Stable /api/model abstraction decouples frontend from versioned file paths
  - **Safety:** Directory.Exists() wraps PhysicalFileProvider, allows tests to run before models/ exists
  - **Range support:** Enables future HTTP Range requests for progressive loading
  - **Tests:** All 4 passing (health 200, model 404 when absent, CORS headers on both endpoints)
  - **Issue #7 closed:** "Merged in PR #9"
  - **Next phase:** Awaits Major's trained ONNX model at models/v1/generator.onnx
- **Decisions inbox merged:** All 5 inbox files consolidated into decisions.md and deleted
- **Session log created:** 20260225T165444-pr8-pr9-merged.md documents handoff to training phase
