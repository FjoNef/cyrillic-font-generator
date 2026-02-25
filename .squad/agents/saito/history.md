# Saito — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **My focus:** Tests, quality, cross-browser validation, inference performance.

## Learnings
<!-- Append new entries below -->

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
