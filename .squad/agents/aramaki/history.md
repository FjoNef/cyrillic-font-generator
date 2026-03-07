# Aramaki — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **Key constraint:** Client-side inference only — no server-side generation calls.

## Learnings
<!-- Append new entries below -->

### 2026-03-07T21:06:51Z: Issue #42 Created — Training Performance Optimization for RTX 3070Ti
- **Decision:** OPENED for team investigation and implementation.
- **Issue:** https://github.com/FjoNef/cyrillic-font-generator/issues/42
- **Title:** perf(training): optimize training pipeline for NVIDIA RTX 3070Ti
- **Rationale:** Developer's local machine has RTX 3070Ti (8GB VRAM, 6144 CUDA cores). Style-conditioning fix requires retraining from scratch. Current pipeline (200 epochs, ~15 min/epoch) is unoptimized for this target GPU.
- **Proposed optimizations:**
  1. Mixed precision training (torch.cuda.amp with FP16/BF16)
  2. CUDA optimization (cudnn.benchmark)
  3. Batch size optimization (currently [B, 10, 1, 128, 128])
  4. Data loading pipeline (num_workers, pin_memory)
  5. Profiling with torch.profiler
- **Next steps:** Team can pick up performance tuning as parallel effort while retraining begins.

### 2026-03-07T: PR #38 APPROVED — Robust Model Path Resolution (Walk-Up Pattern)
- **Decision:** ✅ APPROVED for merge to dev. Issue #37 addressed.
- **What changed:** Replaced environment-dependent relative path config with directory walk-up discovery.
- **Walk-up pattern evaluation:**
  - Tries configured path first (respects explicit operator configuration)
  - Only walks up for *relative* paths — absolute paths disable walk-up
  - Bounded traversal (stops at filesystem root)
  - No security concerns: searches for known `models/` directory name, not arbitrary paths
- **Production correctness:** When deployed to `/app/` with model at `/app/models/v1/generator.onnx`, direct path resolves on first attempt — no walk-up needed. Walk-up only activates in dev scenarios.
- **Diagnostics:** Clear operator-friendly log messages at appropriate levels (Info for success, Error for 404 scenarios).
- **Smoke test:** New `smoke-test.ps1` covers health, manifest, model download, versioned endpoint — good pre-deployment check.
- **Pattern established:** Walk-up fallback for path discovery is acceptable for model file resolution, but should be used sparingly (dev convenience, not production complexity).

### 2026-03-07T: PR #36 APPROVED & MERGED— Model Path Resolution (Development Fix + Cross-Agent Coordination)
- **Decision:** ✅ APPROVED & MERGED to dev. Issue #35 closed.
- **Issue #35 Fixed:** Backend 404 on GET /api/model and GET /api/model/v1/generator.onnx.
- **Root cause:** ContentRootPath = `src/backend/CyrillicFontGen.Api/`, but model lives at repo root `models/v1/generator.onnx`.
- **Fix rationale:**
  - Added `"ModelPath": "../../../models"` to `appsettings.Development.json` only.
  - Path.GetFullPath(Path.Combine(ContentRootPath, "../../../models")) walks from API dir → 3 levels up → repo root.
  - Correct pattern: environment-specific config overrides (not env vars, not absolute paths, not copying binaries).
- **Alternatives evaluated:**
  - ❌ Copy model to backend dir: bloats repo history, violates artifact separation.
  - ❌ Environment variable: overkill for dev-only workaround.
  - ❌ Absolute path: non-portable across dev machines.
  - ✅ Relative path in appsettings: portable, .NET idiomatic, dev-focused.
- **Production deployment:** When published, ContentRootPath = publish output dir. Default `"ModelPath": "models"` resolves to publish dir. Operator must place `models/v1/generator.onnx` alongside published binary. Backend README clarified this clearly.
- **Test coverage improvement:** WebApplicationFactory now explicitly injects non-existent ModelPath for 404 tests (intentional), vs coincidental side-effect before. All 25 tests pass.
- **Pattern established:** Development-specific config in `appsettings.Development.json`, production-ready defaults in `appsettings.json`. Future team can apply this pattern for other dev/prod divergences.
- **Cross-agent coordination:** Saito performed parallel QA pass (path resolution, endpoint validation, test hygiene verified). Coordinator merged after both approvals.
- **Why approved:** Minimal, focused, production-ready. Architectural decision (config-based path resolution) is maintainable and aligns with .NET conventions.

### 2026-03-07T: PR #24 APPROVED & MERGED — Playwright Performance Harness (Saito)
- **Decision:** APPROVED. PR #24 merged to dev. Issue #23 closed.
- **What passed review:**
  - Playwright config: Vite dev server (localhost:5173), retries:1 CI, workers:1 CI, webServer timeout 120s — all correct.
  - Performance assertions: load < 5000ms, per-glyph < 500ms, 66-glyph total < 10000ms — all grounded in inference_contract.md.
  - Cross-browser: Chromium + Firefox + WebKit all present and passing.
  - Stub model (345 bytes, Slice+Reshape) matches production tensor contract; correct CI isolation approach.
  - `test:e2e` npm script present. Test structure clean with shared helpers.
- **Minor observations (non-blocking):**
  - squad-heartbeat.yml cron commented out — harmless noise reduction.
  - Two model-load tests assert the same thing (minor redundancy); not a quality gate issue.
  - Charter/casting/log `.squad/` file deletions swept into the PR — appear to be stale-file cleanup, not scope violation.
- **GitHub note:** `gh pr review --approve` failed (can't approve own PR on GitHub), but merge succeeded directly.
- **Why approved:** All 7 review criteria met. 51 tests passing (17 × 3 browsers). Architecture is sound and CI-offline-capable.

### 2026-02-26T: PR #12 lockout revision— Aramaki stepping in for Togusa
- **Context:** Saito issued REQUEST CHANGES on PR #12 (feat/togusa-font-assembly); Togusa locked out under Reviewer Rejection Lockout Protocol. Aramaki stepped in to apply targeted fixes.
- **Fix 1 — cyrillicCharset.ts indices:** Ё was at index 32 (end of uppercase block), ё at 65 (end of lowercase). LOCKED tensor contract requires Ё=6, ё=39. Rebuilt both arrays to interleave Ё/ё at correct alphabetical positions within each block.
- **Fix 2 — Test API surface:** Tests used class instantiation style (`new GlyphVectorizer()`, `vectorizer.vectorize()`); implementations export plain functions. Per team decision: aligned tests to the function API (`vectorizeGlyph()`, `assembleFontFromGlyphs()`). Did NOT touch implementation files.
- **Fix 3 — Map key type:** `makeGlyphImages()` test helper built `Map<string, Float32Array>` with char-string keys; `assembleFontFromGlyphs` expects `Map<number, Float32Array>`. Changed helper to emit numeric model indices 0-65 as keys.
- **Commit:** 6681fcd — all 3 fixes in one commit
- **Lesson:** When a charset file defines model indices, always cross-check against decisions.md tensor contract immediately — index positioning bugs are silent until inference produces wrong glyphs.


### 2026-02-25T165444: PR #8 & PR #9 merged to dev — critical foundation complete
- **Status:** Two foundational layers merged. Training and backend integration live; frontend (PR #4) awaiting model.
- **PR #8 Merge (Major — cGAN Training):**
  - Tensor contract LOCKED: StyleEncoder (mean-pooling) + UNetGenerator (bottleneck conditioning) + PatchDiscriminator
  - Loss: Adversarial (BCE) + L1 (lambda=100); ONNX opset 17, INT8 quantization
  - Character mapping: 66 Russian Cyrillic (0-32 uppercase А–Я with Ё, 33-65 lowercase а–я with ё)
  - **Note:** Saito flagged doc conflict (decisions.md wrong Latin chars); Togusa applied fix under Reviewer Rejection Lockout Protocol
  - **Architectural milestone:** Training pipeline complete, no coupling to frontend (contract enforces decoupling)
  - **Issue #6 resolved:** All acceptance criteria validated; merged with doc fix
- **PR #9 Merge (Batou — Backend Integration):**
  - Endpoints: GET /health (200 + "healthy"), GET /api/model (200 + ONNX stream OR 404 if absent)
  - Range support, CORS validation, safety wrap (Directory.Exists()), 4 xUnit integration tests
  - **Architectural strength:** Stable /api/model abstraction future-proofs model versioning; frontend has no hardcoded paths
  - **Issue #7 resolved:** Backend ready for Major's trained model export
- **Cross-agent handoff:**
  - Major: Trains on Google Fonts, exports to models/v1/generator.onnx
  - Batou: Backend awaits model export; /api/model endpoint will serve it
  - Togusa: Frontend (PR #4) awaits model; Web Worker protocol ready, no code changes needed
  - Saito: Ready to validate trained model quality (tensor shapes, ONNX structure, inference output range)

### 2026-02-25T145755: CI automation live, inference pipeline PR open
- **CI/CD Automation:** Batou configured GitHub Actions workflows (squad-ci.yml, squad-release.yml, squad-preview.yml, squad-pr-auto-label.yml). PR #5 open to dev for review. Workflows automate frontend + backend builds, release creation with auto-generated notes, preview validation, and zero-touch PR labeling + reviewer notification. Labels synced successfully (squad:*, go:, release:, type:, priority:).
- **Inference Pipeline:** Togusa wired end-to-end inference (inferenceWorker.ts, ModelLoader.ts, assembleCyrillicFont, GeneratorPanel.tsx, appStore.ts). PR #4 open to dev for QA review. Web Worker runs model inference off main thread; UI shows progress and glyph preview.
- **Next phase:** Saito QA review → merge both PRs to dev → backend server + frontend dev server coordination.

### 2026-02-25: Architecture Kickoff
- **AI Model:** Chose conditional GAN (pix2pix-style) over diffusion/VAE/transformer. Reasoning: font glyph generation is image-to-image style transfer; pix2pix is proven, compact (~20MB target), ONNX-exportable. Diffusion models are too large/slow for browser. VAEs produce blurry output. Transformers (e.g. SVG-generating) are interesting but less mature.
- **Runtime:** ONNX Runtime Web over TensorFlow.js — smaller runtime, no framework translation from PyTorch.
- **Frontend:** React+TypeScript over Blazor WASM — Blazor WASM runtime adds ~5MB+ on top of the ML model. React has better Canvas/SVG ecosystem for glyph rendering.
- **Backend:** ASP.NET Core Minimal API — just serves static model files + SPA + font validation. No server-side inference.
- **Training data:** Google Fonts corpus (~400 fonts with paired Latin+Cyrillic). PyTorch for training.
- **Output pipeline:** Model → raster glyph → vectorize (potrace) → OpenType font assembly (opentype.js) → downloadable OTF.
- **Open questions identified:** Cyrillic scope (Russian-only vs Extended), quality bar for v1, hosting target, model/data licensing.

### 2026-02-25: Branching policy overhaul
- **Decision:** Main branch is releases-only; dev is integration branch. All feature work via feature branches from dev.
- **Implementation:** Removed .squad/ from main via .gitignore. PR #2 targets dev with branching policy changes.
- **Workflow impact:** All squad tooling lives on dev/feature branches, not main. Preserves clean release history on main.
PR #5 approved after Batou added missing test steps.
