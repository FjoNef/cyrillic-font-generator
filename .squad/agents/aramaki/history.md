# Aramaki — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **Key constraint:** Client-side inference only — no server-side generation calls.

## Learnings
<!-- Append new entries below -->

### 2026-02-25T160138: Model Training + Backend Integration Delivered

**Session Overview:**
Two critical foundation layers completed in parallel:

**PR #8 — cGAN Training Pipeline (Major):**
- Tensor contract LOCKED: input [B,10,1,128,128] float32 (style glyphs) + [B] int64 (char index) → output [B,1,128,128] float32 (glyph)
- Architecture: StyleEncoder (mean-pooling, permutation-invariant) + UNetGenerator (blank canvas, bottleneck conditioning) + PatchDiscriminator
- Loss: Adversarial (BCE) + L1 (lambda=100)
- ONNX: opset 17, INT8 weight quantization, dynamic batch, float32 activations
- Character mapping: 66 Russian Cyrillic (0-32 uppercase А–Я with Ё, 33-65 lowercase а–я with ё)
- **Status:** Ready for Saito QA review; architecture aligns with contract (no coupling to frontend)
- **Next:** Major trains on Google Fonts, exports to models/v1/generator.onnx

**PR #9 — Backend Integration (Batou):**
- Endpoints: GET /health (200 + "healthy"), GET /api/model (200 + ONNX file stream, or 404 if absent)
- Contract: Stable /api/model abstraction decouples frontend from versioned paths (/models/v1/...)
- Range request support: enables future progressive loading
- Safety: Directory.Exists() check allows tests to run before models/ created
- Tests: 4 xUnit integration tests, all passing (health 200, model 404 absent, CORS headers)
- **Status:** Ready for Saito QA review; awaits Major's model export
- **Next:** Major exports trained model, endpoint serves it

**Architectural Notes:**
- Tensor contract locked (no frontend changes needed); Major exports enforces it
- Backend abstraction future-proofs model versioning
- Both PRs respect branching policy (dev integration branch)
- Cross-agent dependency clear: Major → Batou → Togusa (inference pipeline, PR #4)

**QA Responsibilities:**
- PR #8: ONNX validation (input/output names, shapes, opset), dataset normalization, training metrics
- PR #9: Integration test validation, CORS behavior, endpoint graceful 404 handling

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
