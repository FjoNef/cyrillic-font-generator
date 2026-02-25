# Aramaki — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **Key constraint:** Client-side inference only — no server-side generation calls.

## Learnings
<!-- Append new entries below -->

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
