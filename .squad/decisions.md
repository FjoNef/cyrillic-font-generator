# Decisions

Team decisions, constraints, and accepted patterns. All agents must respect entries here.

<!-- Append new entries below. Scribe merges from inbox. -->

### 2026-02-25T180000: Retroactive branch created for ML pipeline fixes
**By:** Major (AI/ML Engineer) — requested by FjoNef

**What:** Created `feat/major-training-pipeline-fixes` retroactively to fix branching policy violation. Reset dev back to f07d86a (origin/dev). All ML training pipeline work now lives on the feature branch with PR #10 open for Saito review.

**Why:** Branching policy violation — Major's fixes (3 commits: 8f83be0, da3d162, 102db9b) landed directly on dev instead of a feature branch per .squad/decisions.md branching policy. Corrected via retroactive branching approach:
1. Branch creation from current HEAD captured all work
2. Stage cleanup committed remaining changes (models/logs/ TensorBoard events)
3. Dev reset to origin/dev removed misplaced commits
4. Feature branch pushed to origin with PR opened

**Status:** PR #10 awaiting Saito review before merge to dev.

---

### 2026-02-25: Major Model Architecture — cGAN with Style Encoder
**By:** Major (AI/ML Engineer)  
**Status:** Implemented  
**PR:** #8

Use a **conditional GAN (pix2pix-style)** architecture with StyleEncoder (shared CNN + mean-pooling), UNetGenerator (blank canvas + bottleneck conditioning), and PatchDiscriminator (70×70). StyleEncoder mean-pooling provides permutation invariance; conditioning at bottleneck simplifies ONNX export vs. per-layer AdaIN. Loss: adversarial (BCE) + L1 reconstruction (lambda=100). Alternatives (VAE, diffusion, direct CNN) rejected: VAE too blurry, diffusion too slow for browser, direct CNN lacks adversarial signal. Model size: ~15-20M parameters; ONNX float32 ~40-50MB; INT8 quantized ~15-20MB (browser target). Training scalable to 100-400 fonts; zero frontend changes needed.

### 2026-02-25: Major Model Tensor Contract Confirmation — LOCKED
**By:** Major (AI/ML Engineer)  
**Status:** LOCKED — Do NOT change  
**PR:** #8

Training pipeline (PR #8) **exactly implements** the tensor contract expected by frontend (PR #4). **Inputs:** style_glyphs [B, 10, 1, 128, 128] float32 (10 Latin reference chars A,B,C,D,E,H,I,O,R,X at 128×128 grayscale, normalized [-1,1] where +1=black ink, -1=white), char_index [B] int64 (which of 66 Cyrillic chars: 0–32 uppercase А–Я with Ё at 6, 33–65 lowercase а–я with ё at 39). **Output:** generated_glyph [B, 1, 128, 128] float32 (range [-1,1] where +1=black, -1=white). Frontend implements style glyph extraction, character indexing, output postprocessing ((1-value)/2)*255, and Web Worker protocol. Changing contract breaks frontend. All models MUST conform exactly to these shapes and semantics.

### 2026-02-25: Batou Backend API Endpoints — Integration Implementation
**By:** Batou (Backend Dev)  
**PR:** #9  
**Issue:** #7

Implemented two core endpoints: (1) GET /health → { status: "healthy" }; (2) GET /api/model → serves models/v1/generator.onnx with Range support and cache headers. /api/model provides stable abstraction: frontend doesn't hardcode paths (version changes require only backend config update), graceful 404 returns JSON error when model absent, Range support enables HTTP Range requests (future progressive loading). Static file middleware retained for CDN caching benefits. PhysicalFileProvider wrapped in Directory.Exists() check preventing DirectoryNotFoundException when models/ absent (test suite runs before training). Test strategy: 4 xUnit integration tests via WebApplicationFactory<Program> validate health check (200 + "healthy"), model endpoint 404 (absent), CORS headers on health, CORS headers on model. Alternatives rejected: hardcoding frontend paths (couples frontend to backend layout), manifest endpoint only (over-engineering for MVP). Impact: Frontend fetches model via GET /api/model; CI validates 4 integration tests per PR; Major exports model post-training.

### 2026-02-25: Togusa PR #8 Documentation Fix — Lockout Protocol
**By:** Togusa (Frontend Dev)  
**Branch:** feat/major-model-training (PR #8)

Saito review blocked PR #8: decisions.md line 184 listed Latin style chars incorrectly as "A, B, H, O, g, n, o, p, s, x" (mixed case); actual code uses "A, B, C, D, E, H, I, O, R, X" (uppercase only). Major under Reviewer Rejection Lockout — unable to make changes to PR #8 after Saito requested changes. Togusa fixed on Major's behalf: (1) Corrected decisions.md line 184 to "A, B, C, D, E, H, I, O, R, X"; (2) Added clarifying comment to models/train/model.py lines 202-204. Verified dataset.py and README.md already correct. Committed to feat/major-model-training commit bdf2321. Blocking issue resolved; PR #8 ready for Saito re-review and approval.

### 2026-02-25: Model Tensor Contract Confirmation
**By:** Major (AI/ML Engineer)  
**Status:** LOCKED — Do NOT change  
**PR:** #8

The training pipeline (PR #8) **exactly implements** the tensor contract defined in this file and expected by the frontend (PR #4).

**Contract Specification:**

**Inputs:**
1. `style_glyphs`: [B, 10, 1, 128, 128] float32
   - 10 Latin reference characters (A, B, C, D, E, H, I, O, R, X) rendered at 128×128 grayscale
   - Normalization: [-1, 1] where +1.0 = black ink (foreground), -1.0 = white (background)

2. `char_index`: [B] int64
   - Which of the 66 Cyrillic characters to generate (0-indexed)
   - 0–32: uppercase А–Я (with Ё at index 6)
   - 33–65: lowercase а–я (with ё at index 39)

**Output:**
1. `generated_glyph`: [B, 1, 128, 128] float32
   - Generated Cyrillic glyph at 128×128 grayscale
   - Range: [-1, 1] where +1.0 = black ink (foreground), -1.0 = white (background)
   - Postprocessing: Frontend converts via `((1 - value) / 2) * 255`

**Verification:** Model architecture (FontGeneratorGAN), dataset (FontDataset), and ONNX export (export.py) all conform to this contract. Quantization applied to weights; float32 activations preserved.

**Why Locked:** Frontend (PR #4) already implemented style glyph extraction, character indexing, output postprocessing, and Web Worker protocol. Changing this contract breaks integration. All trained models **MUST** conform exactly to these shapes and semantics.

---

### 2026-02-25: Model Architecture — cGAN with Style Encoder
**By:** Major (AI/ML Engineer)  
**PR:** #8

**Decision:** Use a conditional GAN (pix2pix-style) architecture with:
1. StyleEncoder — Shared-weight CNN + mean-pooling over N reference glyphs (permutation-invariant)
2. UNetGenerator — Blank canvas input; character embedding + style vector concatenated at 4×4 bottleneck
3. PatchDiscriminator — 70×70 receptive field for per-patch realism

**Rationale:**
- StyleEncoder mean-pooling: flexible to variable number of reference glyphs, order-independent
- UNetGenerator: proven U-Net architecture for image-to-image tasks; blank canvas forces style-driven generation
- Conditioning at bottleneck: simpler ONNX export than per-layer AdaIN; concatenation avoids dynamic BatchNorm
- Loss: adversarial (BCE) + L1 reconstruction (lambda=100) prioritizes pixel-wise accuracy

**Alternatives considered and rejected:**
- VAE: blurry outputs unsuitable for sharp font strokes
- Diffusion models: too slow for browser inference (50+ steps)
- Direct CNN: inferior results without adversarial signal

**Model size:** ~15-20M parameters (generator); ONNX float32 ~40-50MB; INT8 quantized ~15-20MB (browser delivery target).

**Impact:** Training scalable to 100-400 fonts; inference fits browser constraints; zero frontend changes needed.

---

### 2026-02-25: Backend API Endpoints — Integration Implementation
**By:** Batou (Backend Dev)  
**PR:** #9  
**Issue:** #7

**Decision:** Implemented two core endpoints:
1. `GET /health` → `{ status: "healthy" }`
2. `GET /api/model` → serves `models/v1/generator.onnx` with Range support and cache headers

**Rationale:**
- `/api/model` provides stable abstraction: frontend doesn't hardcode `/models/v1/...` paths. Version changes require only backend config update.
- Graceful 404: returns structured JSON error when model not yet trained, not raw 404.
- Range support: enables HTTP Range requests for large files (future progressive loading optimization).
- Static file middleware retained: CDN-friendly caching headers, redundancy, future manifest endpoint compatibility.

**Safety improvement:** Wrapped `PhysicalFileProvider` in `Directory.Exists()` check, preventing DirectoryNotFoundException when models/ absent (test suite runs before training).

**Test strategy:** 4 xUnit integration tests via WebApplicationFactory<Program>:
1. Health check returns 200 + "healthy"
2. Model endpoint returns 404 when absent
3. CORS headers on health check
4. CORS headers on model endpoint

**Alternatives considered:**
- No `/api/model`, frontend hardcodes file path: rejected (couples frontend to backend layout)
- Manifest endpoint only: deferred as over-engineering for MVP
- API endpoint only, no static middleware: rejected (loses CDN benefits)

**Impact:** Frontend can fetch model via `GET /api/model`; CI validates 4 integration tests per PR; Major can export trained model post-training.

---

### 2026-02-25T152812: PR #4 approved — inference pipeline
**By:** Saito (QA)  
**What:** PR #4 feat/togusa-inference-pipeline approved after Major fixed color inversion bug.
  Fix confirmed: ((1 - output[px]) / 2) * 255 correctly maps +1→black, -1→white.
  All acceptance criteria met. Ready to merge → dev.  
**Why:** QA sign-off on blocking issue resolution. Added colorMapping.test.ts to prevent regression.

### 2026-02-25T153728: CI must run tests, not just build
**By:** Batou (fix per Aramaki review)  
**What:** squad-ci.yml and squad-preview.yml now include `npx vitest run` after frontend build. Backend uses `dotnet test`.  
**Why:** Aramaki flagged missing test step as blocking in PR #5 review. CI workflows were building successfully but not validating correctness via test execution.

### 2026-02-25 15:39:40: PR #5 CI automation — APPROVED
**By:** Aramaki (Lead)
**What:** PR #5 approved after Batou added vitest test step. CI now runs: frontend build + vitest run + dotnet build + dotnet test.
**Why:** All blocking issues resolved. Ready to merge to dev.

### 2026-02-25T145755: CI/CD Automation Configuration
**By:** Batou (Backend Dev)  
**Branch:** chore/batou-ci-automation  
**PR:** #5 to dev

**What:**
Configure GitHub Actions workflows for CI, release automation, preview validation, and PR auto-labeling.
- **CI Pipeline (squad-ci.yml):** Triggers on PRs/pushes to dev/main/preview/insider. Builds + tests React/Vite frontend and .NET 8 backend in parallel. Node.js v20, npm cache.
- **Release Automation (squad-release.yml):** Triggers on main push. Builds both stacks, extracts version from package.json, creates GitHub release with auto-generated notes.
- **Preview Validation (squad-preview.yml):** Triggers on preview branch push. Full CI suite validation.
- **PR Auto-Label (squad-pr-auto-label.yml):** Triggers on PR open/reopen to dev. Parses team.md, extracts author from branch name, applies `squad` + `squad:{author}` labels, posts review notification with Saito (QA) + Aramaki (Lead) pings.
- **Label Sync:** Created squad:aramaki, squad:batou, squad:togusa, squad:major, squad:saito labels + go:/release:/type:/priority: categories (run 22402264965).

**Why:** 
- Automate testing on every PR and push, reducing manual QA burden.
- Enable zero-touch release automation on main branch.
- Route PRs to correct reviewers automatically via branch name → label → mention.
- Align with team branching policy (dev as integration, main as releases-only).

---
### 2026-02-25T152134: Model output tensor convention
**By:** Major  
**What:** Model output range [-1, 1] where +1.0 = black ink (foreground), -1.0 = white background.  
Frontend pixel mapping: value → grayscale = ((1 - value) / 2) * 255  
**Why:** Confirmed by Saito QA rejection of PR #4; fixing inverted color output. The model uses tanh activation, standard for GANs, which outputs [-1, 1]. The semantic convention that +1 = foreground (ink) and -1 = background (paper) must be respected in all downstream processing.

### 2026-02-25T143900: Inference Pipeline Implementation
**By:** Togusa (Frontend Dev)  
**What:** Implemented end-to-end browser-based inference pipeline: Web Worker wrapping ONNX Runtime Web, scanline-based glyph vectorization (raster-to-path), fixed font metrics (1000 UPM, ascender 800, descender -200, advance width 600), model loader singleton with progress tracking, style glyph extraction on font upload.  
Key decisions:
- Web Worker message protocol with per-request IDs for concurrent safety
- Scanline rectangles for vectorization (not potrace — no maintained browser port; acceptable for MVP, future enhancement possible)
- Progress reporting: model load %, generation counter (N/66)
  
**Why:** Decouples frontend UI from inference via dedicated worker thread. Scanline approach is simple, deterministic, and works in browser. Fixed metrics simplify MVP; future iteration can derive from user's font.

### 2026-02-25T140433: Branching policy overhaul
**By:** FjoNef (via Copilot)
**What:**
- **Main branch is releases-only.** No .squad/ files on main. No direct feature work on main.
- **Dev is the integration branch.** All feature branches created from and merged to dev.
- **.squad/ is excluded from main via .gitignore.** Squad state lives on dev and feature branches.
- Branch naming: `<type>/<agent>-<short-description>` branching from dev.
- Scribe branches: `chore/scribe-*` branching from dev.
- PRs require at least a description of what changed; Saito reviews for quality before merge where feasible.
**Why:** User directive — clean main for releases, dev as integration branch, squad tooling off main.

### 2026-02-25T140433: PR #1 review — branching policy overhaul approved
**By:** Aramaki (Lead)
**What:**
- Approved PR #1 "chore: establish branching policy and conclude session 2026-02-25"
- Branching policy changes align with team architecture and workflow
- Ceremony definition properly structured; session log comprehensive
- PR serves as checkpoint before implementation phase
**Why:** Architectural review confirms alignment with team strategy.

### 2026-02-25T112635: User directives — scope answers
**By:** FjoNef (via Copilot)
**What:**
- Cyrillic scope: Russian only (66 glyphs: А–Я + а–я with Ё/ё) for v1. Extended Cyrillic deferred.
- Quality bar: MVP pipeline first — get end-to-end working, iterate quality later.
- Hosting: Self-hosted (not Azure-managed services).
- Generated font licensing: All generated fonts must be OFL (Open Font License) licensed.
**Why:** User request — establishes project constraints and licensing commitments.

### 2026-02-25: ML engineering decisions
**By:** Major (AI/ML Engineer)
**What:**
- ONNX model inputs: style_glyphs [B, 10, 1, 128, 128] float32, char_index [B] int64
- ONNX model output: generated_glyph [B, 1, 128, 128] float32 (values in [-1, 1])
- Training data: Google Fonts OFL-only, both Latin+Cyrillic coverage
- Style glyphs: render Latin A, B, C, D, E, H, I, O, R, X — 10 uppercase chars chosen for maximum structural diversity
- Model checkpoint format: .pth files in models/checkpoints/epoch_NNNN.pth
- ONNX export: opset 17, dynamic INT8 weight quantization, single graph (StyleEncoder + UNetGenerator)
- Recommended browser backends: WebGL first, WASM fallback; run in Web Worker
- Expected inference: ~15–30ms/glyph WebGL, ~80–150ms/glyph WASM (4-thread)
**Why:** Establishes the contract between ML training (Major) and browser inference (Togusa). Full integration details in src/model/export/inference_contract.md.

### 2026-02-25: Frontend scaffold decisions
**By:** Togusa (Frontend Dev)
**What:**
- React 18 + TypeScript + Vite frontend in src/frontend/
- onnxruntime-web integrated with WASM asset handling
- Style glyphs: Latin A,B,C,D,E,H,I,O,R,X rendered at 128x128 grayscale
- Russian charset: 66 chars (33 upper + 33 lower), indices 0-65
- Model loaded via fetch with progress, stored in InferenceSession
- Vite dev server proxies /api to backend :5000
**Why:** Establishes frontend project structure and integration contracts with ML and backend.

### 2026-02-25: Backend scaffold decisions
**By:** Batou (Backend Dev)
**What:**
- .NET 8 ASP.NET Core Minimal API in src/backend/
- Self-hosted on port 5000 (HTTP), no Azure dependencies
- Model served from models/v1/ with immutable cache headers + Range request support
- Font validation via magic byte detection (no heavy font parsing library)
- CORS allows localhost:5173 (Vite dev server); origins are config-driven
- SPA fallback: all unmatched routes → wwwroot/index.html
- ModelManifestCache singleton computes SHA-256 at startup; returns 404 if model not yet present
**Why:** Minimal, self-hostable backend focused on model delivery and font validation. No server-side inference required — all AI runs in the browser via ONNX Runtime Web.

### 2026-02-25: Architecture kickoff decisions
**By:** Aramaki (Lead)

**Decisions:**

1. **AI Model: Conditional GAN (pix2pix-style) with style encoder**
   **Why:** Proven for image-to-image style transfer, compact enough for browser, well-documented ONNX export path. Font glyph generation is fundamentally a style transfer problem — map Latin skeleton shapes to Cyrillic glyphs in the same visual style.

2. **Training framework: PyTorch**
   **Why:** Best ecosystem for research/prototyping GANs, first-class ONNX export support, Major's most productive environment.

3. **Browser inference runtime: ONNX Runtime Web (WebAssembly + WebGL backends)**
   **Why:** Smaller runtime than TensorFlow.js, supports WebGL acceleration, single model format from PyTorch → ONNX → browser. No framework translation needed.

4. **Model size target: < 20MB (compressed)**
   **Why:** Balance between quality and load time. Achievable with a lightweight U-Net generator. Can be lazy-loaded after initial page render.

5. **Frontend: React + TypeScript (Vite)**
   **Why:** Richer ecosystem for canvas/SVG manipulation, better ONNX Runtime Web integration, faster iteration than Blazor WASM. Blazor WASM adds unnecessary runtime overhead alongside the ML model.

6. **Backend: ASP.NET Core Minimal API**
   **Why:** Lightweight, serves static model files + SPA, handles font file validation, no server-side inference needed. MVC is overkill.

7. **Font output: SVG paths → OpenType via opentype.js**
   **Why:** Model generates glyph images, we vectorize with potrace/similar, then assemble into an OTF using opentype.js in-browser. User downloads a real font file.

8. **Input method: User uploads a font file (OTF/TTF/WOFF2), we extract Latin glyphs as style reference**
   **Why:** Most natural UX — user has a font they love, wants Cyrillic added. We render reference glyphs to canvas, feed to model as style conditioning.

9. **Training data: Google Fonts corpus (paired Latin–Cyrillic fonts) + augmentation**
   **Why:** ~400 Google Fonts have both Latin and Cyrillic coverage. Natural paired training data. Augment with weight/slant/size variations.

10. **Repository layout: monorepo with /src/frontend, /src/backend, /src/model, /data**
    **Why:** Single repo keeps all team members coordinated. Clear separation of concerns.

**Open questions for FjoNef:**
- Target Cyrillic script: Russian only (33 letters) or full Extended Cyrillic (Serbian, Ukrainian, Bulgarian, etc.)?
- Quality bar: is "good enough for prototyping" acceptable initially, or must the first release be production-quality?
- Hosting plan: Azure App Service, static site + API, or self-hosted?
- License preference for the AI model and training data?

---

### 2026-02-25T180000: Saito QA verdict — PR #10 (APPROVED)
**By:** Saito (QA)
**Verdict:** APPROVED  
**PR:** #10 (feat/major-training-pipeline-fixes → dev)

**What passed:**
- model.py: UNet dec5/dec6/dec7 skip-connection channel counts corrected (asymmetric sums matching encoder output dims), duplicate self.final removed, forward pass fixed — output [B,1,128,128] float32 [-1,1] ✅
- dataset.py: DEFAULT_STYLE_CHARS = ["A","B","C","D","E","H","I","O","R","X"] uppercase-only 10 chars matches contract; fontTools casing + Windows TTFont.close() fix; SyntheticFontDataset produces correct tensor shapes and dtypes ✅
- train_config.yaml: style_latin_chars = ["A","B","C","D","E","H","I","O","R","X"], paths relative to repo root ✅
- download_fonts.py: fontTools import casing + TTFont.close() Windows file locking fix ✅
- train.py: --synthetic, --batch_size, --num_epochs CLI flags correct; 1-epoch real-data validation passed (47,388 samples) ✅

**Minor non-blocking:** Synthetic mode default config in train.py still has `../../models/checkpoints/` (old relative path) instead of `models/checkpoints/`. Does not affect real training. Can be addressed in follow-up.

**Action:** PR squash-merged to dev; feature branch deleted.

---

### 2026-02-26T122537: QA protocol update — stale-branch verification
**By:** Saito (QA)  
**Decision:** Stale-branch PRs must be verified against merge-base before review

**What:** PR #11 proposed fixing `../../models/` output paths in `train_config.yaml`. Investigation revealed the fix was already in dev via PR #10's squash-merge (commit 3e54f39). The feature branch diverged from `f07d86a` (pre-PR-#10), making it a stale duplicate.

**Risk identified:** Stale branch also carried regressions — `fonts_dir` and `style_latin_chars` had wrong values that would have overwritten correct post-PR-#10 values. Merging would have broken the tensor contract.

**QA protocol:**
1. Before reviewing fix PRs, run `git merge-base <feature> <dev>` and inspect branch divergence point
2. If divergence pre-dates a recent squash-merge, check whether the squash-merge already includes the proposed fix
3. PRs in dirty merge state must be inspected for regressions, not just conflicts

**Action:** Posted REQUEST CHANGES on PR #11 recommending closure as duplicate. No changes to dev required.

---

### 2026-02-26T122537: User directive — branching policy reinforcement
**By:** FjoNef (via Copilot)  
**What:** Always create a separate feature branch before doing any work. Never commit directly to dev. Applies to ALL agents on ALL tasks — no exceptions.  
**Branch naming:** `<type>/<agent>-<short-description>` branching from dev.  
**Why:** User re-issued branching reminder after checkpoint-path fix was initially committed directly to dev (later corrected). Reinforces existing branching policy from 2026-02-25T140433.
