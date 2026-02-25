# Decisions

Team decisions, constraints, and accepted patterns. All agents must respect entries here.

<!-- Append new entries below. Scribe merges from inbox. -->

### 2026-02-25T152812: PR #4 approved — inference pipeline
**By:** Saito (QA)  
**What:** PR #4 feat/togusa-inference-pipeline approved after Major fixed color inversion bug.
  Fix confirmed: ((1 - output[px]) / 2) * 255 correctly maps +1→black, -1→white.
  All acceptance criteria met. Ready to merge → dev.  
**Why:** QA sign-off on blocking issue resolution. Added colorMapping.test.ts to prevent regression.

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
- Style glyphs: render Latin A, B, H, O, g, n, o, p, s, x — 10 chars chosen for maximum structural diversity
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
