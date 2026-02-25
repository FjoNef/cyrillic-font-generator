# Decisions

Team decisions, constraints, and accepted patterns. All agents must respect entries here.

<!-- Append new entries below. Scribe merges from inbox. -->

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
