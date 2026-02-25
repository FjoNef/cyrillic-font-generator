# Cyrillic Font Generator

A web application that generates Cyrillic glyphs for non-Cyrillic fonts using a pre-trained AI model. Upload any Latin font, get a matching Cyrillic extension — all inference runs in your browser.

## Architecture

```
src/
  frontend/   React + TypeScript (Vite) — UI, ONNX Runtime Web inference
  backend/    ASP.NET Core Minimal API (.NET 8) — font validation, model delivery, SPA hosting
  model/      PyTorch training code — pix2pix-style conditional GAN
data/
  fonts/      Google Fonts training corpus (Latin+Cyrillic pairs)
models/       Trained ONNX model artefacts
```

## Quick start

### 1. Train the model (Major's domain)

```bash
cd src/model
pip install -r requirements.txt
python train.py
# Outputs: models/v1/generator.onnx
```

### 2. Start the backend

```bash
cd src/backend
dotnet run --project CyrillicFontGen.Api
# Serves at http://localhost:5000
```

### 3. Start the frontend dev server

```bash
cd src/frontend
npm install
npm run dev
# Vite dev server at http://localhost:5173
# Proxies API calls to http://localhost:5000
```

Open **http://localhost:5173** in your browser.

## Key decisions

- Model size target: **< 20 MB compressed** (lazy-loaded after page render)
- Inference: **ONNX Runtime Web** (WebAssembly + WebGL backends)
- Font output: SVG paths assembled into OTF via **opentype.js** in-browser
- Input: user uploads an OTF/TTF/WOFF2; Latin glyphs are extracted as style reference

## License

See [LICENSE](LICENSE).
