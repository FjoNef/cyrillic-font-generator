# Batou — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **My focus:** .NET backend, APIs, model/asset delivery.

## Learnings
<!-- Append new entries below -->

### 2026-02-25: Backend scaffold

**Project structure:**
```
src/backend/
  CyrillicFontGen.sln
  CyrillicFontGen.Api/
    CyrillicFontGen.Api.csproj   (.NET 8 Web SDK, no heavy deps)
    Program.cs                   (Minimal API entry point)
    appsettings.json
    appsettings.Development.json
    .gitignore
    Endpoints/
      FontEndpoints.cs           (POST /api/font/validate)
      ModelEndpoints.cs          (GET /api/model/manifest + ModelManifestCache singleton)
```

**API contracts:**
- `POST /api/font/validate` — multipart/form-data, font file → `{ valid, fontName, hasLatin, glyphCount, format, error? }`
  - Magic byte detection: OTF=`OTTO`, TTF=`\0\1\0\0` or `true`, WOFF2=`wOF2`
  - Full glyph introspection deferred to client (opentype.js)
- `GET /api/model/manifest` → `{ version, filename, sizeBytes, sha256, downloadUrl }` or 404 if model absent
- `GET /models/v1/generator.onnx` — static file, `Cache-Control: public, max-age=31536000, immutable`

**Model delivery:**
- `ModelManifestCache` singleton reads + SHA-256 hashes the model file at startup; stored in DI, never re-read per-request.
- Model served via `PhysicalFileProvider` mapped to `/models` request path.
- `UseStaticFiles()` provides HTTP Range support (built-in to ASP.NET Core static file middleware).

**CORS:** `localhost:5173` allowed; origins are config-driven (`Cors:AllowedOrigins`).

**SPA fallback:** `MapFallbackToFile("index.html")` — all unmatched routes return `wwwroot/index.html`.

### 2026-02-25: Branching policy overhaul
- **Decision:** Main branch is releases-only; dev is integration branch. All feature work via feature branches from dev.
- **Implementation:** Removed .squad/ from main via .gitignore. PR #2 targets dev with branching policy changes.
- **Backend impact:** All backend development occurs on feature branches from dev, never directly on main. Clear separation between release and development code.
