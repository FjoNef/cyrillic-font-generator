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

### 2026-02-25: CI/CD workflow automation
- **Branch:** `chore/batou-ci-automation` from dev
- **PR:** #5 to dev
- **Changes:**
  - **squad-ci.yml:** Configured Node 20 + npm ci/build/tsc + .NET 8 restore/build/test steps. Runs on PRs to dev/main and pushes to dev/insider.
  - **squad-release.yml:** Build both stacks + create GitHub Release with auto-generated notes from package.json version. Runs on push to main.
  - **squad-preview.yml:** Build validation (same as CI) on preview branch pushes.
  - **squad-pr-auto-label.yml (new):** Auto-labels PRs to dev with `squad` + `squad:{author}` based on branch name prefix (e.g., `chore/batou-ci-automation` → `squad:batou`). Posts review notification comment pinging Saito (QA) and Aramaki (Lead).
- **Tech stack:** Node 20, .NET 8.0.x, npm ci (not install), separate jobs for frontend/backend validation.
- **Why:** Squad triage/heartbeat/main-guard workflows were live but CI was a no-op placeholder. This activates build gates on PRs to dev/main and release automation on main.

### 2026-02-25: CI test fix (Aramaki review finding)
- **Commit:** c6390f5
- **Issue:** Aramaki review of PR #5 found blocking issue: workflows were building but not running tests.
- **Fix:** Added `npx vitest run` step to both squad-ci.yml and squad-preview.yml after frontend build/type-check. Backend already had `dotnet test` step.
- **Learnings:**
  - **Test command:** `npx vitest run` (not `npm test`) for Vitest in CI — ensures non-watch mode.
  - **CI must run tests, not just build.** Build success ≠ code correctness.
  - **Workflows fixed:** squad-ci.yml, squad-preview.yml (both now test frontend + backend).
- **Approval:** PR #5 approved by Aramaki after test step fix.

### 2026-02-25: Backend integration — health check, model delivery endpoint, integration tests
- **Branch:** feat/batou-backend-integration
- **PR:** #9 to dev
- **Issue:** #7 — feat: backend integration — end-to-end API test with frontend
- **Changes:**
  - **Health check endpoint:** `GET /health` → 200 OK with `{ status: "healthy" }`
  - **Model delivery endpoint:** `GET /api/model` → serves `models/v1/generator.onnx` or 404 with clear error message
  - **CORS:** Already configured for localhost:5173, verified in tests
  - **Integration tests (xUnit):** 4 tests covering health check (200), model delivery (404 when model absent), and CORS headers on both endpoints
  - **Static file middleware fix:** Wrapped PhysicalFileProvider in `Directory.Exists()` check to prevent tests failing when models directory doesn't exist
  - **Program class exposure:** Added `public partial class Program { }` to enable WebApplicationFactory<Program> in tests
- **Testing:** `cd src/backend && dotnet test` — all 4 tests pass
- **Why:** Establishes minimum viable backend API surface for frontend integration. Model endpoint returns 404 until Major trains and exports the ONNX model to `models/v1/generator.onnx`.
