# Cyrillic Font Generator — Backend

ASP.NET Core Minimal API (.NET 8) that serves the React SPA, validates font uploads, and delivers the pre-trained ONNX model.

## Quick start

```bash
cd src/backend
dotnet run --project CyrillicFontGen.Api
# API available at http://localhost:5000
```

## Place build artifacts

| Artifact | Location |
|---|---|
| React build output | `CyrillicFontGen.Api/wwwroot/` |
| Trained ONNX model | `models/v1/generator.onnx` (repo root, **not** inside the API project) |

> **Development:** `appsettings.Development.json` overrides `ModelPath` to `"../../../models"`, resolving
> `src/backend/CyrillicFontGen.Api/` → 3 levels up → repo root `models/`. No extra setup needed.
>
> **Production (published):** When publishing with `dotnet publish`, place `models/v1/generator.onnx`
> **alongside the published binary** (i.e., in the same directory as `CyrillicFontGen.Api.dll`).
> The default `ModelPath: "models"` in `appsettings.json` will then resolve correctly from `ContentRootPath`.

## API surface

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/font/validate` | Validate an uploaded OTF/TTF/WOFF2 font (multipart/form-data) |
| `GET`  | `/api/model/manifest` | Model version, SHA-256, size, and download URL |
| `GET`  | `/models/v1/generator.onnx` | Immutable-cached model file (HTTP Range supported) |

## Configuration

See `CyrillicFontGen.Api/appsettings.json`.  
Override CORS origins in `appsettings.Development.json` or via environment variable `Cors__AllowedOrigins__0`.
