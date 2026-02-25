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
| Trained ONNX model | `CyrillicFontGen.Api/models/v1/generator.onnx` |

## API surface

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/font/validate` | Validate an uploaded OTF/TTF/WOFF2 font (multipart/form-data) |
| `GET`  | `/api/model/manifest` | Model version, SHA-256, size, and download URL |
| `GET`  | `/models/v1/generator.onnx` | Immutable-cached model file (HTTP Range supported) |

## Configuration

See `CyrillicFontGen.Api/appsettings.json`.  
Override CORS origins in `appsettings.Development.json` or via environment variable `Cors__AllowedOrigins__0`.
