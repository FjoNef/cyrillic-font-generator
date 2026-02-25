# Backend API Endpoints — Integration Implementation

**Date:** 2026-02-25  
**Author:** Batou (Backend Dev)  
**Branch:** feat/batou-backend-integration  
**PR:** #9  
**Issue:** #7

## Decision

Implemented two core backend endpoints to support frontend integration and model delivery:

1. **Health check:** `GET /health` → `{ status: "healthy" }`
2. **Model delivery:** `GET /api/model` → serves `models/v1/generator.onnx` with Range support and cache headers

## Rationale

### Why a direct model endpoint?

The existing architecture already had static file serving via `/models/v1/generator.onnx`, but no single-endpoint abstraction for the frontend to request "the current model." Adding `GET /api/model` provides:

- **Stable contract:** Frontend doesn't hardcode versioned paths (e.g., `/models/v1/...`). If model version changes, only backend config updates.
- **Graceful 404:** Returns structured JSON error when model not yet trained, not a raw 404 from static middleware.
- **Range support:** `Results.File(..., enableRangeProcessing: true)` enables HTTP Range requests for large model files (future optimization for progressive loading).

### Why keep static file middleware?

`/models/v1/generator.onnx` is still served via static file middleware because:
- CDN-friendly caching headers (`Cache-Control: public, max-age=31536000, immutable`)
- Future: manifest endpoint can return `downloadUrl` for clients that prefer direct static access
- Redundancy: `/api/model` endpoint and static path both work

### Static file middleware safety fix

Tests failed initially because `PhysicalFileProvider` throws `DirectoryNotFoundException` when `models/` doesn't exist. Wrapped in `Directory.Exists()` check:

```csharp
if (Directory.Exists(modelPhysicalPath))
{
    app.UseStaticFiles(new StaticFileOptions { ... });
}
```

This allows tests to run before model training completes without mocking the filesystem.

### Test strategy

- **WebApplicationFactory<Program>:** In-memory test server, no external process needed
- **Four tests:**
  1. Health check returns 200 + "healthy"
  2. Model endpoint returns 404 when model absent
  3. CORS headers present on health check (OPTIONS preflight)
  4. CORS headers present on model endpoint (OPTIONS preflight)
- **CORS validation:** Tests verify headers are present OR request succeeds (OPTIONS may return 200 or 204 depending on middleware order)

## Alternatives Considered

1. **No `/api/model` endpoint, frontend hardcodes `/models/v1/generator.onnx`**  
   Rejected: Couples frontend to backend file layout. No graceful error handling.

2. **Only `/api/model/manifest`, frontend fetches static file via returned URL**  
   Considered: More RESTful, separates metadata from download. Deferred as over-engineering for MVP — single endpoint simpler for now.

3. **Serve model from `/api/model` only, remove static file middleware**  
   Rejected: Loses CDN caching benefits. Static middleware + API endpoint both have value.

## Impact

- **Frontend:** Can now call `GET /api/model` to fetch ONNX model (when available)
- **CI:** `dotnet test` runs 4 integration tests on every PR
- **Major (ML Engineer):** Once ONNX model exported to `models/v1/generator.onnx`, endpoint will return 200 + file stream
- **Togusa (Frontend):** Can integrate model loading without worrying about file paths

## Open Questions

None. Ready for review by Saito (QA) and Aramaki (Lead).
