# SKILL: ASP.NET Core Large Binary Model Delivery

## Summary
Pattern for serving a large ML model file (ONNX, TFLite, etc.) from an ASP.NET Core Minimal API with full HTTP caching semantics: ETag, Cache-Control immutable, Range requests, and Brotli compression.

## When to use
- Backend must serve a large binary asset (50–200 MB) to a browser client
- File changes infrequently; version is embedded in URL for cache-busting
- Client needs to detect stale downloads via ETag/If-None-Match (CDN-friendly)
- Progressive/chunked loading required (Range requests)

---

## Pattern

### 1. Precompute metadata at startup (singleton)

```csharp
public sealed class ModelManifestCache
{
    public string Version   { get; } = "v1";
    public string Filename  { get; } = "generator.onnx";
    public long   SizeBytes { get; private set; }
    public string Sha256    { get; private set; } = string.Empty;
    public bool   Available { get; private set; }

    public ModelManifestCache(IConfiguration config, IWebHostEnvironment env, ILogger<ModelManifestCache> logger)
    {
        var file = Path.Combine(env.ContentRootPath, config["ModelPath"] ?? "models", Version, Filename);
        if (!File.Exists(file)) { logger.LogWarning("Model absent: {Path}", file); return; }

        SizeBytes = new FileInfo(file).Length;
        using var fs = File.OpenRead(file);
        Sha256 = Convert.ToHexString(SHA256.HashData(fs)).ToLowerInvariant();
        Available = true;
    }
}
```

Register as singleton: `builder.Services.AddSingleton<ModelManifestCache>();`

### 2. Versioned download endpoint with ETag + Cache-Control

```csharp
app.MapGet("/api/model/{version}/{filename}", (
    string version, string filename,
    ModelManifestCache cache, IConfiguration config, IWebHostEnvironment env,
    HttpContext httpContext) =>
{
    if (!cache.Available) return Results.NotFound();
    if (!version.Equals(cache.Version, StringComparison.OrdinalIgnoreCase) ||
        !filename.Equals(cache.Filename, StringComparison.OrdinalIgnoreCase))
        return Results.NotFound();

    var etag = $"\"{cache.Sha256}\"";
    if (httpContext.Request.Headers.IfNoneMatch == etag)
        return Results.StatusCode(StatusCodes.Status304NotModified);

    var file = Path.GetFullPath(Path.Combine(env.ContentRootPath, config["ModelPath"] ?? "models", version, filename));
    if (!File.Exists(file)) return Results.NotFound();

    httpContext.Response.Headers.ETag = etag;
    httpContext.Response.Headers.CacheControl = "public, max-age=31536000, immutable";
    return Results.File(file, "application/octet-stream", filename, enableRangeProcessing: true);
});
```

### 3. Manifest endpoint (frontend version check)

```csharp
app.MapGet("/api/model/manifest", (ModelManifestCache cache, HttpRequest req) =>
    cache.Available
        ? Results.Ok(new {
              version     = cache.Version,
              filename    = cache.Filename,
              sizeBytes   = cache.SizeBytes,
              sha256      = cache.Sha256,
              downloadUrl = $"{req.Scheme}://{req.Host}/api/model/{cache.Version}/{cache.Filename}"
          })
        : Results.NotFound(new { error = "Model not yet available" }));
```

### 4. Brotli compression in Program.cs

```csharp
builder.Services.AddResponseCompression(o =>
{
    o.EnableForHttps = true;
    o.MimeTypes = ResponseCompressionDefaults.MimeTypes.Concat(["application/octet-stream"]);
});
builder.Services.Configure<BrotliCompressionProviderOptions>(o => o.Level = CompressionLevel.Optimal);
// ...
app.UseResponseCompression(); // Must be first in pipeline
```

---

## Key decisions
- **SHA-256 as ETag**: Stable across restarts, CDN-compatible, matches manifest.
- **Version in URL**: `/api/model/v1/generator.onnx` → `immutable` cache is safe because URL changes on model update.
- **Singleton precompute**: No per-request I/O; SHA-256 computed once on startup.
- **Range requests**: `enableRangeProcessing: true` in `Results.File` enables HTTP 206 Partial Content.
- **304 before file I/O**: Check `If-None-Match` before opening the file.

## Tested in
`cyrillic-font-generator` — `src/backend/CyrillicFontGen.Api/Endpoints/ModelEndpoints.cs`
