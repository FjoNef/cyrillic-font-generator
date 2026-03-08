using System.Security.Cryptography;

namespace CyrillicFontGen.Api.Endpoints;

/// <summary>Computed once at startup; reused for every manifest request.</summary>
public sealed class ModelManifestCache
{
    public string Version { get; } = "v1";
    public string Filename { get; } = "generator.onnx";
    public long SizeBytes { get; private set; }
    public string Sha256 { get; private set; } = string.Empty;
    public bool Available { get; private set; }
    public string? ResolvedModelPath { get; private set; }

    public ModelManifestCache(IConfiguration config, IWebHostEnvironment env, ILogger<ModelManifestCache> logger)
    {
        var modelRoot = config["ModelPath"] ?? "models";
        var modelFile = ResolveModelFile(env.ContentRootPath, modelRoot, Version, Filename, logger);

        if (modelFile == null)
        {
            var expectedPath = Path.GetFullPath(Path.Combine(env.ContentRootPath, modelRoot, Version, Filename));
            logger.LogError("✗ Model file not found — /api/model will return 404. Place generator.onnx at: {ExpectedPath}", expectedPath);
            return;
        }

        ResolvedModelPath = modelFile;

        try
        {
            var info = new FileInfo(modelFile);
            SizeBytes = info.Length;

            using var fs = File.OpenRead(modelFile);
            var hash = SHA256.HashData(fs);
            Sha256 = Convert.ToHexString(hash).ToLowerInvariant();
            Available = true;

            logger.LogInformation("✓ Model serving ready: {File} ({Size} bytes)", modelFile, SizeBytes);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Failed to read model file at {Path}", modelFile);
        }
    }

    private static string? ResolveModelFile(string contentRootPath, string modelPathConfig, string version, string filename, ILogger logger)
    {
        // 1. Try the configured path first (respects explicit config)
        var configured = Path.GetFullPath(Path.Combine(contentRootPath, modelPathConfig, version, filename));
        if (File.Exists(configured))
        {
            logger.LogInformation("Model found at configured path: {Path}", configured);
            return configured;
        }

        // If the configured path is absolute and doesn't exist, don't walk up
        // (respect explicit absolute path configuration)
        if (Path.IsPathRooted(modelPathConfig))
        {
            logger.LogWarning(
                "Model file not found at absolute configured path: {Configured}",
                configured);
            return null;
        }

        // 2. Walk up the directory tree looking for models/{version}/{filename}
        var dir = new DirectoryInfo(contentRootPath);
        while (dir != null)
        {
            var candidate = Path.Combine(dir.FullName, "models", version, filename);
            if (File.Exists(candidate))
            {
                logger.LogInformation("Model found by directory walk at: {Path}", candidate);
                return candidate;
            }
            dir = dir.Parent;
        }

        logger.LogWarning(
            "Model file not found. Tried configured path: {Configured}. Also searched parent directories for models/{Version}/{Filename}",
            configured, version, filename);
        return null;
    }
}

public static class ModelEndpoints
{
    public static void MapModelEndpoints(this WebApplication app)
    {
        app.MapGet("/api/model/manifest", HandleManifest);
        app.MapGet("/api/model", HandleModelDownload);
        // Versioned URL — cache-busting safe; version is immutable by URL
        app.MapGet("/api/model/{version}/{filename}", HandleVersionedModelDownload);
    }

    private static IResult HandleManifest(ModelManifestCache cache, HttpRequest request)
    {
        if (!cache.Available)
            return Results.NotFound(new { error = "Model not yet available" });

        var baseUrl = $"{request.Scheme}://{request.Host}";
        return Results.Ok(new
        {
            version     = cache.Version,
            filename    = cache.Filename,
            sizeBytes   = cache.SizeBytes,
            sha256      = cache.Sha256,
            downloadUrl = $"{baseUrl}/api/model/{cache.Version}/{cache.Filename}"
        });
    }

    private static IResult HandleModelDownload(ModelManifestCache cache, HttpContext httpContext)
    {
        if (!cache.Available || cache.ResolvedModelPath == null)
            return Results.NotFound(new { error = "Model not yet trained. Please train the model first." });

        if (!File.Exists(cache.ResolvedModelPath))
            return Results.NotFound(new { error = "Model file not found at expected path." });

        var etag = $"\"{cache.Sha256}\"";

        // Honour If-None-Match so the browser only re-downloads when the model changes.
        if (httpContext.Request.Headers.IfNoneMatch == etag)
            return Results.StatusCode(StatusCodes.Status304NotModified);

        httpContext.Response.Headers.ETag = etag;
        // no-cache = always revalidate via ETag; 304 = free if model unchanged.
        httpContext.Response.Headers.CacheControl = "no-cache";

        return Results.File(cache.ResolvedModelPath, "application/octet-stream", cache.Filename, enableRangeProcessing: true);
    }

    /// <summary>
    /// Versioned endpoint: /api/model/v1/generator.onnx
    /// Sets ETag (SHA-256), Cache-Control immutable, and supports Range requests.
    /// Returns 304 Not Modified when client's ETag matches.
    /// </summary>
    private static IResult HandleVersionedModelDownload(
        string version, string filename,
        ModelManifestCache cache,
        HttpContext httpContext)
    {
        if (!cache.Available || cache.ResolvedModelPath == null)
            return Results.NotFound(new { error = "Model not yet available" });

        if (!version.Equals(cache.Version, StringComparison.OrdinalIgnoreCase) ||
            !filename.Equals(cache.Filename, StringComparison.OrdinalIgnoreCase))
            return Results.NotFound(new { error = "Model version or filename not found" });

        var etag = $"\"{cache.Sha256}\"";

        // Honour If-None-Match for browser/CDN cache revalidation
        if (httpContext.Request.Headers.IfNoneMatch == etag)
            return Results.StatusCode(StatusCodes.Status304NotModified);

        if (!File.Exists(cache.ResolvedModelPath))
            return Results.NotFound(new { error = "Model file not found at expected path." });

        httpContext.Response.Headers.ETag = etag;
        // no-cache = always revalidate via ETag; immutable must not be used on a mutable URL.
        // Browser sends If-None-Match → 304 if unchanged (free), 200 if model was retrained.
        httpContext.Response.Headers.CacheControl = "no-cache";

        return Results.File(cache.ResolvedModelPath, "application/octet-stream", filename, enableRangeProcessing: true);
    }
}
