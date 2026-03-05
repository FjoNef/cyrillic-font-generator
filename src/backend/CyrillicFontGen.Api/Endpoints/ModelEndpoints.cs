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

    public ModelManifestCache(IConfiguration config, IWebHostEnvironment env, ILogger<ModelManifestCache> logger)
    {
        var modelRoot = config["ModelPath"] ?? "models";
        var modelFile = Path.GetFullPath(
            Path.Combine(env.ContentRootPath, modelRoot, Version, Filename));

        if (!File.Exists(modelFile))
        {
            logger.LogWarning("Model file not found at {Path}. Manifest will return 404.", modelFile);
            return;
        }

        try
        {
            var info = new FileInfo(modelFile);
            SizeBytes = info.Length;

            using var fs = File.OpenRead(modelFile);
            var hash = SHA256.HashData(fs);
            Sha256 = Convert.ToHexString(hash).ToLowerInvariant();
            Available = true;

            logger.LogInformation("Model loaded: {File} ({Size} bytes, sha256={Hash})",
                modelFile, SizeBytes, Sha256[..16] + "…");
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Failed to read model file at {Path}", modelFile);
        }
    }
}

public static class ModelEndpoints
{
    public static void MapModelEndpoints(this WebApplication app)
    {
        app.MapGet("/api/model/manifest", HandleManifest);
        app.MapGet("/api/model", HandleModelDownload);
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
            downloadUrl = $"{baseUrl}/models/{cache.Version}/{cache.Filename}"
        });
    }

    private static IResult HandleModelDownload(ModelManifestCache cache, IConfiguration config, IWebHostEnvironment env)
    {
        if (!cache.Available)
            return Results.NotFound(new { error = "Model not yet trained. Please train the model first." });

        var modelRoot = config["ModelPath"] ?? "models";
        var modelFile = Path.GetFullPath(
            Path.Combine(env.ContentRootPath, modelRoot, cache.Version, cache.Filename));

        if (!File.Exists(modelFile))
            return Results.NotFound(new { error = "Model file not found at expected path." });

        return Results.File(modelFile, "application/octet-stream", cache.Filename, enableRangeProcessing: true);
    }
}
