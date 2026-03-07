using System.IO.Compression;
using CyrillicFontGen.Api.Endpoints;
using Microsoft.AspNetCore.ResponseCompression;

var builder = WebApplication.CreateBuilder(args);

// CORS — allow Vite dev server
var allowedOrigins = builder.Configuration
    .GetSection("Cors:AllowedOrigins")
    .Get<string[]>() ?? ["http://localhost:5173"];

builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
        policy.WithOrigins(allowedOrigins)
              .AllowAnyHeader()
              .AllowAnyMethod());
});

// Response compression with Brotli for model delivery
builder.Services.AddResponseCompression(options =>
{
    options.EnableForHttps = true;
    options.MimeTypes = ResponseCompressionDefaults.MimeTypes.Concat(
        new[] { "application/octet-stream" });
});
builder.Services.Configure<BrotliCompressionProviderOptions>(options =>
{
    options.Level = CompressionLevel.Optimal;
});

// Model manifest cache (populated at startup, reused per-request)
builder.Services.AddSingleton<ModelManifestCache>();

var app = builder.Build();

app.UseResponseCompression();
app.UseCors();

// -- Static model files: immutable cache + HTTP Range support --
var modelPath = builder.Configuration["ModelPath"] ?? "models";
var modelPhysicalPath = ResolveModelsDirectory(app.Environment.ContentRootPath, modelPath, app.Logger);

if (modelPhysicalPath != null && Directory.Exists(modelPhysicalPath))
{
    app.UseStaticFiles(new StaticFileOptions
    {
        FileProvider = new Microsoft.Extensions.FileProviders.PhysicalFileProvider(modelPhysicalPath),
        RequestPath = "/models",
        OnPrepareResponse = ctx =>
        {
            ctx.Context.Response.Headers.CacheControl = "public, max-age=31536000, immutable";
        }
    });
}

// Enable HTTP Range requests for model streaming
app.UseStaticFiles(); // wwwroot (React SPA)

// -- API routes --
app.MapGet("/health", (ModelManifestCache cache) => Results.Ok(new
{
    status = "healthy",
    model  = cache.Available
        ? new { version = cache.Version, filename = cache.Filename, sizeBytes = cache.SizeBytes, sha256Prefix = cache.Sha256[..16] }
        : null
}));
app.MapFontEndpoints();
app.MapModelEndpoints();

// -- SPA fallback: all unmatched routes → index.html --
app.MapFallbackToFile("index.html");

app.Run();

static string? ResolveModelsDirectory(string contentRootPath, string modelPathConfig, ILogger logger)
{
    // 1. Try the configured path first (respects explicit config)
    var configured = Path.GetFullPath(Path.Combine(contentRootPath, modelPathConfig));
    if (Directory.Exists(configured))
    {
        logger.LogInformation("Models directory found at configured path: {Path}", configured);
        return configured;
    }

    // If the configured path is absolute and doesn't exist, don't walk up
    // (respect explicit absolute path configuration)
    if (Path.IsPathRooted(modelPathConfig))
    {
        logger.LogWarning(
            "Models directory not found at absolute configured path: {Configured}",
            configured);
        return null;
    }

    // 2. Walk up the directory tree looking for models/
    var dir = new DirectoryInfo(contentRootPath);
    while (dir != null)
    {
        var candidate = Path.Combine(dir.FullName, "models");
        if (Directory.Exists(candidate))
        {
            logger.LogInformation("Models directory found by directory walk at: {Path}", candidate);
            return candidate;
        }
        dir = dir.Parent;
    }

    logger.LogWarning(
        "Models directory not found. Tried configured path: {Configured}. Also searched parent directories for models/",
        configured);
    return null;
}

// Expose for integration tests
public partial class Program { }
