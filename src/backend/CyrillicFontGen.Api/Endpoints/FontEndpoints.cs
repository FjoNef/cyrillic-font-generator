namespace CyrillicFontGen.Api.Endpoints;

public static class FontEndpoints
{
    // Magic byte signatures for supported font formats
    private static readonly byte[] MagicOTF    = [0x4F, 0x54, 0x54, 0x4F]; // "OTTO"
    private static readonly byte[] MagicTTF    = [0x00, 0x01, 0x00, 0x00]; // TrueType
    private static readonly byte[] MagicTTFtrue = [0x74, 0x72, 0x75, 0x65]; // "true"
    private static readonly byte[] MagicWOFF2  = [0x77, 0x4F, 0x46, 0x32]; // "wOF2"

    public static void MapFontEndpoints(this WebApplication app)
    {
        app.MapPost("/api/font/validate", HandleValidate)
           .DisableAntiforgery();
    }

    private static async Task<IResult> HandleValidate(HttpRequest request)
    {
        if (!request.HasFormContentType)
            return Results.BadRequest(new { error = "Expected multipart/form-data" });

        IFormFile? file;
        try
        {
            var form = await request.ReadFormAsync();
            file = form.Files.Count > 0 ? form.Files[0] : null;
        }
        catch (Exception ex)
        {
            return Results.BadRequest(new { valid = false, error = ex.Message });
        }

        if (file is null || file.Length == 0)
            return Results.BadRequest(new { valid = false, error = "No file uploaded" });

        // Read magic bytes to detect format
        var header = new byte[4];
        await using var stream = file.OpenReadStream();
        var bytesRead = await stream.ReadAsync(header.AsMemory(0, 4));
        if (bytesRead < 4)
            return Results.Ok(new { valid = false, error = "File too small to be a valid font" });

        var format = DetectFormat(header);
        if (format is null)
            return Results.Ok(new
            {
                valid = false,
                fontName = (string?)null,
                hasLatin = false,
                glyphCount = 0,
                error = "Unrecognised font format. Supported: OTF, TTF, WOFF2"
            });

        // For MVP: format detection is the primary validation.
        // Latin glyph presence and glyph count require full font parsing (deferred to frontend via opentype.js).
        return Results.Ok(new
        {
            valid = true,
            fontName = Path.GetFileNameWithoutExtension(file.FileName),
            hasLatin = true,   // assumed — opentype.js validates this client-side
            glyphCount = -1,   // unknown without full parse
            format,
            error = (string?)null
        });
    }

    private static string? DetectFormat(byte[] magic)
    {
        if (magic.SequenceEqual(MagicOTF))     return "OTF";
        if (magic.SequenceEqual(MagicTTF))     return "TTF";
        if (magic.SequenceEqual(MagicTTFtrue)) return "TTF";
        if (magic.SequenceEqual(MagicWOFF2))   return "WOFF2";
        return null;
    }
}
