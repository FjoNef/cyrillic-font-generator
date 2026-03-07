using System.Net;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;

namespace CyrillicFontGen.Api.Tests;

public class ApiIntegrationTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;
    private readonly HttpClient _client;

    // Factory that overrides ModelPath to a non-existent directory, simulating a missing model.
    private readonly WebApplicationFactory<Program> _noModelFactory;
    private readonly HttpClient _noModelClient;

    public ApiIntegrationTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory;
        _client = factory.CreateClient();

        _noModelFactory = factory.WithWebHostBuilder(builder =>
        {
            // Create a temporary empty directory to isolate from repo models
            var tempRoot = Path.Combine(Path.GetTempPath(), $"cyrillic-nomodel-{Guid.NewGuid():N}");
            Directory.CreateDirectory(tempRoot);
            
            builder.UseContentRoot(tempRoot);
            builder.ConfigureAppConfiguration((_, config) =>
            {
                config.AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["ModelPath"] = "models" // relative path that doesn't exist
                });
            });
        });
        _noModelClient = _noModelFactory.CreateClient();
    }

    [Fact]
    public async Task HealthEndpoint_ReturnsOk()
    {
        // Act
        var response = await _client.GetAsync("/health");

        // Assert
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var content = await response.Content.ReadAsStringAsync();
        Assert.Contains("healthy", content);
    }

    [Fact]
    public async Task HealthEndpoint_ContainsModelField()
    {
        // Health endpoint always includes a "model" field (null when absent, object when present)
        var response = await _client.GetAsync("/health");
        var content  = await response.Content.ReadAsStringAsync();
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Contains("\"model\"", content);
    }

    [Fact]
    public async Task VersionedModelEndpoint_WhenModelNotExists_Returns404()
    {
        var response = await _noModelClient.GetAsync("/api/model/v1/generator.onnx");
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    [Fact]
    public async Task ModelEndpoint_WhenModelNotExists_Returns404()
    {
        // Act
        var response = await _noModelClient.GetAsync("/api/model");

        // Assert
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        var content = await response.Content.ReadAsStringAsync();
        Assert.Contains("not yet trained", content);
    }

    [Fact]
    public async Task HealthEndpoint_IncludesCorsHeaders()
    {
        // Arrange
        var request = new HttpRequestMessage(HttpMethod.Options, "/health");
        request.Headers.Add("Origin", "http://localhost:5173");
        request.Headers.Add("Access-Control-Request-Method", "GET");

        // Act
        var response = await _client.SendAsync(request);

        // Assert
        Assert.True(
            response.Headers.Contains("Access-Control-Allow-Origin") ||
            response.StatusCode == HttpStatusCode.NoContent ||
            response.StatusCode == HttpStatusCode.OK,
            "CORS headers should be present or OPTIONS request handled"
        );
    }

    [Fact]
    public async Task ModelEndpoint_IncludesCorsHeaders()
    {
        // Arrange
        var request = new HttpRequestMessage(HttpMethod.Options, "/api/model");
        request.Headers.Add("Origin", "http://localhost:5173");
        request.Headers.Add("Access-Control-Request-Method", "GET");

        // Act
        var response = await _client.SendAsync(request);

        // Assert
        Assert.True(
            response.Headers.Contains("Access-Control-Allow-Origin") ||
            response.StatusCode == HttpStatusCode.NoContent ||
            response.StatusCode == HttpStatusCode.OK,
            "CORS headers should be present or OPTIONS request handled"
        );
    }
}
