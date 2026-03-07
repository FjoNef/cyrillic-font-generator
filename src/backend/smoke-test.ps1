#!/usr/bin/env pwsh
# Smoke test: verify model endpoints are working
# Usage: ./smoke-test.ps1 [-BaseUrl http://localhost:5000]

param(
    [string]$BaseUrl = "http://localhost:5000"
)

$ErrorActionPreference = "Stop"
$pass = 0
$fail = 0

function Test-Endpoint {
    param([string]$Name, [string]$Url, [scriptblock]$Assert)
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -ErrorAction Stop
        & $Assert $response
        Write-Host "  PASS  $Name" -ForegroundColor Green
        $script:pass++
    } catch {
        Write-Host "  FAIL  $Name - $_" -ForegroundColor Red
        $script:fail++
    }
}

Write-Host "`nSmoke Test: $BaseUrl`n" -ForegroundColor Cyan

Test-Endpoint "Health check" "$BaseUrl/health" {
    param($r) if ($r.StatusCode -ne 200) { throw "Expected 200, got $($r.StatusCode)" }
}

Test-Endpoint "Model manifest (200)" "$BaseUrl/api/model/manifest" {
    param($r)
    if ($r.StatusCode -ne 200) { throw "Expected 200, got $($r.StatusCode)" }
    $json = $r.Content | ConvertFrom-Json
    if (-not $json.sha256) { throw "Missing sha256 in manifest" }
    if (-not $json.downloadUrl) { throw "Missing downloadUrl in manifest" }
    Write-Host "         sha256: $($json.sha256.Substring(0,16))..." -ForegroundColor Gray
    Write-Host "         size:   $($json.sizeBytes) bytes" -ForegroundColor Gray
}

Test-Endpoint "Model download headers" "$BaseUrl/api/model" {
    param($r)
    if ($r.StatusCode -ne 200) { throw "Expected 200, got $($r.StatusCode)" }
    $ct = $r.Headers['Content-Type']
    if ($ct -notlike "*octet-stream*") { throw "Expected octet-stream, got $ct" }
}

Test-Endpoint "Versioned model endpoint" "$BaseUrl/api/model/v1/generator.onnx" {
    param($r)
    if ($r.StatusCode -ne 200) { throw "Expected 200, got $($r.StatusCode)" }
    if (-not $r.Headers['ETag']) { throw "Missing ETag header" }
}

Write-Host "`nResults: $pass passed, $fail failed`n" -ForegroundColor $(if ($fail -eq 0) { "Green" } else { "Red" })
exit $fail
