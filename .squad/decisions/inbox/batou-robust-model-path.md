# Decision: Robust Model Path Resolution with Directory Walk-Up

**Date:** 2026-03-07  
**Author:** Batou  
**Issue:** #37  
**PR:** #38  
**Status:** Proposed

## Context

The model serving endpoints were returning 404 when `ASPNETCORE_ENVIRONMENT` was not explicitly set to "Development". PR #36 added a relative path workaround (`"ModelPath": "../../../models"`) to `appsettings.Development.json`, but this only worked in Development environment. In other environments, the backend used `appsettings.json` with `ModelPath: "models"` which resolved relative to `ContentRootPath` (the API project directory), not the repo root where the model actually lives.

This made the backend brittle and environment-dependent:
- ✅ Works when `ASPNETCORE_ENVIRONMENT=Development` is set
- ❌ Fails otherwise (default environment is Production, not Development)
- ❌ Requires operators to know the exact relative path based on deployment structure
- ❌ No diagnostics to help troubleshoot

## Decision

Implement a directory walk-up search for model files that works in all environments:

1. **Walk-up algorithm:**
   - Try the configured path first (respects explicit configuration)
   - If the file doesn't exist and the path is relative, walk up the directory tree from `ContentRootPath` looking for `models/v1/generator.onnx`
   - If the configured path is absolute, don't walk up (respects explicit operator intent)

2. **Implementation locations:**
   - `ModelManifestCache` constructor: resolves model file once at startup, stores in `ResolvedModelPath`
   - `Program.cs` static file setup: resolves models directory for static file middleware

3. **Startup diagnostics:**
   - Log success with found path and file size
   - Log failure with expected path for troubleshooting
   - Use visual indicators (✓/✗) for easy scanning

4. **Test isolation:**
   - Test fixtures that need "no model" behavior must use `UseContentRoot()` to set an isolated temp directory
   - Otherwise walk-up will find the repo's real model file

## Rationale

**Why walk-up instead of more config:**
- Works in all environments without environment-specific config
- Resilient to working directory changes
- Standard pattern in tooling (e.g., `.git`, `node_modules` resolution)
- Still respects explicit configuration (tries configured path first)

**Why respect absolute paths:**
- If an operator sets an absolute path, they're being explicit about where the model should be
- Walk-up would violate the principle of least surprise
- Absolute paths are typically used in production with explicit model locations

**Why cache the resolved path:**
- Model file location doesn't change during runtime
- Avoids repeated file system checks on every request
- Resolved once in `ModelManifestCache` constructor, reused in all endpoints

**Why visual indicators in logs:**
- "✓" and "✗" make it instantly clear whether model serving is ready
- Operators can scan logs quickly during deployment
- Follows the Principle of Least Astonishment

## Alternatives Considered

1. **Environment-specific config only (PR #36):**
   - ❌ Brittle: requires setting `ASPNETCORE_ENVIRONMENT=Development` explicitly
   - ❌ Doesn't work in default environment (Production)
   - ❌ Doesn't help production deployments

2. **Require absolute paths in production:**
   - ❌ Adds deployment complexity
   - ❌ Makes local development harder
   - ✅ Could still be used if needed (walk-up respects absolute paths)

3. **Copy model into publish directory:**
   - ❌ Increases published artifact size (53 MB)
   - ❌ Requires build pipeline changes
   - ❌ Doesn't help developers
   - ✅ Would still work (walk-up finds it immediately at configured path)

4. **Search only specific known locations:**
   - ❌ Brittle: breaks when project structure changes
   - ❌ Doesn't handle arbitrary deployment scenarios

## Impact

**Positive:**
- ✅ Works in all environments without environment-specific config
- ✅ Resilient to working directory changes and deployment variations
- ✅ Clear diagnostics for troubleshooting
- ✅ No breaking changes (still tries configured path first)
- ✅ All existing tests pass

**Neutral:**
- Configuration stays simple: `"ModelPath": "models"` as a hint
- Absolute paths still work (walk-up is skipped)

**Risks:**
- Could find an unintended model file in a parent directory if multiple versions exist
  - Mitigation: Logs clearly show which path was resolved
  - Mitigation: Absolute paths skip walk-up for explicit control
- Slight startup overhead from directory traversal
  - Mitigation: Only runs once at startup, result is cached
  - Mitigation: Walk-up stops at first match (typically repo root)

## Verification

- All 26 backend tests pass
- Smoke test script created: `src/backend/smoke-test.ps1`
- Integration tests updated to properly isolate "no model" test cases
- Added test verifying walk-up behavior works

## Open Questions

None. Implementation is complete and tested.

## References

- Issue #37: "Model 404 persists: path resolution is environment-dependent, needs robust fix + smoke test"
- PR #36: "fix: update ModelPath in appsettings.Development.json to resolve model 404 (#35)" (superseded by this approach)
- Batou history: Section "2026-03-07: Robust model path resolution with directory walk-up"
