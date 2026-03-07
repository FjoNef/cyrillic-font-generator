# Decision: E2E Glyph Generation UI — Browser Support Gate Pattern

**Author:** Togusa  
**Date:** 2026-03-07  
**Issue:** #27  
**PR:** #31

---

## Decision 1: Browser support detected at module load, not in a React effect

`detectBrowserSupport()` is called once at the top level of `App.tsx` (outside the component), not inside `useEffect`. The result is a module-level constant `browserSupport`.

**Why:** The check is synchronous and cheap. Running it in an effect would add a render cycle before gating the model load. Module-level evaluation means the gate is in place before the first render with zero runtime overhead.

---

## Decision 2: Model load skipped (not error-state) on unsupported browsers

When `browserSupport.supported === false`, the model load `useEffect` returns early. Model status stays at `'idle'`, not `'error'`. The `BrowserUnsupported` banner explains the situation without triggering a "failed to load model" error state.

**Why:** An unsupported browser is not a load failure — it's a capability mismatch. Showing "model load failed" would be misleading. The banner gives actionable guidance (upgrade browser) rather than a generic error.

---

## Decision 3: Model endpoint confirmed as `/api/model` (not `/api/model/v1/generator.onnx`)

The frontend fetches the ONNX model at `GET /api/model`. The backend `ModelEndpoints.HandleModelDownload` serves the binary at this route. The static file path `models/v1/generator.onnx` is an implementation detail of the backend, not part of the public API surface.

**Why:** Team history records this URL fix (Issue #16). Reconfirmed against `ModelEndpoints.cs` — `app.MapGet("/api/model", HandleModelDownload)` is the registered route.

---

## Decision 4: No browser support check in worker — check happens in host only

`browserSupport.ts` gates the entire app in `App.tsx`. The inference worker (`inferenceWorker.ts`) does not independently check browser capabilities. If the host allows the worker to spawn, it is already in a supported context.

**Why:** Defence in depth at the UI layer is sufficient here. Duplicating the check in the worker adds complexity with no user-visible benefit.
