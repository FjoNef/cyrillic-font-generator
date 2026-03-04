---
updated_at: 2026-03-04T23:29:00Z
focus_area: Training resumed at epoch 23/200. ONNX epoch-20 export validated. Two pipeline gaps identified: URL routing (Togusa), backend model path (Batou). Fixes pending, then continue training to epoch 200.
active_issues: ["URL mismatch /api/models/v1/generator.onnx", "Backend model path AppContext.BaseDirectory"]
---

# Now

**Last updated:** 2026-03-04  
**Focus:** Training in progress (~epoch 23/200). ONNX pipeline validation milestone passed with identified gaps.

## Current state
- Branch: dev (aligned with origin/dev)
- **Training:** Resumed after epoch 20 checkpoint export; running on NVIDIA RTX 3070 Laptop GPU
- **Estimated progress:** ~epoch 23 (training at ~3.2 it/s, ~7 min/epoch, ~24 hours total for 200 epochs)
- **ONNX checkpoint:** `models/v1/generator.onnx` (230 MB, fp32, opset 18) — structurally valid, quantization blocked by onnxruntime bug
- **Pipeline validation:** 41 frontend + 4 backend tests pass; 2 critical integration gaps identified (see below)
- **Model quality:** Glyph output poor at epoch 20 (expected); final quality assessment pending epoch 200 re-export

## Critical Issues (Must Fix Before Production)

| Issue | Blocker | Owner | Status |
|-------|---------|-------|--------|
| Frontend URL `/api/models/v1/generator.onnx` → backend has no such route; correct URL is `/api/model` | 🔴 YES | Togusa | Not fixed |
| Backend model path resolves from `AppContext.BaseDirectory` (binary dir) not repo root; file won't be found | 🟡 MEDIUM | Batou | Not fixed |
| Backend happy-path test missing (model present → 200 + stream) | 🟡 MEDIUM | Saito/Batou | Not fixed |

## Next up
1. **Togusa** — Fix URL in `App.tsx` + `ModelLoader.test.ts`: `/api/models/v1/generator.onnx` → `/api/model`
2. **Batou** — Resolve backend model path: copy model to build output OR configure absolute path
3. **Saito** — Add backend test for happy-path (model present)
4. **Saito** — Re-run E2E smoke test after URL/path fixes
5. **Major** — Continue training to epoch 200, then re-export to `models/v1/generator.onnx`
6. **Saito** — Final E2E validation on epoch 200 model

## Completed This Session (2026-03-04)
- ✅ Epoch 20 checkpoint exported to ONNX (230 MB fp32, opset 18)
  - Fixed: onnxscript missing module, opset 17→18 auto-upgrade, INT8 quantization bug fallback, external data sidecar consolidation
- ✅ End-to-end smoke test completed
  - Model file valid ✅
  - Backend endpoints registered ✅
  - Frontend pipeline structure sound ✅
  - **Identified:** URL routing gap ❌ and backend path resolution gap ⚠️
- ✅ Orchestration logs written (Major, Saito)
- ✅ Session log written (ONNX export + E2E)
- ✅ Decisions inbox merged into decisions.md; inbox files deleted

## Blocker Summary
None if URL + path fixes are applied immediately. Training can resume safely in parallel.


