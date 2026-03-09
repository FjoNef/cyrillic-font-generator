# E2E Tests Must Use Real Model for Glyph Quality Validation

**Date:** 2026-03-08  
**Author:** Saito (Tester/QA)  
**Status:** IMPLEMENTED — Commit c0250b2

## Context

User reported that Cyrillic glyphs are still blank and explicitly requested: **"E2E tests should use the ACTUAL production model, NOT the smoke model."**

The existing `full-ui-flow.spec.ts` test used a tiny smoke model (0.1 MB) that produces constant output. This test could not detect blank glyph bugs because it only validated "non-white pixels exist" without checking for actual letter shapes.

## Decision

**E2E tests now use the real production model (50.6 MB) with strict glyph ink validation.**

### Model Assessment

Production model verified working via Python:
- Size: 50.6 MB (models/v1/generator.onnx)
- Output range: [-1.000, 1.000] ✅
- Dark pixels (< 50 brightness): 1303 ✅
- Model produces real ink: TRUE ✅

Attempted FP16 quantization (50.6 MB → 30.7 MB) but FAILED due to ONNX Runtime incompatibility with DynamicQuantizeLinear operator.

Decision: Use full 50.6 MB model with extended timeouts.

### Implementation

**Test Changes (full-ui-flow.spec.ts):**

1. **Model:** Smoke model → Production model (50.6 MB)
2. **Timeouts:**
   - Test suite: 2min → 10min
   - Model load: 30s → 2min
   - Generation: 90s → 5min
3. **Validation:**
   - Old: "Any pixel not pure white" (weak)
   - New: 
     - Count dark pixels (< 100 brightness) per glyph
     - Require ≥3 of 5 sampled glyphs to have >50 dark pixels
     - Check variance (different chars → different ink)
     - Log detailed stats for debugging

### Rationale

1. **Blank Glyph Detection:** Test will FAIL if glyphs are all-white or constant output
2. **Production Fidelity:** Validates actual model behavior, not stub
3. **Regression Prevention:** Catches quality issues that smoke model misses
4. **CI Feasibility:** 10min timeout is acceptable for quality-critical test
5. **Chromium-only:** Already scoped to Chromium (Firefox/WebKit too slow)

### Trade-offs

- **Pro:** Catches real bugs (blank glyphs, constant output, quality regressions)
- **Pro:** Validates production code path end-to-end
- **Pro:** User-requested behavior (explicit requirement)
- **Con:** Slower test (10min vs 2min)
- **Con:** Larger CI artifact (50.6 MB model must be in repo)

**Accepted:** Slower test time is justified by higher bug detection rate.

### CI Impact

- Model file: Already committed to repo (models/v1/generator.onnx)
- Test runs: Chromium only (existing limitation)
- Workflow: No changes needed (timeout already sufficient)
- Performance: +8min per CI run (acceptable for quality gate)

## Alternatives Considered

1. **Keep smoke model:** REJECTED — Cannot detect blank glyphs (user requirement)
2. **FP16 quantization:** REJECTED — ONNX Runtime compatibility issues
3. **INT8 quantization:** NOT ATTEMPTED — Production model already uses INT8
4. **Smaller cGAN export:** NOT ATTEMPTED — Would require retraining
5. **Two-tier testing (smoke + real):** NOT NEEDED — Only one UI flow test exists

## Pattern

**E2E tests should validate production behavior.**

- Smoke/stub models are useful for fast unit tests
- At least one E2E test should use the real model with strict validation
- Quality validation: check for actual expected output, not just "something happened"
- User reports of quality issues → stricter test validation

## Related

- Issue #57: Fix ORT WASM Vite 5 error (parent issue)
- PR squad/57-fix-ort-wasm-vite-error
- Commit c0250b2: test(e2e): use real model in E2E tests, validate actual glyph ink
- History: .squad/agents/saito/history.md (2026-03-08)
