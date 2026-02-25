# Decision: PR #4 Changes Requested (Color Inversion Bug)

**Date:** 2026-02-25T150330  
**Reviewer:** Saito (Tester)  
**Author:** Togusa (Frontend Dev)  
**PR:** #4 `feat/togusa-inference-pipeline` → `dev`

---

## What Must Change

**Blocking Bug:** Color inversion in glyph rendering (App.tsx line 67-70)

**Current Code:**
```typescript
// App.tsx line 67-70
const val = Math.round(((output[px] + 1) / 2) * 255);
pixels[px * 4 + 0] = val; // R
pixels[px * 4 + 1] = val; // G
pixels[px * 4 + 2] = val; // B
```

**Problem:**  
This maps model output incorrectly:
- Model: `1 = ink (black), -1 = background (white)` (per decisions.md and FontLoader.ts)
- Code maps: `-1 → 0 (black)`, `1 → 255 (white)`
- Result: Inverted glyphs (black backgrounds, white ink)

**Required Fix:**
```typescript
const val = Math.round(((1 - output[px]) / 2) * 255); // 1 → 0 (black), -1 → 255 (white)
```

**Location:** `src/frontend/src/App.tsx` line 67

---

## Who Should Fix

**Cannot be Togusa** (reviewer lockout — author cannot fix issues found in their own PR review).

**Recommended:** Major (AI/ML Engineer)  
**Rationale:** Major owns the model contract and color convention decisions. This is a model-to-UI mapping issue.

**Alternative:** Batou (Backend Dev) if Major unavailable  
**Rationale:** Batou can make integration fixes.

---

## Verification After Fix

1. Verify formula:
   - Input: `output[px] = 1.0` → Output: `val = 0` (black pixel) ✅
   - Input: `output[px] = -1.0` → Output: `val = 255` (white pixel) ✅
   - Input: `output[px] = 0.0` → Output: `val = 127` (mid-gray) ✅

2. Test with synthetic data:
   ```typescript
   const testOutput = new Float32Array(16384);
   testOutput.fill(1.0); // Should produce all-black image (ink)
   testOutput.fill(-1.0); // Should produce all-white image (background)
   ```

3. Visual inspection:
   - Generated glyphs should have white backgrounds and black ink
   - Not inverted (black backgrounds, white ink)

---

## Additional Context

**Why This Matters:**  
- Generated font glyphs will be unusable (inverted colors)
- Vectorization in `FontLoader.assembleCyrillicFont()` expects correct color convention
- Blocks all downstream testing (font assembly, download, real-world usage)

**Related Files:**
- `src/frontend/src/App.tsx` (line 67 — needs fix)
- `src/frontend/src/font/FontLoader.ts` (line 54-55 — documents correct convention)
- `.squad/decisions.md` (ML engineering decisions — defines model contract)

---

## Timeline

**Target:** Fix and re-submit within 1 business day  
**Blocker For:** PR #4 merge to dev, all downstream integration testing

---

## Full Review

See `.squad/decisions/inbox/saito-pr4-review.md` for complete QA evaluation, including non-blocking suggestions for follow-up PRs.
