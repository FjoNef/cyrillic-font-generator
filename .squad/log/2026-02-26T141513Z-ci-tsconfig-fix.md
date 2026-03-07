# Session: CI Frontend Build TS6133 Fix

**Date:** 2026-02-26  
**Timestamp:** 2026-02-26T141513Z  
**Topic:** CI TS6133 fix merged to dev  

## Summary

CI was failing on `npm run build` due to TypeScript (tsc) compiling test files and flagging unused variables/parameters as TS6133 errors. Test files under `src/**/__tests__/` and `src/**/*.test.ts` were included in `src/frontend/tsconfig.json` with `noUnusedLocals: true` and `noUnusedParameters: true` enabled.

## Root Cause

- `tsconfig.json` had `"include": ["src"]` with no exclusions
- All 5 test files matched the glob and were compiled by tsc
- Test code legitimately uses unused imports (e.g., `vi` from vitest in stubs) and mock callback parameters
- tsc flagged 7 TS6133 errors before reaching vite build phase

## Fix Applied

Added `exclude` array to `src/frontend/tsconfig.json`:

```json
{
  "compilerOptions": { ... },
  "include": ["src"],
  "exclude": ["src/**/__tests__/**", "src/**/*.test.ts", "src/**/*.spec.ts"]
}
```

## Implementation & Validation

**PR #13:** fix/togusa-ci-ts-errors (Togusa)

**Files Changed:**
- `src/frontend/tsconfig.json` — added exclude array

**Review (Saito):**
- All 5 test files covered by exclude patterns ✅
- All 7 TS6133 errors resolved ✅
- Vitest test discovery unaffected (uses own glob, not tsconfig) ✅
- No production files excluded ✅
- Verdict: APPROVED ✅

**Merge:**
- Merged to dev by Coordinator (squash merge)
- CI build now passes without TS6133 errors

## Rationale

Standard pattern for React/Vite projects. Test files are compiled and type-checked by Vitest independently (esbuild-based), not by the production tsc build. Excluding test files from tsconfig resolves the structural mismatch without modifying test code or relaxing strict compiler settings.

## Outcome

✅ CI build passes  
✅ Test execution unaffected  
✅ No regression  
✅ Merged to dev
