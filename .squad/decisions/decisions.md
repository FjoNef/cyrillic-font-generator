# Decisions

## 2026-02-26T12:50:27: User directive — feature branch per iteration (re-stated)
**By:** FjoNef (via Copilot)

**What:** Every iteration of work MUST be on a separate feature branch created from dev. Each iteration ends with a PR to dev. No work lands directly on dev. This supersedes and clarifies any wording that was overwritten in commit 19ba8eb.

**Why:** User re-stated after discovering the policy was lost/diluted in a prior Scribe commit. This is a hard rule with no exceptions.

---

## 2026-03-08: Issue #48 — Cyrillic Glyphs Blank (Root Cause: ORT WASM Path Misconfiguration)
**By:** Major, Togusa, Saito

**What:** Cyrillic glyphs appearing blank in downloaded fonts were caused by missing `ort.env.wasm.wasmPaths` configuration in the browser inference worker. ORT 1.20, when bundled by Vite, cannot auto-resolve WASM binary location from the worker's hashed chunk URL. Without a valid WASM binary, ORT silently falls back to JS-only execution that does not correctly implement INT8 QLinear operators, returning all-background output (-1.0 for every pixel).

**Fixes Applied:**
1. Set `ort.env.wasm.wasmPaths = '/ort-wasm/'` before `InferenceSession.create()`
2. Set `ort.env.wasm.numThreads = 1` (worker already single-threaded; nested threading unnecessary)
3. Add SharedArrayBuffer guard for `styleGlyphs` (mirror existing guard in OnnxInference.ts)
4. Upgrade blank-output check to `console.warn` (detect issue immediately)
5. Create `scripts/copy-ort-wasm.cjs` to copy all 8 ORT WASM variants to `public/ort-wasm/` during postinstall/dev/build

**Decision: WASM Serving Strategy** — ORT WASM files must be served from a stable, non-hashed URL. Using Vite's `public/` directory (served as-is) is correct. Alternative (`?url` imports with hashing) rejected: inner `.mjs` shim has internal imports that would break if binary URL changed independently.

**Why:** Complete inference stack now has correct WASM backend access. No more silent fallback to incomplete JS execution. INT8 model can now run correctly in browser across all ORT 1.20 probed backends (base, JSEP, asyncify, JSPI).

**Files Changed:**
- `src/frontend/src/inference/worker/inferenceWorker.ts`
- `src/frontend/scripts/copy-ort-wasm.cjs` (new)
- `src/frontend/package.json`
- `.gitignore`
- `src/frontend/src/GlyphVectorizer.ts` (added zero-path warning guard)
- `src/frontend/src/App.tsx` (fixed `uploadedFont` in useCallback deps)

**PR:** #50 (APPROVED & MERGED to dev)

---

## 2026-03-08: PR #50 Verdict — ORT WASM All Variants (Infrastructure Fix)
**By:** Saito (QA/Tester)

**What:** PR #50 expanded the ORT WASM copy script from 2 files to 8 files to support all ORT 1.20 variant probing (base + JSEP + asyncify + JSPI, 2 files each).

**Status:** ✅ APPROVED & MERGED

**Rationale:** Code change trivial and correct (file list expansion). All 8 files verified present. E2E test failures pre-existing on dev branch, unrelated to this infrastructure fix. Zero regression risk.

**Why:** Infrastructure fix should not be blocked by pre-existing model bugs. All required ORT WASM variants now available for browser fallback chain.
