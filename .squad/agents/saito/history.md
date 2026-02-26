# Saito — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **My focus:** Tests, quality, cross-browser validation, inference performance.

## Learnings
<!-- Append new entries below -->

### 2026-02-26: PR #15 Review — APPROVED (opentype.js CJS/ESM interop fix)

**PR #15 — fix/togusa-opentype-vitest-interop → dev:**
- **Verdict:** ✅ APPROVED — Merged to dev
- **What was verified:**
  1. **opentype.module.js availability** ✅ — Confirmed dist/opentype.module.js exists in node_modules.
  2. **All 41 tests pass** ✅ — 5 suites, 100% pass rate under Vitest 1.6 with jsdom.
  3. **vitest.config.ts alias correct** ✅ — `resolve.alias` mapping forces ESM resolution, bypasses CJS bundle.
  4. **test-setup.ts stubs safe** ✅ — `URL.createObjectURL/revokeObjectURL`, `canvas.getContext('2d')`, `Path2D` stubs guarded with `typeof` checks; node-env tests unaffected.
  5. **jsdom directive placed correctly** ✅ — `// @vitest-environment jsdom` in FontLoader.test.ts enables DOM APIs.
- **CI Status:** ✅ All 11 steps pass (build frontend, type check, vitest, dotnet restore/build/test).

### 2026-02-26: PR #14 — APPROVED (CI test failure fix)

**PR #14 — fix/togusa-ci-test-failures → dev:**
- **Verdict:** APPROVED (GitHub self-review restriction → posted as comment)

**What was verified:**

1. **jsdom in devDependencies** ✅ — Correct placement; dev-only test environment dependency, not needed at runtime.

2. **new ModelLoader() in beforeEach** ✅ — Good pattern. Fresh instance per test prevents stale `loadPromise`/worker state leaking between tests — this was the root cause of CI failures.

3. **Removing async from load()** ✅ — Correct. `load()` uses no internal `await`; it constructs and returns `new Promise<void>()` directly and stores it in `this.loadPromise`. Removing `async` is accurate — callers still receive `Promise<void>` but the implementation no longer wraps in an unnecessary implicit promise.

4. **await Promise.resolve() flushes** ✅ — Correctly placed after each `infer()` call and before `mockWorker.postMessage` assertions. `infer()` contains `await this.loadPromise` internally; one microtask flush is sufficient to let that continuation execute and `postMessage` to be called before assertions run.

5. **No other test files modified** ✅ — Diff scoped to exactly 4 files: `package.json`, `package-lock.json`, `ModelLoader.ts` (export added), `ModelLoader.test.ts`.

**Patterns learned:**
- When mocking async code in Vitest, `await Promise.resolve()` is the minimal flush needed to advance microtask continuations (e.g., code after `await somePromise` inside the SUT). One flush suffices when the promise is already resolved.
- Singleton instances in tests cause inter-test state pollution. Always create fresh instances in `beforeEach` for stateful classes.
- Removing `async` from a method that only returns a manually constructed `Promise` (no internal `await`) is a correctness improvement, not a breaking change.

### 2026-02-26: PR #13 — APPROVED (CI TS6133 fix)

**PR #13 — fix/togusa-ci-ts-errors → dev:**
- **Verdict:** APPROVED (GitHub self-review restriction → posted as comment)

**What was verified:**

1. **Exclude patterns correct:** Three patterns added to `tsconfig.json` — `src/**/__tests__/**`, `src/**/*.test.ts`, `src/**/*.spec.ts`. All 5 test files on the branch are covered. No production files (.ts/.tsx) are accidentally excluded.

2. **All 7 CI TS6133 errors addressed:** Root cause was `"include": ["src"]` with `noUnusedLocals/noUnusedParameters: true` pulling in test files. Excluding test files from tsc build resolves all errors in a single line.

3. **Vitest unaffected:** Vitest discovers tests by glob (`**/*.{test,spec}.ts`) via its own esbuild transform — independent of tsconfig `include`/`exclude`. All 5 test files still execute under `npx vitest run`.

4. **No vitest.config.ts:** `vite.config.ts` has no `test` block; Vitest uses defaults. Confirmed glob-based discovery is unaffected.

**Patterns learned:**
- tsconfig `include`/`exclude` affects tsc build only, not Vitest's file discovery. These are orthogonal concerns.
- `noUnusedLocals` + `noUnusedParameters` in tsconfig will always cause TS6133 errors if test files are included in the build tsconfig — test helpers by nature declare variables for descriptive clarity.
- The correct fix is excluding test files from the build tsconfig (this PR), not weakening compiler flags.

### 2026-02-26: PR #12 — REQUEST CHANGES (font assembly pipeline)

**PR #12 — feat/togusa-font-assembly → dev:**
- **Verdict:** REQUEST CHANGES (GitHub self-review restriction → posted as comment)

**What was found:**

1. **API surface mismatch (Blocking):** Tests import `GlyphVectorizer` and `FontAssembler` as classes (`new GlyphVectorizer()`, `.vectorize()`, `new FontAssembler()`, `.assemble()`). Implementation exports plain functions `vectorizeGlyph` and `assembleFontFromGlyphs`. 12 of 15 tests fail at import. Fix: add class wrappers.

2. **cyrillicCharset.ts Yo/yo indices violate LOCKED tensor contract (Blocking):** decisions.md LOCKED specifies Yo (U+0401) at model index 6 and yo (U+0451) at index 39. cyrillicCharset.ts places Yo at index 32 (end of uppercase block) and yo at index 65 (end of lowercase block). App.tsx passes this index directly to model inference → model generates wrong characters. Fix: alphabetical ordering in cyrillicCharset.ts.

3. **makeGlyphImages() key type mismatch (Blocking):** Test helper returns `Map<string, Float32Array>` (char string keys), but `assembleFontFromGlyphs` expects `Map<number, Float32Array>` (model index keys). At runtime all `glyphImages.get(index)` calls return undefined → every glyph is blank path. FontAssembler tests 7-11 silently validate an empty font. Fix: return `Map<number, Float32Array>` keyed 0-65.

**What passed:** Coordinate math (X=600/128, Y flip row0→800, row127→-200), threshold `> 0`, opentype.js metrics (UPM=1000, asc=800, desc=-200, adv=600), .notdef slot 0, OFL license in name table, download button gating, progress counter, single inference pass, opentype.js in package.json, FontDownloader lifecycle.

**Patterns learned:**
- Spec-first tests must explicitly document expected API surface (class vs function) to prevent this mismatch.
- cyrillicCharset.ts model index ordering must be validated against decisions.md LOCKED contract at review time — silent ordering bugs produce no TypeScript errors but break inference output.
- Test helper types must match the implementation signatures exactly; `Map<string,…>` vs `Map<number,…>` is a silent bug that produces no assertion failures, just blank output.

### 2026-02-26: Font pipeline spec tests written — feat/togusa-font-assembly

**Task:** Write spec-first tests for three modules Togusa is building on `feat/togusa-font-assembly`.

**Test file created:** `src/frontend/src/fontPipeline.test.ts` (15 test cases)

**GlyphVectorizer (6 tests):**
- Test 1: all-white (-1.0) input → `path.commands.length === 0`
- Test 2: all-black (+1.0) input → `path.commands.length > 0`
- Test 3: single horizontal run (row 64, cols 10-19) → exactly 5 commands (M,L,L,L,Z); X start ≈ 10 × (600/128)
- Test 4: Y-axis flip — row 0 max Y = 800 (ascender), row 127 min Y = -200 (descender)
- Test 5: X scaling — full row spans x [0, 600] (advanceWidth)
- Test 6: threshold — pixel at 0.0 → no ink; pixel at 0.001 → ink (rule: > 0 = ink)

**FontAssembler (6 tests):**
- Test 7: returns `ArrayBuffer` with `byteLength > 0`
- Test 8: parsed font has exactly 67 glyphs (66 Cyrillic + .notdef); slot 0 = `.notdef`
- Test 9: cmap maps А → glyph with unicode 0x0410
- Test 10: Ё sits at glyph slot 7 (Cyrillic index 6, alphabetical position after А,Б,В,Г,Д,Е); unicode 0x0401
- Test 11: `font.names.fontFamily.en` matches passed `familyName` string
- Test 12: empty glyphImages map → valid ArrayBuffer with .notdef

**FontDownloader (3 tests):**
- Test 13: `URL.createObjectURL` called once; `URL.revokeObjectURL` called with the returned URL
- Test 14: intercepted anchor's `.click()` is called; `.download === filename`, `.href === blob URL`
- Test 15: Blob passed to createObjectURL has `type === 'font/otf'`

**Design decisions:**
- Used `// @vitest-environment jsdom` directive for FontDownloader DOM/URL tests
- Defined local `XYCommand` type alias (`PathCommand & { x; y }`) to work with opentype.js union type
- Explicit `as opentype.PathCommand[]` casts on `path.commands` to avoid implicit-any cascade from missing modules
- Tests 4/5 rely on Y mapping formula: `y_top = ascender − row × (1000/128)`, `y_bottom = ascender − (row+1) × (1000/128)`; matches LOCKED font metrics (ascender=800, descender=-200, advanceWidth=600)
- Font Ё ordering: alphabetical position 6 in Cyrillic set (А=0…Е=5, Ё=6) — per decisions.md LOCKED tensor contract; differs from cyrillicCharset.ts which puts Ё at index 32 (model ordering). Togusa's FontAssembler is expected to use alphabetical order for the font glyph array.
- Tests are spec-first: two expected TS2305 errors (GlyphVectorizer, FontAssembler modules not yet created by Togusa). All other TS errors resolved. Vitest runs tests regardless of tsc errors.
- **Cross-team notes:** Togusa implemented all three modules; coordinate mapping bugs found and fixed (X scale 600/128, Y baseline 800-row*scale). Single inference pass eliminates redundancy. Ready for code review and merge.

### 2026-02-26: PR #11 — REQUEST CHANGES (stale duplicate)

**PR #11 — feat/fix-checkpoint-paths → dev:**
- **Verdict:** REQUEST CHANGES / CLOSE AS DUPLICATE
- **GitHub self-review restriction → posted as comment**

**What was found:**

1. **Duplicate fix:** The three output path changes (`model_output_dir`, `checkpoint_dir`, `sample_dir`) in `train_config.yaml` were already applied to dev by the PR #10 squash-merge (commit 3e54f39). The feature branch was branched from `f07d86a` (pre-PR-#10), so PR #10 landed the fix first.

2. **Regression risk:** Because the feature branch pre-dates PR #10, it carries stale values for `fonts_dir` (`../../data/fonts` instead of `data/fonts`) and `style_latin_chars` (old incorrect mixed-case list instead of `["A","B","C","D","E","H","I","O","R","X"]`). Merging would cause merge conflicts, and resolving them in favour of the feature branch would break the tensor contract and font loading.

3. **PR dirty state confirmed:** `mergeable_state: "dirty"` — conflicts exist for the stale fields.

4. **No other missed references:** Scanned all of `src/` for `../../models` and `../../data` patterns — only `train_config.yaml` had them, and they are already correct on dev.

5. **train.py synthetic defaults:** Feature branch's `train.py` has no hardcoded `../../` paths. The flagged issue from PR #10 review appears already resolved.

**Patterns learned:**
- Always verify the merge-base of a PR branch, especially after squash-merges to dev. A fix PR created before a squash-merge can duplicate or regress work.
- `git show <merge-base>:file` and `git merge-base` are essential to diagnose stale-branch PRs.
- PRs in `dirty` merge state against a recently-updated dev deserve extra scrutiny for regressions.

### 2026-02-25T180000: PR #10 APPROVED & MERGED — training pipeline fixes

**PR #10 — feat/major-training-pipeline-fixes → dev:**
- **Verdict:** APPROVED (GitHub self-approve restriction → posted as comment, then merged)
- **Squash-merged to dev; feature branch deleted**

**What was verified:**
1. **model.py — UNet skip-connections fixed:** dec5/dec6/dec7 input channels corrected to asymmetric sums (nf*8+nf*4, nf*4+nf*2, nf*2+nf) matching actual encoder output dimensions; duplicate `self.final` ConvTranspose2d removed; forward pass no longer double-concatenates e1; output [B,1,128,128] float32 Tanh [-1,1]. Contract: PASS.
2. **dataset.py — style chars contract enforced:** `DEFAULT_STYLE_CHARS = ["A","B","C","D","E","H","I","O","R","X"]` uppercase-only 10 chars; fontTools casing fixed; TTFont.close() in finally (Windows locking); SyntheticFontDataset produces correct tensor shapes/dtypes. Contract: PASS.
3. **train_config.yaml:** style_latin_chars = ["A","B","C","D","E","H","I","O","R","X"] ✅; paths relative to repo root (data/fonts, models/). Contract: PASS.
4. **download_fonts.py:** fontTools casing + TTFont.close() Windows file locking fix. PASS.
5. **train.py CLI:** --synthetic/--batch_size/--num_epochs all correctly implemented; 1-epoch real-data validation passed (47,388 samples, no crashes). PASS.

**Minor non-blocking finding:** Synthetic default config in train.py hardcodes `../../models/checkpoints/` (old relative path). Does not affect real training (which uses config file). Suggest follow-up fix.

**Patterns learned:**
- Always verify feature branch content via GitHub API when local checkout is on a different branch
- The recurring style chars bug (lowercase in default list) is a known issue that has now been fixed in both dataset.py, train_config.yaml, AND the synthetic defaults (though checkpoint path in synthetic defaults missed)
- SyntheticFontDataset is a good addition for CI — validates pipeline without requiring 718 font downloads

### 2026-02-25T160138: Major & Batou Sessions — Ready for QA Review

**PR #8 — cGAN Training Pipeline (Major):**
- Deliverables: models/train/{model.py, train.py, dataset.py, export.py, requirements.txt, README.md}
- Architecture: StyleEncoder (mean-pooling) + UNetGenerator (blank canvas, bottleneck conditioning) + PatchDiscriminator
- Loss: Adversarial (BCE) + L1 reconstruction (lambda=100)
- ONNX: opset 17, INT8 weight quantization, dynamic batch, float32 activations
- Tensor contract LOCKED: style_glyphs [B,10,1,128,128] → generated_glyph [B,1,128,128], [-1,1] range
- Character mapping: 0-32 uppercase (А–Я with Ё), 33-65 lowercase (а–я with ё), 66 total Russian Cyrillic chars
- **Action:** Validate ONNX export (opset 17, input/output names, shape matching), verify dataset normalization, check training loop convergence metrics

**PR #9 — Backend Integration (Batou):**
- Deliverables: GET /health, GET /api/model endpoints, 4 xUnit integration tests
- Integration tests all passing: health check 200, model endpoint 404 (absent), CORS headers validated
- Safety wrap: Directory.Exists() check prevents test failures when models/ missing
- **Action:** Validate integration tests run successfully, check CORS headers on actual OPTIONS requests, verify /api/model serves ONNX correctly once Major exports it

**QA focus areas:**
1. **PR #8:** ONNX model structure (input/output validation), dataset normalization (-1→1 range correct), training stability
2. **PR #9:** Integration test coverage, CORS behavior, graceful 404 handling for missing model

### 2026-02-25T145755: PR #4 & #5 ready for QA review
- **PR #4 — Inference Pipeline (Togusa):** Web Worker integration complete. Model loading, glyph inference, and UI feedback wired end-to-end. Ready for QA functional testing and performance validation (target: ~15–30ms/glyph WebGL, ~80–150ms/glyph WASM).
- **PR #5 — CI Automation (Batou):** GitHub Actions workflows configured (CI, release, preview, PR auto-label). Ready for QA validation that pipelines run correctly on PRs to dev and pushes to main. Check label creation, reviewer notifications, and release artifact generation.
- **Focus areas for QA:** Inference accuracy on style transfer (test multiple fonts), performance benchmarking, CI pipeline success rate across branches, release automation idempotency, PR auto-label routing correctness.

### 2026-02-25T152700: PR #4 APPROVED — color inversion fix complete
- **Re-review verdict:** APPROVED after Major's fix
- **Fix confirmed:** All instances of pixel mapping now use correct formula `((1 - output[px]) / 2) * 255`
  - Before: `((output[px] + 1) / 2) * 255` — inverted colors (+1→white, -1→black) ❌
  - After: `((1 - output[px]) / 2) * 255` — correct colors (+1→black, -1→white) ✅
- **Files verified:** App.tsx (line 67), OnnxInference.ts (line 90) — both corrected
- **Test coverage:** Added colorMapping.test.ts with 5 passing tests validating the formula
- **Acceptance criteria:** All 5 criteria met (model loading, glyph rendering, 66 chars, Web Worker, .otf download)
- **Outcome:** PR #4 ready to merge → dev

### 2026-02-25T152319: PR #4 blocking bug resolved — color inversion fix
- **Issue:** PR #4 (feat/togusa-inference-pipeline → dev) had inverted color output mapping in App.tsx line 67-70
- **Root cause:** Formula `((output[px] + 1) / 2) * 255` mapped model output incorrectly: -1 (background) → 0 (black), 1 (ink) → 255 (white) — inverted
- **Fix applied by:** Major (AI/ML Engineer)
- **Correction:** Formula changed to `((1 - output[px]) / 2) * 255`: 1 → 0 (black ink), -1 → 255 (white background) — correct
- **Files fixed:** `src/frontend/src/App.tsx`, `src/frontend/src/inference/OnnxInference.ts`
- **Status:** Pushed to feat/togusa-inference-pipeline; PR #4 unblocked, ready for re-review
- **Learning:** Model convention (+1 = foreground, -1 = background per tanh GAN standard) must be documented and enforced at integration boundaries

### 2026-02-25T165444: PR #8 & PR #9 merged to dev — training + backend ready
- **Session:** Final PR review and merge by Saito and Aramaki
- **PR #8 (Major — cGAN Training Pipeline):**
  - **Final status:** ✅ APPROVED & MERGED after Togusa fixed doc conflict
  - **Key fix:** Togusa corrected `.squad/decisions.md` line 184 (Latin chars were wrong: `A,B,H,O,g,n,o,p,s,x` → fixed to `A,B,C,D,E,H,I,O,R,X`)
  - **Also:** Added clarifying comment to `models/train/model.py` explaining 10 uppercase Latin chars
  - **Deliverables:** models/train/{model.py, train.py, dataset.py, export.py, requirements.txt, README.md}
  - **Tensor contract LOCKED:** input [B,10,1,128,128] float32 (style glyphs) + [B] int64 (char index) → output [B,1,128,128] float32 in [-1,1] range
  - **ONNX:** opset 17, INT8 quantization, ready for export post-training
  - **Issue #6 closed:** "Merged in PR #8"
  - **Next phase:** Major trains on Google Fonts, exports model to models/v1/generator.onnx
- **PR #9 (Batou — Backend Integration):**
  - **Final status:** ✅ APPROVED & MERGED by Aramaki
  - **Deliverables:** GET /health, GET /api/model endpoints with 4 xUnit integration tests
  - **Architecture:** Stable /api/model abstraction decouples frontend from versioned file paths
  - **Safety:** Directory.Exists() wraps PhysicalFileProvider, allows tests to run before models/ exists
  - **Range support:** Enables future HTTP Range requests for progressive loading
  - **Tests:** All 4 passing (health 200, model 404 when absent, CORS headers on both endpoints)
  - **Issue #7 closed:** "Merged in PR #9"
  - **Next phase:** Awaits Major's trained ONNX model at models/v1/generator.onnx
- **Decisions inbox merged:** All 5 inbox files consolidated into decisions.md and deleted
- **Session log created:** 20260225T165444-pr8-pr9-merged.md documents handoff to training phase
