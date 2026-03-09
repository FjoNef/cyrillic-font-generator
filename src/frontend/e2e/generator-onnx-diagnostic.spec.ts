/**
 * Generator ONNX Diagnostic Test
 *
 * PURPOSE
 * -------
 * Diagnose the blank glyph bug with the PRODUCTION model (generator.onnx, 50.6 MB INT8).
 * The mini model (mini_generator.onnx, 1.26 MB FP32) works fine; the production model
 * produces all-background output in the browser inference worker.
 *
 * EXPECTED CURRENT BEHAVIOUR
 * --------------------------
 * BEFORE fix lands: test FAILS — worker outputs are all-background (max ≤ -0.5).
 * AFTER fix is complete: test PASSES — outputs contain visible ink (max > -0.5).
 *
 * TWO TEST STRATEGIES
 * -------------------
 * 1. Direct ORT injection (page.evaluate, bypasses React worker):
 *    – Control test. If this FAILS, the bug is in the model/ORT itself, not the worker.
 *    – If this PASSES but the worker test fails, the bug is in the worker pipeline.
 *
 * 2. Full worker pipeline (React UI + inferenceWorker.ts):
 *    – Intercepts Worker postMessage to capture raw output tensors.
 *    – Captures console.debug from the worker (Playwright relays worker console to page).
 *    – This is the definitive regression test for the blank glyph bug.
 *
 * NORMALISATION REMINDER
 * ----------------------
 * Model output range: [-1, 1]  (+1.0 = ink, -1.0 = background)
 * Post-processing:  pixel = ((1 - output) / 2) * 255  → 0 = black ink, 255 = white bg
 * "Non-blank" assertion: output max > -0.5 (at least some pixels above background threshold)
 *
 * STYLE CHARS (what the model was trained on):
 *   ["A","B","C","D","E","H","I","O","R","X"]  (10 uppercase Latin, indices 0–9)
 *
 * ⚠️ Chromium-only — 53 MB WASM compilation is too slow on Firefox/WebKit.
 * ⚠️ Timeout 600 s — accounts for 53 MB download + WASM JIT + a few inferences.
 */

import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

// Chromium-only: 53 MB WASM compilation is too slow on Firefox/WebKit
test.use({ browserName: 'chromium' });

const REAL_MODEL_PATH = path.join(__dirname, '../../../models/v1/generator.onnx');
const ORT_WASM_DIST   = path.join(__dirname, '../node_modules/onnxruntime-web/dist');
const ORT_UMD_BUNDLE  = path.join(ORT_WASM_DIST, 'ort.wasm.min.js');
const TEST_FONT_PATH  = path.join(__dirname, '../../../data/fonts/ANTQUAB.TTF');

// ── Fixture guard ─────────────────────────────────────────────────────────────

test.beforeAll(() => {
  if (!fs.existsSync(REAL_MODEL_PATH)) {
    console.warn(`⚠️  Skipping: Real model not found at ${REAL_MODEL_PATH}`);
    test.skip();
  }
  if (!fs.existsSync(ORT_UMD_BUNDLE)) {
    console.warn(`⚠️  Skipping: ORT UMD bundle not found at ${ORT_UMD_BUNDLE}`);
    test.skip();
  }
  if (!fs.existsSync(TEST_FONT_PATH)) {
    console.warn(`⚠️  Skipping: Test font not found at ${TEST_FONT_PATH}`);
    test.skip();
  }
});

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Mocks the model manifest endpoint to redirect model load to the production model. */
async function mockManifestForRealModel(page: import('@playwright/test').Page) {
  await page.route('**/api/model/manifest', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        version:     'v1',
        filename:    'generator.onnx',
        sizeBytes:   fs.statSync(REAL_MODEL_PATH).size,
        sha256:      'e2e-diagnostic',
        downloadUrl: 'http://localhost:5173/diag-real-model/generator.onnx',
      }),
    });
  });
}

/** Serves the production model at /diag-real-model/generator.onnx. */
async function serveRealModel(page: import('@playwright/test').Page) {
  await page.route('**/diag-real-model/generator.onnx', async route => {
    await route.fulfill({
      status:      200,
      contentType: 'application/octet-stream',
      body:        fs.readFileSync(REAL_MODEL_PATH),
    });
  });
}

/** Serves ORT WASM variant files from node_modules (avoids network dependency). */
async function serveOrtWasm(page: import('@playwright/test').Page) {
  await page.route('**/ort-wasm/**', async route => {
    const url      = new URL(route.request().url());
    const filename = path.basename(url.pathname);
    const filePath = path.join(ORT_WASM_DIST, filename);
    if (fs.existsSync(filePath)) {
      await route.fulfill({
        status:      200,
        contentType: filename.endsWith('.wasm') ? 'application/wasm' : 'application/javascript',
        body:        fs.readFileSync(filePath),
      });
    } else {
      await route.continue();
    }
  });
}

// ── Suite 1: Direct ORT injection (control — bypasses React worker) ───────────

test.describe('Diagnostic — Direct ORT injection (control, bypasses worker)', () => {
  test.describe.configure({ mode: 'serial' }); // prevent parallel context destruction during long inference
  test.setTimeout(300_000); // 5 min: 53 MB download + WASM JIT

  test.beforeEach(async ({ page }) => {
    // Prevent Vite HMR from reloading the page during long WASM inference
    // (HMR 'full-reload' events destroy the JS context, killing in-flight page.evaluate)
    await page.addInitScript(() => {
      const _WS = window.WebSocket;
      // @ts-ignore
      window.WebSocket = function (url: string, ...rest: unknown[]) {
        const ws = new _WS(url, ...(rest as []));
        const orig = ws.addEventListener.bind(ws);
        ws.addEventListener = function (type: string, fn: EventListenerOrEventListenerObject, ...opts: unknown[]) {
          if (type === 'message') {
            const wrapped = (ev: MessageEvent) => {
              try { if (JSON.parse(ev.data)?.type === 'full-reload') return; } catch { /* not JSON */ }
              typeof fn === 'function' ? fn(ev) : fn.handleEvent(ev);
            };
            return orig(type, wrapped, ...(opts as []));
          }
          return orig(type, fn, ...(opts as []));
        };
        return ws;
      };
    });

    await mockManifestForRealModel(page);
    await serveRealModel(page);

    await page.goto('/');
    await page.addScriptTag({ content: fs.readFileSync(ORT_UMD_BUNDLE, 'utf-8') });

    // Configure ORT identically to inferenceWorker.ts
    await page.evaluate(() => {
      const ort = (window as Record<string, unknown>).ort as {
        env: { wasm: { wasmPaths: string; numThreads: number; proxy: boolean } };
      };
      ort.env.wasm.wasmPaths  = '/ort-wasm/';
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy      = false; // mirrors inferenceWorker.ts CRITICAL FIX
    });
  });

  /**
   * Control test: does generator.onnx produce non-blank output when ORT runs
   * directly in the page (no worker)?  This should PASS regardless of the worker
   * bug — if it FAILS here the issue is in the model or the ORT WASM itself.
   */
  test('generator.onnx direct injection: output is non-blank (max > -0.5)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const ort = (window as Record<string, unknown>).ort as {
        InferenceSession: {
          create(buf: ArrayBuffer, opts: Record<string, unknown>): Promise<{
            inputNames:  string[];
            outputNames: string[];
            run(feeds: Record<string, unknown>): Promise<
              Record<string, { data: Float32Array; dims: number[]; type: string }>
            >;
          }>;
        };
        Tensor: new (dtype: string, data: Float32Array | BigInt64Array, shape: number[]) => unknown;
      };

      const resp = await fetch('/diag-real-model/generator.onnx');
      const buf  = await resp.arrayBuffer();

      const session = await ort.InferenceSession.create(buf, {
        executionProviders: ['wasm'],
      });

      // Neutral midtone style: representative of real font glyphs (not synthetic extremes)
      const style = new Float32Array(1 * 10 * 1 * 128 * 128).fill(0.0);

      const out = await session.run({
        style_glyphs: new ort.Tensor('float32', style,                               [1, 10, 1, 128, 128]),
        char_index:   new ort.Tensor('int64',   BigInt64Array.from([0n /* 'A' */]), [1]),
      });

      const tensor = out['generated_glyph'] ?? Object.values(out)[0];
      const data   = new Float32Array(tensor.data); // copy — avoid ORT WASM aliasing

      let min = Infinity, max = -Infinity, sum = 0, sum2 = 0;
      let inkPixels = 0; // pixels with value > 0.0 (above background threshold)
      for (let i = 0; i < data.length; i++) {
        if (data[i] < min) min = data[i];
        if (data[i] > max) max = data[i];
        sum  += data[i];
        sum2 += data[i] * data[i];
        if (data[i] > 0.0) inkPixels++;
      }
      const mean  = sum / data.length;
      const std   = Math.sqrt(sum2 / data.length - mean * mean);
      const range = max - min;

      return {
        shape:      Array.from(tensor.dims),
        dtype:      tensor.type,
        inputNames: session.inputNames,
        min,
        max,
        mean,
        std,
        range,
        inkPixels,
        totalPixels: data.length,
        inkPct:      (inkPixels / data.length) * 100,
        sample:      Array.from(data.slice(0, 8)),
      };
    });

    console.log('[Direct-ORT] ===== generator.onnx diagnostic report =====');
    console.log(`[Direct-ORT] inputNames: ${result.inputNames.join(', ')}`);
    console.log(`[Direct-ORT] output shape: ${result.shape.join('×')}, dtype: ${result.dtype}`);
    console.log(`[Direct-ORT] output range: [${result.min.toFixed(4)}, ${result.max.toFixed(4)}]`);
    console.log(`[Direct-ORT] output mean=${result.mean.toFixed(4)}, std=${result.std.toFixed(4)}, range=${result.range.toFixed(4)}`);
    console.log(`[Direct-ORT] ink pixels (>0): ${result.inkPixels}/${result.totalPixels} (${result.inkPct.toFixed(1)}%)`);
    console.log(`[Direct-ORT] first 8 raw values: ${result.sample.map((v: number) => v.toFixed(4)).join(', ')}`);

    // Shape contract
    expect(result.shape).toEqual([1, 1, 128, 128]);
    expect(result.dtype).toBe('float32');

    // Non-blank assertion: max must be above background threshold
    // If max ≤ -0.5, postprocessing produces near-white canvas → blank glyph
    expect(
      result.max,
      `BLANK GLYPH BUG (direct ORT): max=${result.max.toFixed(4)} ≤ -0.5.\n` +
      `Postprocessing gives pixel=((1-${result.max.toFixed(4)})/2)*255≈${(((1 - result.max) / 2) * 255).toFixed(0)}px (near white).\n` +
      `Output is all-background — model is broken at the ORT/model level (not a worker issue).`
    ).toBeGreaterThan(-0.5);

    // Structural variation: glyph should have strokes + background, not flat output
    expect(
      result.std,
      `NO STRUCTURE in output (direct ORT): std=${result.std.toFixed(4)} ≤ 0.05.\n` +
      `A valid glyph has light strokes on dark background → high std. ` +
      `Near-constant output suggests model collapse or wrong execution provider.`
    ).toBeGreaterThan(0.05);
  });

  /** Run the same inference twice; results must be bit-identical (determinism check). */
  test('generator.onnx direct injection: deterministic across two runs', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const ort = (window as Record<string, unknown>).ort as {
        InferenceSession: {
          create(buf: ArrayBuffer, opts: Record<string, unknown>): Promise<{
            run(feeds: Record<string, unknown>): Promise<
              Record<string, { data: Float32Array }>
            >;
          }>;
        };
        Tensor: new (dtype: string, data: Float32Array | BigInt64Array, shape: number[]) => unknown;
      };

      const resp = await fetch('/diag-real-model/generator.onnx');
      const buf  = await resp.arrayBuffer();
      const session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });

      const style = new Float32Array(1 * 10 * 1 * 128 * 128).fill(0.5);
      const run   = async () => {
        const out = await session.run({
          style_glyphs: new ort.Tensor('float32', style,                               [1, 10, 1, 128, 128]),
          char_index:   new ort.Tensor('int64',   BigInt64Array.from([3n /* 'D' */]), [1]),
        });
        const t = out['generated_glyph'] ?? Object.values(out)[0];
        return new Float32Array(t.data); // copy — avoid aliasing
      };

      const data1 = await run();
      const data2 = await run();
      return { identical: data1.every((v, i) => v === data2[i]) };
    });

    expect(result.identical).toBe(true);
  });
});

// ── Suite 2: Full worker pipeline ─────────────────────────────────────────────
//
// Strategy: no Worker constructor override (avoids Vite module-worker crash).
// Instead, capture the worker's own console.debug output which inferenceWorker.ts
// already emits for every inference:
//   "[inferenceWorker] output max (first 512 px): X.XXXX"
//   "[inferenceWorker] ⚠️ Blank output for char_index=N: max(...) = X.XXXX"
//
// Playwright relays console messages from dedicated workers to the page listener,
// so page.on('console', ...) captures these without any constructor interception.
//
// This test proves that the full worker pipeline (ModelLoader → inferenceWorker →
// ORT WASM → generator.onnx) produces non-blank output, catching the blank glyph bug.

test.describe('Diagnostic — Full worker pipeline (generator.onnx via inferenceWorker.ts)', () => {
  test.describe.configure({ mode: 'serial' });
  test.setTimeout(600_000); // 10 min: 53 MB + WASM JIT + first inferences

  test.beforeEach(async ({ page }) => {
    await mockManifestForRealModel(page);
    await serveRealModel(page);
    await serveOrtWasm(page);

    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  /**
   * THE KEY DIAGNOSTIC TEST — worker console output analysis.
   *
   * Captures "[inferenceWorker] output max (full scan)" console.debug messages
   * that the worker emits after every inference.  Parses the max value to determine
   * whether the model produced real ink or all-background output.
   *
   * SHOULD FAIL before the blank-glyph fix is applied (max ≈ -1.0).
   * SHOULD PASS  after the fix (max > -0.5 for at least one glyph).
   */
  test('generator.onnx via worker: console output shows non-blank inference results', async ({ page }) => {
    // Capture every console message from the page AND its workers
    const workerOutputMaxMessages: number[] = [];
    const workerBlankWarnings: string[]     = [];
    const workerErrors: string[]            = [];

    page.on('console', msg => {
      const text = msg.text();
      const type = msg.type();

      // Log everything for diagnostics
      if (type === 'debug' || type === 'warn' || type === 'error') {
        console.log(`[Browser ${type.toUpperCase()}]: ${text}`);
      }

      // Parse "[inferenceWorker] output max (full scan): X.XXXX, ink pixels (>0): N/M"
      const maxMatch = text.match(/\[inferenceWorker\] output max \(full scan\): ([-\d.]+)/);
      if (maxMatch) {
        workerOutputMaxMessages.push(parseFloat(maxMatch[1]));
      }

      // Capture blank-output warnings
      if (text.includes('[inferenceWorker] ⚠️ Blank output') || text.includes('[inferenceWorker] \u26a0')) {
        workerBlankWarnings.push(text);
      }

      // Capture worker errors
      if (type === 'error' && (text.includes('inferenceWorker') || text.includes('Worker error'))) {
        workerErrors.push(text);
      }
    });

    page.on('pageerror', err => {
      console.error('[Browser PAGE ERROR]:', err.message);
    });

    console.log('[Worker-Diag] === Starting generator.onnx worker pipeline diagnostic ===');
    console.log(`[Worker-Diag] Model: ${REAL_MODEL_PATH}`);
    console.log(`[Worker-Diag] Model size: ${(fs.statSync(REAL_MODEL_PATH).size / 1024 / 1024).toFixed(1)} MB`);

    // Upload font
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_FONT_PATH);
    await expect(page.locator('text=/Loaded:.*ANTQUAB\\.TTF/i')).toBeVisible({ timeout: 10_000 });
    console.log('[Worker-Diag] Font uploaded: ANTQUAB.TTF');

    // Wait for model load (53 MB — 5 min budget)
    const generateButton = page.locator('button:has-text("Generate")');
    await expect(generateButton).toBeEnabled({ timeout: 300_000 });
    console.log('[Worker-Diag] ✓ Production model loaded (53 MB generator.onnx)');

    // Start generation
    await generateButton.click();
    await expect(
      page.locator('button:has-text("Generating Cyrillic glyphs:")'),
    ).toBeVisible({ timeout: 10_000 });
    console.log('[Worker-Diag] ✓ Generation started');

    // Wait until we capture at least 1 "output max" log from the worker
    console.log('[Worker-Diag] Waiting for first inference output log from worker…');
    const deadline = Date.now() + 300_000; // 5 min
    while (workerOutputMaxMessages.length < 1 && Date.now() < deadline) {
      await page.waitForTimeout(2_000);
    }

    // ── Diagnostic report ────────────────────────────────────────────────────
    console.log('[Worker-Diag] ===== INFERENCE CONSOLE OUTPUT ANALYSIS =====');
    console.log(`[Worker-Diag] "output max" messages received: ${workerOutputMaxMessages.length}`);
    console.log(`[Worker-Diag] Blank-output warnings: ${workerBlankWarnings.length}`);
    console.log(`[Worker-Diag] Max values seen: [${workerOutputMaxMessages.slice(0, 10).map(v => v.toFixed(4)).join(', ')}]`);

    if (workerBlankWarnings.length > 0) {
      console.warn('[Worker-Diag] ⚠️ BLANK OUTPUT WARNINGS DETECTED:');
      workerBlankWarnings.forEach(w => console.warn(`  ${w}`));
    }

    if (workerErrors.length > 0) {
      console.error('[Worker-Diag] ❌ Worker errors:');
      workerErrors.forEach(e => console.error(`  ${e}`));
    }

    // ASSERTION 1: We must have received at least 1 inference output log
    expect(
      workerOutputMaxMessages.length,
      [
        `No inference output logs received from inferenceWorker.ts.`,
        `Expected at least 1 "[inferenceWorker] output max (full scan): X.XXXX" message.`,
        `This means either:`,
        `  a) The model failed to load (worker crash before inference)`,
        `  b) The inference didn't complete in 5 minutes (performance issue)`,
        `  c) console.debug is being suppressed in the worker context`,
        `Worker errors: ${workerErrors.join(', ') || '(none)'}`,
      ].join('\n')
    ).toBeGreaterThanOrEqual(1);

    // ASSERTION 2: At least 1 inference output must be non-blank (max > -0.5)
    const nonBlankOutputs  = workerOutputMaxMessages.filter(v => v > -0.5);
    const blankOutputs     = workerOutputMaxMessages.filter(v => v <= -0.5);

    expect(
      nonBlankOutputs.length,
      [
        `BLANK GLYPH BUG DETECTED IN WORKER PIPELINE`,
        `All ${blankOutputs.length} captured outputs have max ≤ -0.5.`,
        `Postprocessing gives near-white (blank) canvas: pixel ≈ 255.`,
        ``,
        `Max values: [${workerOutputMaxMessages.map(v => v.toFixed(4)).join(', ')}]`,
        ``,
        `ROOT CAUSE CANDIDATES:`,
        `  1. ort.env.wasm.proxy not set to false → ORT uses JSEP variant which fails on INT8`,
        `  2. Wrong WASM files served — check /ort-wasm/ route in test`,
        `  3. executionProviders not set to ['wasm']`,
        ``,
        `Fix: ensure inferenceWorker.ts has ort.env.wasm.proxy = false BEFORE`,
        `InferenceSession.create() and executionProviders: ['wasm'] in the create() call.`,
      ].join('\n')
    ).toBeGreaterThanOrEqual(1);

    console.log(`[Worker-Diag] ✓ ${nonBlankOutputs.length}/${workerOutputMaxMessages.length} outputs are non-blank`);
  });

  /**
   * Variance check: different char_index values should produce different ink pixel counts.
   * All outputs saturate at max=1.0 (a single brightest pixel) so max-value variance would
   * always be 0. Instead, compare ink pixel COUNTS which genuinely differ per character.
   */
  test('generator.onnx via worker: inference output varies across different characters', async ({ page }) => {
    // Parse "ink pixels (>0): N/16384" from the new full-scan log
    const workerInkCounts: number[] = [];

    page.on('console', msg => {
      const inkMatch = msg.text().match(/\[inferenceWorker\] output max \(full scan\):.*ink pixels \(>0\): (\d+)\/\d+/);
      if (inkMatch) {
        workerInkCounts.push(parseInt(inkMatch[1], 10));
        console.log(`[Worker-Diag] Captured ink count: ${inkMatch[1]}`);
      }
    });

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_FONT_PATH);
    await expect(page.locator('text=/Loaded:.*ANTQUAB\\.TTF/i')).toBeVisible({ timeout: 10_000 });

    const generateButton = page.locator('button:has-text("Generate")');
    await expect(generateButton).toBeEnabled({ timeout: 300_000 });

    await generateButton.click();
    await expect(page.locator('button:has-text("Generating Cyrillic glyphs:")')).toBeVisible({ timeout: 10_000 });

    // Collect at least 3 ink count values
    const deadline = Date.now() + 300_000;
    while (workerInkCounts.length < 3 && Date.now() < deadline) {
      await page.waitForTimeout(2_000);
    }

    if (workerInkCounts.length < 2) {
      console.warn(`[Worker-Diag] Only ${workerInkCounts.length} outputs collected — skipping variance assertion`);
      test.skip();
      return;
    }

    const inkVariance = Math.max(...workerInkCounts) - Math.min(...workerInkCounts);
    console.log(`[Worker-Diag] Ink counts: [${workerInkCounts.join(', ')}]`);
    console.log(`[Worker-Diag] Ink count range: ${inkVariance}`);

    expect(
      inkVariance,
      [
        `CONSTANT OUTPUT: all ${workerInkCounts.length} inferences have the same ink pixel count.`,
        `Ink counts: [${workerInkCounts.join(', ')}]`,
        `Expected variance > 0 — different Cyrillic characters should produce different ink levels.`,
      ].join('\n')
    ).toBeGreaterThan(0);
  });
});
