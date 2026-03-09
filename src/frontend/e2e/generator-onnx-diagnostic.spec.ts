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

test.describe('Diagnostic — Full worker pipeline (generator.onnx via inferenceWorker.ts)', () => {
  test.describe.configure({ mode: 'serial' }); // prevent parallel context destruction during long inference
  test.setTimeout(600_000); // 10 min: 53 MB download + WASM JIT + first inferences

  test.beforeEach(async ({ page }) => {
    // Prevent Vite HMR from reloading the page during long WASM inference
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

    // Capture ALL console output — Playwright relays dedicated-worker console to page
    page.on('console', msg => {
      const type = msg.type();
      if (type === 'debug' || type === 'log' || type === 'warn' || type === 'error') {
        console.log(`[Browser ${type.toUpperCase()}]: ${msg.text()}`);
      }
    });
    page.on('pageerror', err => {
      console.error('[Browser PAGE ERROR]:', err.message);
    });

    await mockManifestForRealModel(page);
    await serveRealModel(page);
    await serveOrtWasm(page);

    // Intercept Worker construction to capture raw output tensors.
    //
    // Strategy: attach our OWN message listener directly to the worker instance
    // (in addition to whatever React attaches). This avoids the pitfall of React
    // using `worker.onmessage =` (property assignment) which would bypass any
    // `addEventListener` override. Both listeners fire independently.
    await page.addInitScript(() => {
      (window as any).__inferenceResults = [];
      const OriginalWorker = window.Worker;

      // @ts-ignore — replace global Worker to tap into all worker instances
      window.Worker = function (scriptURL: string | URL, options?: WorkerOptions) {
        const worker = new OriginalWorker(scriptURL, options);

        // Attach our own listener — does NOT replace React's listener
        worker.addEventListener('message', (event: MessageEvent) => {
          if (event.data?.type === 'result' && event.data.output) {
            // Copy the data safely — ORT WASM may return SAB-backed view
            const raw  = event.data.output;
            const data: Float32Array =
              raw instanceof Float32Array ? new Float32Array(raw) : new Float32Array(raw);
            const sampleSize = Math.min(data.length, 512);

            let min = Infinity, max = -Infinity;
            for (let i = 0; i < sampleSize; i++) {
              if (data[i] < min) min = data[i];
              if (data[i] > max) max = data[i];
            }

            // Count "ink" pixels over full tensor: value > 0.0 in [-1, 1] space
            let inkPixels = 0;
            for (let i = 0; i < data.length; i++) {
              if (data[i] > 0.0) inkPixels++;
            }

            (window as any).__inferenceResults.push({
              requestId:   event.data.requestId,
              minSample:   min,
              maxSample:   max,
              inkPixels:   inkPixels,
              totalPixels: data.length,
              isBlank:     max <= -0.5, // blank = no ink above background threshold
            });
          }
        });

        return worker;
      };
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  /**
   * THE KEY DIAGNOSTIC TEST
   *
   * Runs generator.onnx through the actual inferenceWorker.ts pipeline.
   * Captures raw output tensors via Worker postMessage interception.
   * Checks whether output contains real ink or is all-background (blank glyph bug).
   *
   * SHOULD FAIL before the blank-glyph fix is applied.
   * SHOULD PASS  after the fix (ort.env.wasm.proxy=false + correct WASM backend).
   */
  test('generator.onnx via worker: output should contain non-blank glyph pixels', async ({ page }) => {
    console.log('[Worker-Diag] === Starting generator.onnx worker pipeline diagnostic ===');
    console.log(`[Worker-Diag] Model: ${REAL_MODEL_PATH}`);
    console.log(`[Worker-Diag] Model size: ${(fs.statSync(REAL_MODEL_PATH).size / 1024 / 1024).toFixed(1)} MB`);

    // Upload font to trigger the full pipeline
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_FONT_PATH);
    await expect(page.locator('text=/Loaded:.*ANTQUAB\\.TTF/i')).toBeVisible({ timeout: 10_000 });
    console.log('[Worker-Diag] Font uploaded: ANTQUAB.TTF');

    // Wait for the model to finish loading — 53 MB over Playwright route interception
    const generateButton = page.locator('button:has-text("Generate")');
    await expect(generateButton).toBeEnabled({ timeout: 300_000 }); // 5 min
    console.log('[Worker-Diag] ✓ Production model loaded (53 MB generator.onnx)');

    // Start generation
    await generateButton.click();
    await expect(
      page.locator('button:has-text("Generating Cyrillic glyphs:")'),
    ).toBeVisible({ timeout: 10_000 });
    console.log('[Worker-Diag] ✓ Generation started');

    // Wait for at least 1 raw result tensor to arrive — first glyph proves the pipeline works
    console.log('[Worker-Diag] Waiting for first inference result from worker…');
    await page.waitForFunction(
      () => ((window as any).__inferenceResults?.length ?? 0) >= 1,
      undefined,
      { timeout: 300_000 }, // 5 min: 53MB model + single-threaded WASM init + first inference
    );

    // Read captured results
    const results: Array<{
      requestId:   string;
      minSample:   number;
      maxSample:   number;
      inkPixels:   number;
      totalPixels: number;
      isBlank:     boolean;
    }> = await page.evaluate(() => (window as any).__inferenceResults);

    // ── Diagnostic report ────────────────────────────────────────────────────
    console.log('[Worker-Diag] ===== RAW TENSOR RESULTS FROM WORKER =====');
    console.log(`[Worker-Diag] Total results captured: ${results.length}`);
    console.log(`[Worker-Diag] Expected output range: [-1, 1] (+1=ink, -1=background)`);
    console.log(`[Worker-Diag] Postprocessing: pixel = ((1 - v) / 2) * 255  [0=black, 255=white]`);
    console.log('');

    for (let i = 0; i < results.length; i++) {
      const r = results[i];
      const whiteVal = (((1 - r.maxSample) / 2) * 255).toFixed(0);
      console.log(
        `[Worker-Diag] Glyph ${i + 1}: ` +
        `max(512px)=${r.maxSample.toFixed(4)}, min(512px)=${r.minSample.toFixed(4)}, ` +
        `ink>${r.totalPixels === 16384 ? '0' : '?'}=${r.inkPixels}/${r.totalPixels} ` +
        `(${((r.inkPixels / r.totalPixels) * 100).toFixed(1)}%), ` +
        `blank=${r.isBlank}, ` +
        `brightest_pixel_after_postproc=${whiteVal}px`
      );
    }

    const blankCount   = results.filter(r => r.isBlank).length;
    const nonBlankCount = results.filter(r => !r.isBlank).length;
    const withInkCount  = results.filter(r => r.inkPixels > 50).length;
    const maxValues     = results.map(r => r.maxSample.toFixed(4)).join(', ');

    console.log('');
    console.log(`[Worker-Diag] SUMMARY:`);
    console.log(`[Worker-Diag]   Blank glyphs (max≤-0.5): ${blankCount}/${results.length}`);
    console.log(`[Worker-Diag]   Non-blank glyphs:        ${nonBlankCount}/${results.length}`);
    console.log(`[Worker-Diag]   Glyphs with >50 ink px:  ${withInkCount}/${results.length}`);
    console.log(`[Worker-Diag]   Max values: [${maxValues}]`);

    // PASS criteria: the first glyph must be non-blank.
    // A "blank" glyph has max ≤ -0.5 → postprocessing yields near-white canvas (≥191px brightness).
    //
    // FAIL = blank glyph bug is present → output is all-background.
    expect(
      nonBlankCount,
      [
        `BLANK GLYPH BUG DETECTED IN WORKER PIPELINE`,
        `generator.onnx produces blank output when run through inferenceWorker.ts.`,
        ``,
        `Blank glyphs: ${blankCount}/${results.length}`,
        `Non-blank glyphs: ${nonBlankCount}/${results.length}`,
        `Raw max values: [${maxValues}]`,
        ``,
        `All outputs have max ≤ -0.5, meaning postprocessing gives near-white (blank) canvas.`,
        `Expected: at least 2/${results.length} glyphs to have max > -0.5.`,
        ``,
        `ROOT CAUSE CANDIDATES:`,
        `  1. ort.env.wasm.proxy is not set to false before InferenceSession.create()`,
        `  2. ORT is loading the JSEP (WebGPU) variant instead of plain WASM`,
        `     — the JSEP variant silently fails on INT8 quantized models`,
        `  3. WASM files served from wrong path — check /ort-wasm/ route`,
        `  4. executionProviders not explicitly set to ['wasm']`,
        ``,
        `Check inferenceWorker.ts: ort.env.wasm.proxy = false must appear BEFORE`,
        `ort.InferenceSession.create() is called.`,
      ].join('\n')
    ).toBeGreaterThanOrEqual(1);

    // Ink pixel count assertion: at least 1 glyph should have meaningful ink coverage
    expect(
      withInkCount,
      [
        `NO INK PIXELS DETECTED in any glyph output.`,
        `Expected at least 1 glyph to have >50 ink pixels (value > 0.0 in [-1,1] space).`,
        `Ink pixel counts: ${results.map(r => r.inkPixels).join(', ')}`,
      ].join('\n')
    ).toBeGreaterThanOrEqual(1);
  });

  /**
   * Variance check: different char_index values should produce different outputs.
   * If all 3+ captured glyphs are near-identical, the model is ignoring char_index
   * (a different but related pathology to blank output).
   *
   * Prerequisite: at least 3 results captured (implies the main diagnostic test ran first
   * or the model loaded fast enough to generate multiple glyphs).
   */
  test('generator.onnx via worker: different char_index produces different ink levels', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_FONT_PATH);
    await expect(page.locator('text=/Loaded:.*ANTQUAB\\.TTF/i')).toBeVisible({ timeout: 10_000 });

    const generateButton = page.locator('button:has-text("Generate")');
    await expect(generateButton).toBeEnabled({ timeout: 300_000 });

    await generateButton.click();
    await expect(page.locator('button:has-text("Generating Cyrillic glyphs:")')).toBeVisible({ timeout: 10_000 });

    // Wait for at least 2 results — enough to detect variance between different chars
    await page.waitForFunction(
      () => ((window as any).__inferenceResults?.length ?? 0) >= 2,
      undefined,
      { timeout: 300_000 },
    );

    const results: Array<{ maxSample: number; inkPixels: number; isBlank: boolean }> =
      await page.evaluate(() => (window as any).__inferenceResults);

    const inkCounts = results.map(r => r.inkPixels);
    const variance  = Math.max(...inkCounts) - Math.min(...inkCounts);

    console.log(`[Worker-Diag] Ink pixel counts across ${results.length} glyphs: ${inkCounts.join(', ')}`);
    console.log(`[Worker-Diag] Variance (max-min): ${variance}`);

    // Different characters should produce different amounts of ink.
    // Variance = 0 means the model outputs the same tensor for every char_index
    // (either all-blank or identical non-blank — both are bugs).
    expect(
      variance,
      [
        `CONSTANT OUTPUT: all ${results.length} glyphs have identical ink pixel counts.`,
        `Ink counts: ${inkCounts.join(', ')}`,
        `Expected variance > 0 — different characters (А, Б, В…) have different stroke counts.`,
        `If all counts are 0 → blank glyph bug. If all identical non-zero → char_index ignored.`,
      ].join('\n')
    ).toBeGreaterThan(0);
  });
});
