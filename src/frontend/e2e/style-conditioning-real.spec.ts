/**
 * Style Conditioning Smoke Test — Real ONNX Model (models/v1/generator.onnx)
 *
 * PRIMARY GOAL: Confirm that the retrained model responds differently to
 * different style_glyphs inputs. If outputs are identical (or near-identical)
 * across two maximally different style inputs, style conditioning is still broken.
 *
 * Model: models/v1/generator.onnx (53.1 MB INT8)
 * Contract: input style_glyphs [1, 10, 1, 128, 128], char_index [1] → generated_glyph [1, 1, 128, 128]
 * Normalisation: white glyph on black bg (+1.0 glyph / -1.0 bg)
 *
 * ⚠️ Chromium-only — the 53 MB WASM compilation is too slow on Firefox/WebKit for a smoke test.
 *    Timeout: 180 s to allow for JIT compilation + two sequential inferences.
 */

import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

// Chromium-only: 53 MB WASM compilation is too slow on Firefox/WebKit for a smoke test.
test.use({ browserName: 'chromium' });

const REAL_MODEL_PATH = path.join(__dirname, '../../../models/v1/generator.onnx');
const ORT_WASM_DIST = path.join(__dirname, '../node_modules/onnxruntime-web/dist');
const ORT_UMD_BUNDLE = path.join(ORT_WASM_DIST, 'ort.wasm.min.js');

// ── Fixture guard ─────────────────────────────────────────────────────────────

if (!fs.existsSync(REAL_MODEL_PATH)) {
  throw new Error(
    `Real model not found at ${REAL_MODEL_PATH}. Run model export before running this test.`
  );
}

// ── Test: Chromium only ───────────────────────────────────────────────────────

test.describe('Style Conditioning: Real Model Smoke Test', () => {
  test.setTimeout(180_000);

  test.beforeEach(async ({ page }) => {
    // Serve ORT WASM files from node_modules
    await page.route('**/ort-wasm-dist/**', async route => {
      const url = new URL(route.request().url());
      const filename = path.basename(url.pathname);
      const filePath = path.join(ORT_WASM_DIST, filename);
      if (fs.existsSync(filePath)) {
        await route.fulfill({
          status: 200,
          contentType: filename.endsWith('.wasm') ? 'application/wasm' : 'application/javascript',
          body: fs.readFileSync(filePath),
        });
      } else {
        await route.continue();
      }
    });

    // Serve the real production model at a dedicated route
    await page.route('**/smoke-real-model/generator.onnx', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/octet-stream',
        body: fs.readFileSync(REAL_MODEL_PATH),
      });
    });

    await page.goto('/');

    // Inject ORT UMD bundle and configure single-threaded WASM
    await page.addScriptTag({ content: fs.readFileSync(ORT_UMD_BUNDLE, 'utf-8') });
    await page.evaluate(() => {
      const ort = (window as Record<string, unknown>).ort as {
        env: { wasm: { wasmPaths: string; numThreads: number } };
      };
      ort.env.wasm.wasmPaths = '/ort-wasm-dist/';
      ort.env.wasm.numThreads = 1; // avoid SharedArrayBuffer COOP requirement
    });
  });

  // ── Shape and range ────────────────────────────────────────────────────────

  test('real model: output shape is [1,1,128,128] and values in [-1,1]', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const ort = (window as Record<string, unknown>).ort as {
        InferenceSession: {
          create(
            buffer: ArrayBuffer,
            opts: Record<string, unknown>
          ): Promise<{
            run(feeds: Record<string, unknown>): Promise<
              Record<string, { data: Float32Array; dims: number[]; type: string }>
            >;
          }>;
        };
        Tensor: new (t: string, d: Float32Array | BigInt64Array, s: number[]) => unknown;
      };

      const resp = await fetch('/smoke-real-model/generator.onnx');
      const buf = await resp.arrayBuffer();
      const session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });

      const styleGlyphs = new Float32Array(1 * 10 * 1 * 128 * 128).fill(0.5);
      const out = await session.run({
        style_glyphs: new ort.Tensor('float32', styleGlyphs, [1, 10, 1, 128, 128]),
        char_index: new ort.Tensor('int64', BigInt64Array.from([5n]), [1]),
      });

      const tensor = out['generated_glyph'] ?? Object.values(out)[0];
      const data = new Float32Array(tensor.data); // copy — ORT WASM output aliasing gotcha
      return {
        shape: Array.from(tensor.dims),
        dtype: tensor.type,
        min: Math.min(...data),
        max: Math.max(...data),
      };
    });

    expect(result.shape).toEqual([1, 1, 128, 128]);
    expect(result.dtype).toBe('float32');
    // Allow tiny epsilon for INT8 quantization rounding (~1.2e-7)
    expect(result.min).toBeGreaterThanOrEqual(-1.0 - 1e-6);
    expect(result.max).toBeLessThanOrEqual(1.0 + 1e-6);
  });

  // ── Style conditioning: THE KEY TEST ──────────────────────────────────────

  test(
    'STYLE CONDITIONING: two maximally-different font styles produce different outputs',
    async ({ page }) => {
      const result = await page.evaluate(async () => {
        const ort = (window as Record<string, unknown>).ort as {
          InferenceSession: {
            create(
              buffer: ArrayBuffer,
              opts: Record<string, unknown>
            ): Promise<{
              inputNames: string[];
              run(feeds: Record<string, unknown>): Promise<
                Record<string, { data: Float32Array; dims: number[] }>
              >;
            }>;
          };
          Tensor: new (t: string, d: Float32Array | BigInt64Array, s: number[]) => unknown;
        };

        const resp = await fetch('/smoke-real-model/generator.onnx');
        const buf = await resp.arrayBuffer();
        const session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });

        const inputNames = session.inputNames;
        const CHAR_IDX = 5; // "H" — middle of the style char list

        // Font A: maximally bright (white glyphs, no background) → +1.0
        const styleA = new Float32Array(1 * 10 * 1 * 128 * 128).fill(1.0);
        const resA = await session.run({
          style_glyphs: new ort.Tensor('float32', styleA, [1, 10, 1, 128, 128]),
          char_index: new ort.Tensor('int64', BigInt64Array.from([BigInt(CHAR_IDX)]), [1]),
        });
        const tensorA = resA['generated_glyph'] ?? Object.values(resA)[0];
        const dataA = new Float32Array(tensorA.data); // copy — ORT WASM aliasing gotcha

        // Font B: maximally dark (background only, no glyphs) → -1.0
        const styleB = new Float32Array(1 * 10 * 1 * 128 * 128).fill(-1.0);
        const resB = await session.run({
          style_glyphs: new ort.Tensor('float32', styleB, [1, 10, 1, 128, 128]),
          char_index: new ort.Tensor('int64', BigInt64Array.from([BigInt(CHAR_IDX)]), [1]),
        });
        const tensorB = resB['generated_glyph'] ?? Object.values(resB)[0];
        const dataB = new Float32Array(tensorB.data);

        // Compute mean absolute difference between outputs
        let totalDiff = 0;
        for (let i = 0; i < dataA.length; i++) {
          totalDiff += Math.abs(dataA[i] - dataB[i]);
        }
        const meanAbsDiff = totalDiff / dataA.length;
        const areIdentical = dataA.every((v, i) => v === dataB[i]);

        // Sample first 8 values from each output for diagnostic logging
        const sampleA = Array.from(dataA.slice(0, 8));
        const sampleB = Array.from(dataB.slice(0, 8));

        return {
          inputNames,
          meanAbsDiff,
          areIdentical,
          sampleA,
          sampleB,
        };
      });

      console.log(`[style-conditioning] inputNames: ${result.inputNames.join(', ')}`);
      console.log(`[style-conditioning] Font A output sample: ${result.sampleA.map(v => v.toFixed(4)).join(', ')}`);
      console.log(`[style-conditioning] Font B output sample: ${result.sampleB.map(v => v.toFixed(4)).join(', ')}`);
      console.log(`[style-conditioning] Mean absolute diff: ${result.meanAbsDiff.toFixed(6)}`);
      console.log(`[style-conditioning] Outputs identical: ${result.areIdentical}`);

      // Input names must match the documented contract
      expect(result.inputNames).toContain('style_glyphs');
      expect(result.inputNames).toContain('char_index');

      // Style conditioning is working if outputs are NOT identical
      expect(result.areIdentical).toBe(false);

      // MAD > 0.01 means the model is genuinely responding to style differences
      // A style-invariant model would score ~0.0 here
      expect(result.meanAbsDiff).toBeGreaterThan(0.01);
    }
  );

  // ── Same char, same style → same output (determinism) ────────────────────

  test('determinism: same inputs produce identical outputs on two runs', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const ort = (window as Record<string, unknown>).ort as {
        InferenceSession: {
          create(
            buffer: ArrayBuffer,
            opts: Record<string, unknown>
          ): Promise<{
            run(feeds: Record<string, unknown>): Promise<
              Record<string, { data: Float32Array; dims: number[] }>
            >;
          }>;
        };
        Tensor: new (t: string, d: Float32Array | BigInt64Array, s: number[]) => unknown;
      };

      const resp = await fetch('/smoke-real-model/generator.onnx');
      const buf = await resp.arrayBuffer();
      const session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });

      const style = new Float32Array(1 * 10 * 1 * 128 * 128).fill(0.5);

      const run = async () => {
        const res = await session.run({
          style_glyphs: new ort.Tensor('float32', style, [1, 10, 1, 128, 128]),
          char_index: new ort.Tensor('int64', BigInt64Array.from([3n]), [1]),
        });
        const t = res['generated_glyph'] ?? Object.values(res)[0];
        return new Float32Array(t.data);
      };

      const data1 = await run();
      const data2 = await run();

      return {
        identical: data1.every((v, i) => v === data2[i]),
      };
    });

    expect(result.identical).toBe(true);
  });
});
