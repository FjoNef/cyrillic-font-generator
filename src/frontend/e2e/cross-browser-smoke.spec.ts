/**
 * Cross-Browser Smoke Tests — Output Shape and Range Assertions
 *
 * Validates that onnxruntime-web WASM inference produces correctly shaped output
 * in the expected [-1, 1] range on all three browsers (Chromium, Firefox, WebKit).
 *
 * These tests run against the stub model (same tensor contract as production):
 *   Input:  style_glyphs [1, 10, 1, 128, 128] float32
 *   Input:  char_index   [1] int64
 *   Output: generated_glyph [1, 1, 128, 128] float32 in [-1, 1]
 */

import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const STUB_MODEL_BYTES = fs.readFileSync(
  path.join(__dirname, '../tests/fixtures/stub-generator.onnx')
);
const ORT_WASM_DIST = path.join(__dirname, '../node_modules/onnxruntime-web/dist');
const ORT_UMD_BUNDLE = path.join(ORT_WASM_DIST, 'ort.wasm.min.js');

async function setupPage(page: import('@playwright/test').Page) {
  await page.route('**/api/model**', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/octet-stream',
      body: STUB_MODEL_BYTES,
    });
  });

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

  await page.goto('/');
  await page.addScriptTag({ content: fs.readFileSync(ORT_UMD_BUNDLE, 'utf-8') });
  await page.evaluate(() => {
    const ort = (window as Record<string, unknown>).ort as {
      env: { wasm: { wasmPaths: string; numThreads: number } };
    };
    ort.env.wasm.wasmPaths = '/ort-wasm-dist/';
    ort.env.wasm.numThreads = 1;
  });
}

async function runOneInference(page: import('@playwright/test').Page, modelArray: number[]) {
  return page.evaluate(async (mArray: number[]) => {
    const ort = (window as Record<string, unknown>).ort as {
      InferenceSession: {
        create(
          buffer: ArrayBuffer,
          options: Record<string, unknown>
        ): Promise<{
          run(feeds: Record<string, unknown>): Promise<
            Record<string, { data: Float32Array; dims: number[]; type: string }>
          >;
        }>;
      };
      Tensor: new (
        type: string,
        data: Float32Array | BigInt64Array,
        dims: number[]
      ) => unknown;
    };

    const modelBytes = new Uint8Array(mArray).buffer;
    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
    });

    const styleGlyphs = new Float32Array(1 * 10 * 1 * 128 * 128).fill(0.5);
    const styleTensor = new ort.Tensor('float32', styleGlyphs, [1, 10, 1, 128, 128]);
    const indexTensor = new ort.Tensor('int64', BigInt64Array.from([5n]), [1]);

    const results = await session.run({
      style_glyphs: styleTensor,
      char_index: indexTensor,
    });

    const outputTensor = results['generated_glyph'] ?? Object.values(results)[0];
    const outputData = Array.from(outputTensor.data);
    return {
      shape: Array.from(outputTensor.dims),
      min: Math.min(...outputData),
      max: Math.max(...outputData),
      dtype: outputTensor.type,
    };
  }, modelArray);
}

test.describe('Cross-Browser Smoke: Inference Output Contract', () => {
  test('output shape is [1, 1, 128, 128]', async ({ page }) => {
    await setupPage(page);
    const result = await runOneInference(page, Array.from(STUB_MODEL_BYTES));
    expect(result.shape).toEqual([1, 1, 128, 128]);
  });

  test('output values are in range [-1.0, 1.0]', async ({ page }) => {
    await setupPage(page);
    const result = await runOneInference(page, Array.from(STUB_MODEL_BYTES));
    expect(result.min).toBeGreaterThanOrEqual(-1.0);
    expect(result.max).toBeLessThanOrEqual(1.0);
  });

  test('output dtype is float32', async ({ page }) => {
    await setupPage(page);
    const result = await runOneInference(page, Array.from(STUB_MODEL_BYTES));
    expect(result.dtype).toBe('float32');
  });

  test('different char_index values produce valid output', async ({ page }) => {
    await setupPage(page);
    const modelArray = Array.from(STUB_MODEL_BYTES);

    const results = await page.evaluate(async (mArray: number[]) => {
      const ort = (window as Record<string, unknown>).ort as {
        InferenceSession: {
          create(
            buffer: ArrayBuffer,
            options: Record<string, unknown>
          ): Promise<{
            run(feeds: Record<string, unknown>): Promise<
              Record<string, { data: Float32Array; dims: number[] }>
            >;
          }>;
        };
        Tensor: new (
          type: string,
          data: Float32Array | BigInt64Array,
          dims: number[]
        ) => unknown;
      };

      const modelBytes = new Uint8Array(mArray).buffer;
      const session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ['wasm'],
      });

      const outputs: { index: number; shapeOk: boolean; rangeOk: boolean }[] = [];
      for (const idx of [0, 32, 65]) {
        const styleGlyphs = new Float32Array(1 * 10 * 1 * 128 * 128).fill(0.5);
        const styleTensor = new ort.Tensor('float32', styleGlyphs, [1, 10, 1, 128, 128]);
        const indexTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(idx)]), [1]);

        const results = await session.run({
          style_glyphs: styleTensor,
          char_index: indexTensor,
        });

        const out = results['generated_glyph'] ?? Object.values(results)[0];
        const data = Array.from(out.data);
        outputs.push({
          index: idx,
          shapeOk:
            out.dims[0] === 1 && out.dims[1] === 1 && out.dims[2] === 128 && out.dims[3] === 128,
          rangeOk: Math.min(...data) >= -1.0 && Math.max(...data) <= 1.0,
        });
      }
      return outputs;
    }, modelArray);

    for (const r of results) {
      expect(r.shapeOk, `char_index ${r.index}: shape wrong`).toBe(true);
      expect(r.rangeOk, `char_index ${r.index}: range wrong`).toBe(true);
    }
  });
});
