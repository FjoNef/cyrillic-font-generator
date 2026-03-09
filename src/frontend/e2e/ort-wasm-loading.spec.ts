/**
 * ORT WASM Loading E2E Test
 *
 * This test verifies that ORT WASM files are served correctly by the Vite dev server
 * WITHOUT any route mocking for /ort-wasm/ paths.
 *
 * WHY THIS TEST EXISTS:
 * The full-ui-flow.spec.ts test mocks ALL /ort-wasm/ requests via page.route(), which
 * means it never exercises the actual Vite dev server path for those files. This caused
 * us to miss the following hard error:
 *
 *   "Failed to load url /ort-wasm/ort-wasm-simd-threaded.jsep.mjs. This file is in
 *    /public and will be copied as-is during build without going through the plugin
 *    transforms, and therefore should not be imported from source code."
 *
 * ORT 1.20 dynamically import()s JSEP/asyncify/jspi .mjs files for WebGPU feature
 * detection even when executionProviders: ['wasm'] is set. Vite 5 intercepts the
 * dynamic import, finds the target in /public, and returns HTTP 500.
 *
 * WHY WE DO NOT MOCK /ort-wasm/ ROUTES HERE:
 * Mocking those routes is exactly what caused the gap. This test deliberately leaves
 * /ort-wasm/ routes unmocked so that Vite serves them for real — any 500 or module
 * pipeline error will be caught as a test failure.
 *
 * The smoke model route IS still mocked (we don't need real 53MB inference here —
 * we're only testing that ORT WASM files load without errors).
 */

import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const SMOKE_MODEL_PATH = path.join(__dirname, '../../../models/v1/smoke_generator.onnx');
const ORT_WASM_DIST = path.join(__dirname, '../node_modules/onnxruntime-web/dist');

test.describe('ORT WASM Loading', () => {
  // Chromium-only: consistent with other E2E tests that require WASM inference
  test.beforeEach(async ({ browserName }) => {
    test.skip(browserName !== 'chromium', 'Chromium only: WASM loading verification');
  });

  test.beforeAll(() => {
    if (!fs.existsSync(SMOKE_MODEL_PATH)) {
      test.skip();
    }
    if (!fs.existsSync(ORT_WASM_DIST)) {
      test.skip();
    }
  });

  test.setTimeout(30_000);

  test('ORT WASM files are served without Vite module pipeline errors', async ({ page }) => {
    // Collect failed /ort-wasm/ network requests
    const failedOrtRequests: string[] = [];

    page.on('response', response => {
      const url = response.url();
      if (url.includes('/ort-wasm/') && !response.ok()) {
        failedOrtRequests.push(`HTTP ${response.status()} — ${url}`);
      }
    });

    page.on('requestfailed', request => {
      if (request.url().includes('/ort-wasm/')) {
        failedOrtRequests.push(`FAILED — ${request.url()} (${request.failure()?.errorText ?? 'unknown'})`);
      }
    });

    // Collect console errors that match the Vite public-import error pattern
    const vitePublicErrors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        const text = msg.text();
        if (
          text.includes('should not be imported from source code') ||
          text.includes('/public and will be copied') ||
          (text.includes('/ort-wasm/') && (text.includes('.jsep') || text.includes('.jspi') || text.includes('.asyncify')))
        ) {
          vitePublicErrors.push(text);
        }
      }
    });

    // Mock ONLY the model — NOT the /ort-wasm/ routes (that's the critical difference)
    await page.route('**/api/model/manifest', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          version: 'v1',
          filename: 'generator.onnx',
          sizeBytes: fs.statSync(SMOKE_MODEL_PATH).size,
          sha256: 'e2e-test',
          downloadUrl: 'http://localhost:5173/smoke-model/generator.onnx',
        }),
      });
    });

    await page.route('**/smoke-model/generator.onnx', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/octet-stream',
        body: fs.readFileSync(SMOKE_MODEL_PATH),
      });
    });

    // Navigate — this starts the worker and triggers ORT WASM loading
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Give the worker time to initialize and trigger JSEP probe
    await page.waitForTimeout(3_000);

    // Assert: no /ort-wasm/ requests failed
    expect(
      failedOrtRequests,
      `ORT WASM files failed to load:\n${failedOrtRequests.join('\n')}`
    ).toEqual([]);

    // Assert: no Vite public-import errors in console
    expect(
      vitePublicErrors,
      `Vite public-import errors detected (ORT JSEP fix regression):\n${vitePublicErrors.join('\n')}`
    ).toEqual([]);
  });

  test('worker initialization completes without error messages', async ({ page }) => {
    const workerErrors: string[] = [];

    // Inject a script that intercepts Worker postMessage to capture error payloads
    await page.addInitScript(() => {
      const OriginalWorker = window.Worker;
      // @ts-ignore
      window.Worker = function (scriptURL: string | URL, options?: WorkerOptions) {
        const worker = new OriginalWorker(scriptURL, options);
        const originalAddEventListener = worker.addEventListener.bind(worker);
        worker.addEventListener = function (
          type: string,
          listener: EventListenerOrEventListenerObject,
          options?: boolean | AddEventListenerOptions
        ) {
          if (type === 'message') {
            const wrapped = (event: MessageEvent) => {
              if (event.data?.type === 'error') {
                (window as any).__workerErrors = (window as any).__workerErrors || [];
                (window as any).__workerErrors.push(event.data.message);
              }
              if (typeof listener === 'function') {
                listener(event);
              } else {
                listener.handleEvent(event);
              }
            };
            return originalAddEventListener(type, wrapped, options);
          }
          return originalAddEventListener(type, listener, options);
        };
        return worker;
      };
    });

    // Mock model manifest + smoke model (not /ort-wasm/)
    await page.route('**/api/model/manifest', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          version: 'v1',
          filename: 'generator.onnx',
          sizeBytes: fs.statSync(SMOKE_MODEL_PATH).size,
          sha256: 'e2e-test',
          downloadUrl: 'http://localhost:5173/smoke-model/generator.onnx',
        }),
      });
    });

    await page.route('**/smoke-model/generator.onnx', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/octet-stream',
        body: fs.readFileSync(SMOKE_MODEL_PATH),
      });
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3_000);

    // Read any worker errors captured by our intercept
    const capturedErrors: string[] = await page.evaluate(() => {
      return (window as any).__workerErrors || [];
    });

    expect(
      capturedErrors,
      `Worker posted error messages:\n${capturedErrors.join('\n')}`
    ).toEqual([]);
  });
});
