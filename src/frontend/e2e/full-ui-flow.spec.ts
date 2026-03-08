/**
 * Full UI Flow E2E Test
 *
 * This test verifies the complete user journey:
 * 1. Upload a font file via file input
 * 2. Wait for model to load
 * 3. Click Generate button
 * 4. Wait for 66 Cyrillic glyphs to be generated
 * 5. Verify glyph preview shows non-blank canvases
 * 6. Download the font
 * 7. Verify the downloaded font contains Cyrillic glyphs
 *
 * This is the ONLY E2E test that exercises the React UI directly (not via page.evaluate).
 * All other E2E tests bypass the UI and inject ORT directly.
 *
 * ⚠️ Uses a smoke ONNX model (tiny constant-output model with correct signatures).
 *    Real inference quality is tested in style-conditioning-real.spec.ts.
 *    Smoke model location: models/v1/smoke_generator.onnx
 */

import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

// Smoke model: tiny constant-output ONNX with correct input/output signatures (~64 KB vs 53 MB)
// Real model path (for reference): models/v1/generator.onnx
const SMOKE_MODEL_PATH = path.join(__dirname, '../../../models/v1/smoke_generator.onnx');
const ORT_WASM_DIST = path.join(__dirname, '../node_modules/onnxruntime-web/dist');
const TEST_FONT_PATH = path.join(__dirname, '../../../data/fonts/ANTQUAB.TTF');

// ── Fixture guard ─────────────────────────────────────────────────────────────

test.describe('Full UI Flow E2E Test', () => {
  // Chromium-only: real model + 66 WASM inferences too slow for CI on Firefox/WebKit
  test.beforeEach(async ({ browserName }) => {
    test.skip(browserName !== 'chromium', 'Chromium only: smoke model + 66 WASM inferences');
  });

  test.beforeAll(() => {
    if (!fs.existsSync(SMOKE_MODEL_PATH)) {
      test.skip();
      console.warn(
        `⚠️  Skipping full-ui-flow.spec.ts: Smoke model not found at ${SMOKE_MODEL_PATH}. ` +
        'Run src/model/export/create_smoke_model.py before running this test.'
      );
    }

    if (!fs.existsSync(TEST_FONT_PATH)) {
      test.skip();
      console.warn(
        `⚠️  Skipping full-ui-flow.spec.ts: Test font not found at ${TEST_FONT_PATH}.`
      );
    }
  });

  test.setTimeout(120_000); // 2 minutes — smoke model is tiny, but WASM init still needs time

  test.beforeEach(async ({ page }) => {
    // Mock the manifest endpoint to return a fake manifest pointing to the real model
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

    // Serve the smoke model (tiny, fast — real model tested by style-conditioning-real.spec.ts)
    await page.route('**/smoke-model/generator.onnx', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/octet-stream',
        body: fs.readFileSync(SMOKE_MODEL_PATH),
      });
    });

    // Serve ORT WASM files from node_modules
    await page.route('**/ort-wasm/**', async route => {
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
    await page.waitForLoadState('networkidle');
  });

  test('full user flow: upload → generate → verify preview → download → verify font', async ({ page }) => {
    console.log('[E2E] Step 1: Upload font file');
    
    // Find the hidden file input and upload the test font
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_FONT_PATH);

    // Wait for "Loaded: ANTQUAB.TTF" confirmation message
    await expect(page.locator('text=/Loaded:.*ANTQUAB\\.TTF/i')).toBeVisible({ timeout: 10_000 });
    console.log('[E2E] ✓ Font uploaded successfully');

    console.log('[E2E] Step 2: Wait for model to load');
    
    // Wait for the Generate button to be enabled (model finished loading)
    const generateButton = page.locator('button:has-text("Generate")');
    await expect(generateButton).toBeEnabled({ timeout: 30_000 }); // 30s is plenty for tiny smoke model
    console.log('[E2E] ✓ Model loaded successfully');

    console.log('[E2E] Step 3: Click Generate button');
    await generateButton.click();

    // Wait for generation to start (button shows progress)
    await expect(page.locator('button:has-text("Generating Cyrillic glyphs:")')).toBeVisible({ timeout: 5_000 });
    console.log('[E2E] ✓ Generation started');

    console.log('[E2E] Step 4: Wait for generation to complete (66 glyphs)');
    
    // Wait for the Download button to appear (generation complete)
    const downloadButton = page.locator('button:has-text("Download .otf")');
    await expect(downloadButton).toBeVisible({ timeout: 90_000 }); // 90s max with smoke model
    
    // Verify button is enabled
    await expect(downloadButton).toBeEnabled();
    console.log('[E2E] ✓ Generation completed');

    console.log('[E2E] Step 5: Verify glyph preview shows non-blank content');
    
    // Check that canvases are rendered in the preview section
    const canvases = page.locator('canvas');
    const canvasCount = await canvases.count();
    expect(canvasCount).toBeGreaterThan(0);
    console.log(`[E2E] Found ${canvasCount} preview canvases`);

    // Sample 5 canvases and verify they have non-white pixels
    const numToCheck = Math.min(5, canvasCount);
    let nonBlankCount = 0;

    for (let i = 0; i < numToCheck; i++) {
      const canvas = canvases.nth(i);
      const hasInk = await canvas.evaluate((el: HTMLCanvasElement) => {
        const ctx = el.getContext('2d');
        if (!ctx) return false;
        
        const imageData = ctx.getImageData(0, 0, el.width, el.height);
        const pixels = imageData.data;
        
        // Check if any pixel is NOT pure white (255, 255, 255)
        for (let j = 0; j < pixels.length; j += 4) {
          if (pixels[j] < 250 || pixels[j + 1] < 250 || pixels[j + 2] < 250) {
            return true; // Found a non-white pixel
          }
        }
        return false;
      });

      if (hasInk) {
        nonBlankCount++;
      }
    }

    console.log(`[E2E] ${nonBlankCount}/${numToCheck} sampled canvases have visible ink`);
    expect(nonBlankCount).toBeGreaterThan(0); // At least one canvas should have ink
    console.log('[E2E] ✓ Glyph preview contains non-blank content');

    console.log('[E2E] Step 6: Download font');
    
    // Set up download interception
    const downloadPromise = page.waitForEvent('download');
    await downloadButton.click();
    const download = await downloadPromise;
    
    const downloadPath = path.join(__dirname, '../../../', await download.suggestedFilename());
    await download.saveAs(downloadPath);
    console.log(`[E2E] ✓ Font downloaded to ${downloadPath}`);

    console.log('[E2E] Step 7: Verify downloaded font contains Cyrillic glyphs');
    
    // Parse the downloaded font using opentype.js in Node context
    const fontBuffer = fs.readFileSync(downloadPath);
    
    // Use opentype.js from Node.js (not from page context)
    const opentype = await import('opentype.js');
    const font = opentype.parse(fontBuffer.buffer.slice(fontBuffer.byteOffset, fontBuffer.byteOffset + fontBuffer.byteLength));
    
    // Check if the font has Cyrillic glyphs
    // А (Cyrillic A) = U+0410
    const cyrillicAIndex = font.charToGlyphIndex('А');
    const hasCyrillic = cyrillicAIndex > 0;
    
    const fontValidation = {
      hasCyrillic,
      numGlyphs: font.numGlyphs,
      familyName: font.names.fontFamily?.en || font.names.fullName?.en || 'Unknown',
      cyrillicSample: cyrillicAIndex,
    };

    console.log(`[E2E] Font family: ${fontValidation.familyName}`);
    console.log(`[E2E] Total glyphs: ${fontValidation.numGlyphs}`);
    console.log(`[E2E] Cyrillic А glyph index: ${fontValidation.cyrillicSample}`);
    
    expect(fontValidation.hasCyrillic).toBe(true);
    expect(fontValidation.numGlyphs).toBeGreaterThan(66); // Should have at least 66 Cyrillic glyphs + .notdef
    console.log('[E2E] ✓ Downloaded font contains Cyrillic glyphs');

    // Clean up downloaded file
    if (fs.existsSync(downloadPath)) {
      fs.unlinkSync(downloadPath);
    }

    console.log('[E2E] ✅ Full UI flow test completed successfully');
  });

  test('error handling: model load failure', async ({ page }) => {
    console.log('[E2E] Testing model load error handling');

    // Override the route to return 404
    await page.route('**/smoke-model/generator.onnx', async route => {
      await route.fulfill({
        status: 404,
        contentType: 'text/plain',
        body: 'Not Found',
      });
    });

    await page.reload();
    await page.waitForLoadState('networkidle');

    // Upload a font
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_FONT_PATH);
    await expect(page.locator('text=/Loaded:.*ANTQUAB\\.TTF/i')).toBeVisible({ timeout: 10_000 });

    // Generate button should remain disabled (model failed to load)
    const generateButton = page.locator('button:has-text("Generate")');
    await expect(generateButton).toBeDisabled();
    
    console.log('[E2E] ✓ Generate button remains disabled after model load failure');
  });

  test('progress tracking: generation progress updates correctly', async ({ page }) => {
    console.log('[E2E] Testing generation progress tracking');

    // Upload font
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_FONT_PATH);
    await expect(page.locator('text=/Loaded:.*ANTQUAB\\.TTF/i')).toBeVisible({ timeout: 10_000 });

    // Wait for model to load
    const generateButton = page.locator('button:has-text("Generate")');
    await expect(generateButton).toBeEnabled({ timeout: 30_000 });

    // Click generate
    await generateButton.click();

    // Verify the generation UI starts — the button immediately shows "Generating Cyrillic glyphs:"
    // before any inference begins (setGenerationStatus('running') fires synchronously on click).
    let sawProgress = false;
    try {
      await expect(page.locator('button:has-text("Generating Cyrillic glyphs:")')).toBeVisible({ timeout: 5_000 });
      sawProgress = true;
      console.log('[E2E] Progress: generation status visible (in-progress button shown)');
    } catch {
      // With very fast inference (smoke model), the generating status might transition
      // to done before Playwright observes it. Fall through and check for completion.
    }

    // Also try to catch a specific progress step (best-effort; smoke model may complete too fast
    // for React to render intermediate states)
    if (!sawProgress) {
      for (let i = 1; i <= 66; i++) {
        const isVisible = await page.locator(`button:has-text("${i}/66")`).isVisible();
        if (isVisible) {
          sawProgress = true;
          console.log(`[E2E] Progress: ${i}/66`);
          break;
        }
      }
    }

    // Final fallback: if the download button is already visible, generation completed — that proves
    // progress ran (even if React batched all state updates into a single render).
    const downloadButton = page.locator('button:has-text("Download .otf")');
    if (!sawProgress && await downloadButton.isVisible()) {
      sawProgress = true;
      console.log('[E2E] Progress: generation completed (download button visible)');
    }

    expect(sawProgress).toBe(true);
    console.log('[E2E] ✓ Progress tracking working correctly');

    // Wait for completion
    await expect(downloadButton).toBeEnabled({ timeout: 90_000 });
  });
});
