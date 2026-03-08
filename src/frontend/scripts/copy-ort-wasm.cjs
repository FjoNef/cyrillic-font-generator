/**
 * Copies the ORT WASM EP files needed by inferenceWorker.ts into public/ort-wasm/.
 *
 * These files are NOT committed to git (public/ort-wasm/ is in .gitignore).
 * This script runs via `postinstall` so they are always present after `npm install`.
 *
 * Files needed for executionProviders: ['wasm'] in ORT 1.20:
 *   ort-wasm-simd-threaded.mjs  — inner worker script (loads the WASM binary)
 *   ort-wasm-simd-threaded.wasm — the WASM binary (~12 MB)
 */

'use strict';

const fs   = require('fs');
const path = require('path');

const SRC = path.join(__dirname, '..', 'node_modules', 'onnxruntime-web', 'dist');
const DST = path.join(__dirname, '..', 'public', 'ort-wasm');

const FILES = [
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
];

fs.mkdirSync(DST, { recursive: true });

for (const file of FILES) {
  const src = path.join(SRC, file);
  const dst = path.join(DST, file);
  fs.copyFileSync(src, dst);
  const kb = Math.round(fs.statSync(dst).size / 1024);
  console.log(`  copied ${file} → public/ort-wasm/ (${kb} kB)`);
}

console.log('ORT WASM files ready.');
