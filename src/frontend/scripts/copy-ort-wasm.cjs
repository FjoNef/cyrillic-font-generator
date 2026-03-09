/**
 * Copies ALL ORT WASM variant files into public/ort-wasm/.
 *
 * These files are NOT committed to git (public/ort-wasm/ is in .gitignore).
 * This script runs via `postinstall` so they are always present after `npm install`.
 *
 * ORT 1.20 probes for different WASM backend variants based on browser capabilities:
 *   - Base: ort-wasm-simd-threaded.{mjs,wasm} — standard SIMD+threads backend
 *   - JSEP: ort-wasm-simd-threaded.jsep.{mjs,wasm} — JavaScript Execution Provider
 *   - Asyncify: ort-wasm-simd-threaded.asyncify.{mjs,wasm} — async operations support
 *   - JSPI: ort-wasm-simd-threaded.jspi.{mjs,wasm} — JavaScript Promise Integration
 *
 * All 8 files must be present to prevent 404s during ORT's capability probing.
 *
 * NOTE: Non-threaded fallback files (ort-wasm-simd.{mjs,wasm}) are not available in
 * onnxruntime-web 1.24.x. The threaded WASM files work with COOP/COEP headers enabled
 * (which enable SharedArrayBuffer support). Environments without COOP/COEP will not be
 * able to use multi-threaded inference but can still run with numThreads=1 when the
 * headers are set.
 */

'use strict';

const fs   = require('fs');
const path = require('path');

const SRC = path.join(__dirname, '..', 'node_modules', 'onnxruntime-web', 'dist');
const DST = path.join(__dirname, '..', 'public', 'ort-wasm');

const FILES = [
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.jsep.mjs',
  'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
  'ort-wasm-simd-threaded.jspi.mjs',
  'ort-wasm-simd-threaded.jspi.wasm',
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
