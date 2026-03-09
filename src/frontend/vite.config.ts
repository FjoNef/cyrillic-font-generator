import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'ort-wasm-runtime-external',
      enforce: 'pre',
      resolveId(source) {
        // ORT 1.20 dynamically imports WASM runtime shims from /ort-wasm/.
        // These live in /public and must NOT go through Vite's module pipeline.
        // Vite 5 intercepts dynamic import() calls, resolves them against the
        // module graph, and hard-errors when it finds the file is in /public.
        // Marking these paths as external bypasses bundling and allows the
        // browser to load them directly at runtime.
        //
        // CRITICAL: Must mark BOTH .mjs loaders AND .wasm binaries as external.
        // Otherwise Vite bundles the .wasm files, gives them hashed names, and
        // ORT's loader fails to find them at the expected /ort-wasm/ paths.
        if (/\/ort-wasm\/.*\.(m?js|wasm)/.test(source)) {
          return { id: source, external: true };
        }
      },
    },
  ],
  assetsInclude: ['**/*.wasm'],
  optimizeDeps: {
    // Exclude ORT from pre-bundling: it loads WASM binaries at runtime via dynamic import()
    // and must not be pre-bundled or have its dynamic import() calls rewritten by Vite.
    // 'onnxruntime-web/wasm' is the sub-path used by inferenceWorker and OnnxInference.
    exclude: ['onnxruntime-web', 'onnxruntime-web/wasm'],
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
