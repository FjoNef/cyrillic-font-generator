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
        if (/\/ort-wasm\/.*\.m?js/.test(source)) {
          return { id: source, external: true };
        }
      },
    },
  ],
  assetsInclude: ['**/*.wasm'],
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
});
