import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      'opentype.js': path.resolve(__dirname, 'node_modules/opentype.js/dist/opentype.module.js'),
    },
  },
  test: {
    setupFiles: ['./src/test-setup.ts'],
    exclude: ['**/node_modules/**', '**/dist/**', '**/e2e/**'],
    deps: {
      optimizer: {
        web: {
          include: ['opentype.js'],
        },
      },
    },
  },
});
