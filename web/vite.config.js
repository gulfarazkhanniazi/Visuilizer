import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const API = 'http://localhost:5178';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5177,
    proxy: {
      '/api': { target: API, changeOrigin: true },
      '/uploads': { target: API, changeOrigin: true },
    },
  },
  build: { target: 'es2022', chunkSizeWarningLimit: 1200 },
});
