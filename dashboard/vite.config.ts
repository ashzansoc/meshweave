import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dashboard talks to a MESHWEAVE coordinator. In development, proxy /api
// to it so the browser never needs to know the coordinator's address.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5273,
    proxy: {
      '/api': {
        target: process.env.MESHWEAVE_COORDINATOR || 'http://127.0.0.1:8700',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})