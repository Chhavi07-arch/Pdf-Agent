import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy is optional — the app uses the full backend URL from VITE_API_BASE_URL.
    // Uncomment if you want to route through /api instead:
    // proxy: { '/api': { target: 'http://localhost:8000', rewrite: path => path.replace(/^\/api/, '') } }
  },
})
