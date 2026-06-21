import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Bind on all interfaces (0.0.0.0) so http://127.0.0.1:5173 and http://localhost:5173
    // both work. Vite's default ("localhost") can bind IPv6 [::1] only, which leaves
    // browsers that resolve localhost to 127.0.0.1 (IPv4) staring at a blank page.
    host: true,
    // Proxy is optional — the app uses the full backend URL from VITE_API_BASE_URL.
    // Uncomment if you want to route through /api instead:
    // proxy: { '/api': { target: 'http://localhost:8000', rewrite: path => path.replace(/^\/api/, '') } }
  },
})
