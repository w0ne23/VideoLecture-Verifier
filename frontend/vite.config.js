import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 백엔드 라우트에는 /api 프리픽스가 없으므로 프록시에서 벗겨서 전달한다.
// 백엔드 포트가 다르면 VLVERIFIER_API_TARGET=http://localhost:8001 npm run dev 로 실행.
const apiTarget = process.env.VLVERIFIER_API_TARGET || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api/, ''),
      },
      '/files': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
})
