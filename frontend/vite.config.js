import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 백엔드 라우트에는 /api 프리픽스가 없으므로 프록시에서 벗겨서 전달한다.
// docker-compose.yml의 호스트 공개 포트는 8003이다.
// 다른 포트를 사용할 때는 VLVERIFIER_API_TARGET으로 덮어쓴다.
const apiTarget = process.env.VLVERIFIER_API_TARGET || 'http://localhost:8003'

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
