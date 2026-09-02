// Vite 빌드/개발 서버 설정

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// dev 서버 프록시가 /api·/files 요청을 전달할 백엔드 주소
// 백엔드 라우트에는 /api 프리픽스가 없어 프록시에서 제거 후 전달
// docker-compose.yml 의 호스트 공개 포트는 8003, 다른 포트는 VLVERIFIER_API_TARGET 로 덮어씀
const apiTarget = process.env.VLVERIFIER_API_TARGET || 'http://localhost:8003'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // /api 프리픽스를 벗겨 백엔드로 전달 (nginx.conf 와 동일 규칙)
      '/api': {
        target: apiTarget,
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api/, ''),
      },
      // 정적 산출물(영상·슬라이드 이미지)은 프리픽스 유지
      '/files': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
})
