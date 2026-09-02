// React 진입점 — #root 에 App 마운트, 전역 스타일 로드

import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles.css'

// StrictMode 는 개발 모드에서 부수효과 검증을 위해 컴포넌트를 두 번 렌더링함
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
