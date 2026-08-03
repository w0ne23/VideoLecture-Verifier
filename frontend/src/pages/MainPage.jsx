import { useNavigate } from 'react-router-dom'

export default function MainPage() {
  const navigate = useNavigate()

  return (
    <div className="main-menu">
      <button type="button" className="main-menu-item" onClick={() => navigate('/upload')}>
        <span className="main-menu-title">강의 검증</span>
        <span className="main-menu-desc">강의 영상을 업로드하고 검증 결과를 확인합니다.</span>
      </button>
      <button type="button" className="main-menu-item" onClick={() => navigate('/model-setup')}>
        <span className="main-menu-title">Multi-LLM 설정</span>
        <span className="main-menu-desc">검증 파이프라인 단계별로 사용할 모델을 설정합니다.</span>
      </button>
    </div>
  )
}
