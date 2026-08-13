import { useNavigate } from 'react-router-dom'
import ActiveLlmSetBanner from '../components/model-setup/ActiveLlmSetBanner'

export default function MainPage() {
  const navigate = useNavigate()

  return (
    <div className="main-menu main-menu--three">
      <div className="main-menu-cell">
        <div className="main-menu-cell-slot" />
        <button type="button" className="main-menu-item" onClick={() => navigate('/model-setup')}>
          <span className="main-menu-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l-.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
          </span>
          <span className="main-menu-title">Multi-LLM 사용자 설정</span>
          <span className="main-menu-desc">LLM을 등록하고, 검증에 쓸 Multi-LLM 셋을 구성합니다.</span>
        </button>
      </div>

      <div className="main-menu-cell">
        <div className="main-menu-cell-slot">
          <ActiveLlmSetBanner />
        </div>
        <button type="button" className="main-menu-item" onClick={() => navigate('/verify')}>
          <span className="main-menu-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <path d="M9 12l2 2 4-4" />
              <circle cx="12" cy="12" r="9" />
            </svg>
          </span>
          <span className="main-menu-title">강의 검증</span>
          <span className="main-menu-desc">강의를 검증하거나 검증된 강의 목록을 확인합니다.</span>
        </button>
      </div>

      <div className="main-menu-cell">
        <div className="main-menu-cell-slot" />
        <button type="button" className="main-menu-item" onClick={() => navigate('/stats')}>
          <span className="main-menu-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <path d="M4 19V5" />
              <path d="M4 19h16" />
              <path d="M8 16V10" />
              <path d="M12 16V7" />
              <path d="M16 16v-4" />
            </svg>
          </span>
          <span className="main-menu-title">통계</span>
          <span className="main-menu-desc">출처·도메인·길이별 이슈 분포와 수정 전후 비교를 봅니다.</span>
        </button>
      </div>
    </div>
  )
}
