import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import ActiveLlmSetBanner from '../components/model-setup/ActiveLlmSetBanner'
import { listLectures } from '../api/pipeline'

const STATUS_META = {
  done: { label: '완료', tone: 'done', width: 100 },
  running: { label: '분석 중', tone: 'review', width: 55 },
  pending: { label: '대기 중', tone: 'review', width: 15 },
  error: { label: '오류', tone: 'error', width: 100 },
}

function formatDate(value) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleDateString('ko-KR', { month: '2-digit', day: '2-digit' })
}

export default function MainPage() {
  const navigate = useNavigate()
  const { data: lectures = [] } = useQuery({
    queryKey: ['lectures'],
    queryFn: () => listLectures(),
  })

  const recent = useMemo(
    () =>
      lectures
        .slice()
        .sort((a, b) => String(b.created_at || '').localeCompare(String(a.created_at || '')))
        .slice(0, 3),
    [lectures],
  )

  const openLecture = lecture => navigate(lecture.status === 'done' ? `/result/${lecture.id}` : `/verify/${lecture.id}`)

  return (
    <div className="home-page">
      <div className="hero">
        <svg className="hero-deco" viewBox="0 0 980 420" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          <g fill="none" strokeLinecap="round">
            <path d="M-400 340 L-40 340 C 160 280, 280 380, 480 300 S 820 260, 1040 320 L1400 320" stroke="#e2a91a" strokeWidth="1.4" opacity="0.35" />
            <path d="M-400 300 L-40 300 C 180 240, 300 340, 500 260 S 840 220, 1040 280 L1400 280" stroke="#0d9488" strokeWidth="1.4" opacity="0.3" />
            <path d="M-400 260 L-40 260 C 200 210, 320 300, 520 230 S 860 190, 1040 240 L1400 240" stroke="#7c3aed" strokeWidth="1.4" opacity="0.22" />
            <circle cx="120" cy="80" r="2" fill="#e2a91a" opacity="0.6" />
            <circle cx="220" cy="120" r="1.6" fill="#e2a91a" opacity="0.45" />
            <circle cx="860" cy="70" r="2" fill="#0d9488" opacity="0.5" />
            <circle cx="900" cy="140" r="1.6" fill="#7c3aed" opacity="0.5" />
          </g>
        </svg>
        <div className="hero-inner">
          <h1>
            강의 영상을 <em>멀티모달</em>로 분석하고,
            <br />
            <em>여러 LLM</em>이 강의 속 지식 오류를 탐지한다.
          </h1>
          <p className="lede">
            사용자가 설정한 Multi-LLM으로 강의 영상 속 지식 오류를 찾고 피드백을 제공합니다.
          </p>
        </div>
      </div>

      <div className="actions">
        <div className="action-cell-slot" />
        <div className="action-cell-slot">
          <ActiveLlmSetBanner />
        </div>
        <div className="action-cell-slot" />

        <button type="button" className="action-card card--setup" onClick={() => navigate('/model-setup')}>
          <span className="action-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l-.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
          </span>
          <span className="action-title">Multi-LLM 사용자 설정</span>
          <span className="action-desc">LLM을 등록하고, 검증에 쓸 Multi-LLM 셋을 구성합니다.</span>
          <span className="action-go">
            바로가기
            <svg viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
          </span>
        </button>

        <button type="button" className="action-card card--verify" onClick={() => navigate('/verify')}>
          <span className="action-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <path d="M9 12l2 2 4-4" />
              <circle cx="12" cy="12" r="9" />
            </svg>
          </span>
          <span className="action-title">강의 검증</span>
          <span className="action-desc">강의를 검증하거나 검증된 강의 목록을 확인합니다.</span>
          <span className="action-go">
            바로가기
            <svg viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
          </span>
        </button>

        <button type="button" className="action-card card--stats" onClick={() => navigate('/stats')}>
          <span className="action-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <path d="M4 19V5" />
              <path d="M4 19h16" />
              <path d="M8 16V10" />
              <path d="M12 16V7" />
              <path d="M16 16v-4" />
            </svg>
          </span>
          <span className="action-title">통계</span>
          <span className="action-desc">출처·도메인·길이별 이슈 분포와 수정 전후 비교를 봅니다.</span>
          <span className="action-go">
            바로가기
            <svg viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
          </span>
        </button>
      </div>

      <div className="activity">
        <div className="activity-head">
          <h2>최근 검증한 강의</h2>
          <button type="button" className="activity-head-link" onClick={() => navigate('/lectures')}>전체 보기</button>
        </div>

        {recent.length === 0 ? (
          <p className="activity-empty">아직 검증한 강의가 없습니다.</p>
        ) : (
          <div className="activity-grid">
            {recent.map(lecture => {
              const meta = STATUS_META[lecture.status] || { label: lecture.status, tone: 'review', width: 30 }
              return (
                <button
                  key={lecture.id}
                  type="button"
                  className="activity-item"
                  onClick={() => openLecture(lecture)}
                >
                  <div className="row1">
                    <span className="name">{lecture.title}</span>
                    <span className={`pill pill--${meta.tone}`}>{meta.label}</span>
                  </div>
                  <div className="meta">
                    {lecture.status === 'error' ? lecture.error_message : lecture.current_stage}
                    {lecture.created_at ? ` · ${formatDate(lecture.created_at)}` : ''}
                  </div>
                  <div className="bar-track">
                    <div className={`bar-fill bar-fill--${meta.tone}`} style={{ width: `${meta.width}%` }} />
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </div>

      <footer className="home-footer">
        <div className="home-footer-inner">
          <span>VeriLec</span>
          <span>Multi-LLM 기반 강의 자료 오류 검증</span>
        </div>
      </footer>
    </div>
  )
}
