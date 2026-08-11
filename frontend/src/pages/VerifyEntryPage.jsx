import { useNavigate } from 'react-router-dom'

const ENTRIES = [
  {
    path: '/upload',
    title: '검증할 강의 업로드',
    desc: '강의 영상을 업로드하고 검증을 시작합니다.',
  },
  {
    path: '/lectures',
    title: '검증된 강의 목록',
    desc: '업로드한 강의의 진행 상태와 검증 결과를 확인합니다.',
  },
]

export default function VerifyEntryPage() {
  const navigate = useNavigate()

  return (
    <section className="model-setup ms-entry-page">
      <div className="ms-header-row">
        <h2 className="ms-app-title">강의 검증</h2>
        <button className="ms-back-btn" type="button" onClick={() => navigate('/')} aria-label="메인으로">
          ←
        </button>
      </div>
      <div className="ms-entry-grid">
        {ENTRIES.map(entry => (
          <button
            className="ms-entry-card"
            type="button"
            key={entry.path}
            onClick={() => navigate(entry.path)}
          >
            <strong>{entry.title}</strong>
            <span>{entry.desc}</span>
          </button>
        ))}
      </div>
    </section>
  )
}
