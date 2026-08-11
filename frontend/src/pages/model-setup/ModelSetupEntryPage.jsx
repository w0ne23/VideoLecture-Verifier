import { useNavigate } from 'react-router-dom'

const ENTRIES = [
  {
    path: '/model-setup/models',
    title: 'LLM 모델',
    desc: '등록된 LLM을 확인하고, API 키로 새 모델을 등록합니다.',
  },
  {
    path: '/model-setup/sets',
    title: 'LLM 셋',
    desc: '만들어 둔 셋을 확인·적용하고, 새 셋을 구성합니다.',
  },
]

export default function ModelSetupEntryPage() {
  const navigate = useNavigate()

  return (
    <section className="model-setup ms-entry-page">
      <div className="ms-header-row">
        <h2 className="ms-app-title">Multi-LLM 사용자 설정</h2>
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
