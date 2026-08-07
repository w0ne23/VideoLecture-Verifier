import { useNavigate } from 'react-router-dom'

const ENTRIES = [
  {
    path: '/model-setup/models',
    title: 'LLM 모델 등록',
    desc: 'API 키로 사용할 LLM을 등록합니다.',
  },
  {
    path: '/model-setup/sets/new',
    title: 'LLM 셋 만들기',
    desc: '등록한 LLM으로 검증에 쓸 셋을 구성합니다.',
  },
  {
    path: '/model-setup/sets',
    title: 'LLM 설정 목록',
    desc: '만들어 둔 LLM 셋을 확인하고 적용·수정·삭제합니다.',
  },
]

export default function ModelSetupEntryPage() {
  const navigate = useNavigate()

  return (
    <section className="model-setup ms-entry-page">
      <div className="ms-header-row">
        <h2 className="ms-app-title">Multi-LLM 사용자 설정</h2>
        <button className="ms-link-btn ms-link-btn--compact" type="button" onClick={() => navigate('/')}>
          메인으로
        </button>
      </div>
      <div className="ms-entry-grid ms-entry-grid--three">
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
