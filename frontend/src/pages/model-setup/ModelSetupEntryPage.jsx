import { useNavigate } from 'react-router-dom'
import brainIcon from '../../assets/brain-icon-gold.png'
import llmSetsIcon from '../../assets/group1-icon-gold.png'

const ENTRIES = [
  {
    path: '/model-setup/models',
    title: 'LLM 모델 등록',
    desc: '등록된 LLM을 확인하고, API 키로 새 모델을 등록합니다.',
    icon: brainIcon,
  },
  {
    path: '/model-setup/sets',
    title: 'Multi LLM 셋 만들기',
    desc: '만들어 둔 셋을 확인·적용하고, 새 셋을 구성합니다.',
    icon: llmSetsIcon,
  },
]

export default function ModelSetupEntryPage() {
  const navigate = useNavigate()

  return (
    <section className="model-setup ms-entry-page ms-entry-page--model-setup">
      <div className="ms-header-row">
        <h2 className="ms-app-title">Multi-LLM 사용자 설정</h2>
        <button className="ms-back-btn" type="button" onClick={() => navigate('/')} aria-label="메인으로">
          ←
        </button>
      </div>
      <div className="ms-hero-entry-grid">
        {ENTRIES.map(entry => (
          <button
            className="ms-hero-entry-card"
            type="button"
            key={entry.path}
            onClick={() => navigate(entry.path)}
          >
            <span className="ms-hero-entry-band">
              <img src={entry.icon} alt="" />
            </span>
            <span className="ms-hero-entry-body">
              <strong className="ms-hero-entry-title">{entry.title}</strong>
              <span className="ms-hero-entry-desc">{entry.desc}</span>
            </span>
          </button>
        ))}
      </div>
    </section>
  )
}
