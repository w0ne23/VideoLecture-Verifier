import { useNavigate } from 'react-router-dom'

export default function ModelSetupEntryPage() {
  const navigate = useNavigate()

  return (
    <section className="model-setup ms-entry-page">
      <div className="ms-header-row">
        <h2 className="ms-app-title">Multi-LLM 설정</h2>
      </div>
      <div className="ms-entry-grid">
        <button className="ms-entry-card" type="button" onClick={() => navigate('/model-setup/presets')}>
          <strong>프리셋 목록</strong>
          <span>저장된 프리셋을 검색하고, 적용/수정/삭제할 수 있어요.</span>
        </button>
        <button className="ms-entry-card" type="button" onClick={() => navigate('/model-setup/new')}>
          <strong>새로 만들기</strong>
          <span>이름을 정하고 새 프리셋을 작성해 저장해요.</span>
        </button>
      </div>
    </section>
  )
}
