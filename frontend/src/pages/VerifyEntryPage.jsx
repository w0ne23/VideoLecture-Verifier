import { useNavigate } from 'react-router-dom'
import uploadIcon from '../assets/upload-icon-teal.png'
import listIcon from '../assets/list-icon-teal.png'

const ENTRIES = [
  {
    // 실제 파이프라인이 아직 붙어 있지 않아, 완성된 흐름을 보여주는 데모로 대신 보낸다.
    path: '/dev/verify-demo',
    title: '검증할 강의 업로드',
    desc: '강의 영상을 업로드하고 검증을 시작합니다.',
    icon: uploadIcon,
  },
  {
    path: '/lectures',
    title: '검증된 강의 목록',
    desc: '업로드한 강의의 진행 상태와 검증 결과를 확인합니다.',
    icon: listIcon,
  },
]

export default function VerifyEntryPage() {
  const navigate = useNavigate()

  return (
    <section className="model-setup ms-entry-page ms-entry-page--verify">
      <div className="ms-header-row">
        <h2 className="ms-app-title">강의 검증</h2>
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
