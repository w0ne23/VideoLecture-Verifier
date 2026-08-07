import { useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import UploadForm from '../components/UploadForm'

export default function UploadPage() {
  const navigate = useNavigate()
  const openLecture = useCallback(id => navigate(`/verify/${id}`), [navigate])

  return (
    <section className="upload-page">
      <div className="page-header-row">
        <h2 className="list-heading">강의 검증</h2>
        <button type="button" className="btn" onClick={() => navigate('/')}>
          메인으로
        </button>
      </div>
      <p className="upload-page-hint">강의 영상을 업로드하면 검증이 시작됩니다. 진행·결과는 강의 목록에서 확인할 수 있어요.</p>
      <UploadForm onUploaded={openLecture} />
    </section>
  )
}
