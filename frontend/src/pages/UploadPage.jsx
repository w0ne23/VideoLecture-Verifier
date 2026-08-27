import { useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import UploadForm from '../components/UploadForm'

export default function UploadPage() {
  const navigate = useNavigate()
  const openLecture = useCallback(id => navigate(`/verify/${id}`), [navigate])

  return (
    <section className="upload-page">
      <div className="page-header-row">
        <h2 className="list-heading">검증할 강의 업로드</h2>
        <button className="ms-back-btn" type="button" onClick={() => navigate('/verify')} aria-label="강의 검증으로">
          ←
        </button>
      </div>
      <UploadForm onUploaded={openLecture} />
    </section>
  )
}
