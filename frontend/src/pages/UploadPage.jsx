import { useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import UploadForm from '../components/UploadForm'
import LectureList from '../components/LectureList'

export default function UploadPage() {
  const navigate = useNavigate()
  const openLecture = useCallback(id => navigate(`/lecture/${id}`), [navigate])

  return (
    <>
      <UploadForm onUploaded={openLecture} />
      <h2 className="list-heading">강의 목록</h2>
      <LectureList onSelect={openLecture} />
    </>
  )
}
