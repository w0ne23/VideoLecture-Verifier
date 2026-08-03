import { useEffect, useState } from 'react'
import { listLectures } from '../lib/api'

const STATUS_LABELS = {
  pending: '대기 중',
  running: '분석 중',
  done: '완료',
  error: '오류',
}

export default function LectureList({ onSelect }) {
  const [lectures, setLectures] = useState([])
  const [errorMessage, setErrorMessage] = useState('')
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setIsLoading(true)

    listLectures()
      .then(rows => {
        if (!cancelled) {
          setLectures(rows)
          setErrorMessage('')
        }
      })
      .catch(error => {
        if (!cancelled) setErrorMessage(String(error?.message || error))
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [])

  if (isLoading) return <p className="list-note">목록을 불러오는 중...</p>
  if (errorMessage) return <p className="error-text">목록 조회 실패: {errorMessage}</p>
  if (lectures.length === 0) return <p className="list-note">업로드된 강의가 없습니다.</p>

  return (
    <ul className="lecture-list">
      {lectures.map(lecture => (
        <li key={lecture.id}>
          <button type="button" className="lecture-row" onClick={() => onSelect(lecture.id)}>
            <span className="lecture-title">{lecture.title}</span>
            <span className={`status-chip status-chip--${lecture.status}`}>
              {STATUS_LABELS[lecture.status] || lecture.status}
            </span>
            <span className="lecture-stage">
              {lecture.status === 'error' ? lecture.error_message : lecture.current_stage}
            </span>
          </button>
        </li>
      ))}
    </ul>
  )
}
