import { useQuery } from '@tanstack/react-query'
import { listLectures } from '../api/pipeline'

const STATUS_LABELS = {
  pending: '대기 중',
  running: '분석 중',
  done: '완료',
  error: '오류',
}

export default function LectureList({ onSelect }) {
  const { data: lectures = [], isLoading, error } = useQuery({
    queryKey: ['lectures'],
    queryFn: () => listLectures(),
  })

  if (isLoading) return <p className="list-note">목록을 불러오는 중...</p>
  if (error) return <p className="error-text">목록 조회 실패: {String(error?.message || error)}</p>
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
