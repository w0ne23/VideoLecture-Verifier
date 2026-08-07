import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { listLectures } from '../api/pipeline'
import { lectureTagLabel } from '../constants/lectureTags'

const STATUS_LABELS = {
  pending: '대기 중',
  running: '분석 중',
  done: '완료',
  error: '오류',
}

function formatDate(value) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString('ko-KR', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function compareBySort(a, b, sort) {
  if (sort === 'date-asc') {
    return String(a.created_at || '').localeCompare(String(b.created_at || ''))
  }
  if (sort === 'source') {
    const bySource = lectureTagLabel(a.source_tag).localeCompare(lectureTagLabel(b.source_tag), 'ko')
    if (bySource !== 0) return bySource
    return String(b.created_at || '').localeCompare(String(a.created_at || ''))
  }
  if (sort === 'title') {
    return String(a.title || '').localeCompare(String(b.title || ''), 'ko')
  }
  // date-desc (default)
  return String(b.created_at || '').localeCompare(String(a.created_at || ''))
}

export default function LectureList({ onSelect, filters = {} }) {
  const { query = '', sort = 'date-desc', sourceFilter = 'all' } = filters
  const { data: lectures = [], isLoading, error } = useQuery({
    queryKey: ['lectures'],
    queryFn: () => listLectures(),
  })

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return lectures
      .filter(lecture => {
        if (sourceFilter !== 'all' && lecture.source_tag !== sourceFilter) return false
        if (!needle) return true
        return String(lecture.title || '').toLowerCase().includes(needle)
      })
      .slice()
      .sort((a, b) => compareBySort(a, b, sort))
  }, [lectures, query, sort, sourceFilter])

  if (isLoading) return <p className="list-note">목록을 불러오는 중...</p>
  if (error) return <p className="error-text">목록 조회 실패: {String(error?.message || error)}</p>
  if (lectures.length === 0) return <p className="list-note">업로드된 강의가 없습니다.</p>
  if (visible.length === 0) return <p className="list-note">검색 결과가 없습니다.</p>

  return (
    <ul className="lecture-list">
      {visible.map(lecture => (
        <li key={lecture.id}>
          <button type="button" className="lecture-row" onClick={() => onSelect(lecture.id)}>
            <span className="lecture-title">{lecture.title}</span>
            <span className="lecture-row-meta">
              {lecture.source_tag && (
                <span className="tag-chip">{lectureTagLabel(lecture.source_tag)}</span>
              )}
              <span className={`status-chip status-chip--${lecture.status}`}>
                {STATUS_LABELS[lecture.status] || lecture.status}
              </span>
            </span>
            <span className="lecture-stage">
              {lecture.status === 'error' ? lecture.error_message : lecture.current_stage}
              {lecture.created_at ? ` · ${formatDate(lecture.created_at)}` : ''}
            </span>
          </button>
        </li>
      ))}
    </ul>
  )
}
