// 강의 목록 그리드 — 검색·정렬·출처 필터 적용 후 페이지네이션

import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { listLectures } from '../api/pipeline'
import { lectureTagLabel } from '../constants/lectureTags'

// job 상태 → 상태 칩 라벨
const STATUS_LABELS = {
  pending: '대기 중',
  running: '분석 중',
  done: '완료',
  error: '오류',
}

// 페이지당 카드 수
const PAGE_SIZE = 6

// ISO 문자열 → ko-KR 날짜 (YYYY. MM. DD)
function formatDate(value) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleDateString('ko-KR', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  })
}

// sort 값에 따른 정렬 비교자 (date-desc 기본)
// source 정렬은 출처 라벨 우선, 동일하면 최신순
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
  // date-desc (기본): 최신순
  return String(b.created_at || '').localeCompare(String(a.created_at || ''))
}

// 썸네일(슬라이드 프레임)이 아직 없는 강의용 자리표시 아이콘
function ThumbnailPlaceholder() {
  return (
    <span className="lecture-card-thumb-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24">
        <rect x="2.5" y="5" width="14" height="14" rx="2.5" />
        <path d="M16.5 9.5 21 7v10l-4.5-2.5" />
      </svg>
    </span>
  )
}

// filters: { query, sort, sourceFilter } — 상위(강의 목록 화면)에서 주입
// onSelect(lectureId): 카드 클릭 시 상세 화면 이동
export default function LectureList({ onSelect, filters = {} }) {
  const { query = '', sort = 'date-desc', sourceFilter = 'all' } = filters
  const { data: lectures = [], isLoading, error } = useQuery({
    queryKey: ['lectures'],
    queryFn: () => listLectures(),
  })
  const [page, setPage] = useState(0)

  // 출처 필터 + 제목 검색 적용 후 정렬
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

  const pageCount = Math.max(1, Math.ceil(visible.length / PAGE_SIZE))

  // 검색어·정렬·필터 변경 시 첫 페이지로 복귀
  useEffect(() => {
    setPage(0)
  }, [query, sort, sourceFilter])

  // 필터 결과가 줄어 현재 페이지가 범위를 벗어나면 마지막 페이지로 당김
  useEffect(() => {
    setPage(current => Math.min(current, pageCount - 1))
  }, [pageCount])

  if (isLoading) return <p className="list-note">목록을 불러오는 중...</p>
  if (error) return <p className="error-text">목록 조회 실패: {String(error?.message || error)}</p>
  if (lectures.length === 0) return <p className="list-note">업로드된 강의가 없습니다.</p>
  if (visible.length === 0) return <p className="list-note">검색 결과가 없습니다.</p>

  const paged = visible.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE)

  return (
    <>
      <ul className="lecture-grid">
        {paged.map(lecture => (
          <li key={lecture.id}>
            <button type="button" className="lecture-card" onClick={() => onSelect(lecture.id)}>
              <span className="lecture-card-thumb">
                {lecture.thumbnail_url
                  ? <img className="lecture-card-thumb-img" src={lecture.thumbnail_url} alt="" />
                  : <ThumbnailPlaceholder />}
              </span>
              <span className="lecture-card-body">
                <span className="lecture-card-title">{lecture.title}</span>
                <span className="lecture-card-meta">
                  {lecture.source_tag && (
                    <span className="tag-chip">{lectureTagLabel(lecture.source_tag)}</span>
                  )}
                  <span className={`status-chip status-chip--${lecture.status}`}>
                    {STATUS_LABELS[lecture.status] || lecture.status}
                  </span>
                </span>
                {lecture.created_at && <span className="lecture-card-date">{formatDate(lecture.created_at)}</span>}
                {/* 진행 중 강의는 상태 칩('분석 중')으로 충분하므로 단계 텍스트 생략, 오류만 사유 표시 */}
                {lecture.status === 'error' && lecture.error_message && (
                  <span className="lecture-card-stage">{lecture.error_message}</span>
                )}
              </span>
            </button>
          </li>
        ))}
      </ul>

      {pageCount > 1 && (
        <div className="lecture-pagination">
          <button type="button" className="btn" onClick={() => setPage(p => p - 1)} disabled={page <= 0}>
            ← 이전
          </button>
          <span className="lecture-pagination-status">{page + 1} / {pageCount}</span>
          <button type="button" className="btn" onClick={() => setPage(p => p + 1)} disabled={page >= pageCount - 1}>
            다음 →
          </button>
        </div>
      )}
    </>
  )
}
