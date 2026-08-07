import { useCallback, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import LectureList from '../components/LectureList'
import { LECTURE_SOURCE_TAGS } from '../constants/lectureTags'

const SORT_OPTIONS = [
  { value: 'date-desc', label: '최신순' },
  { value: 'date-asc', label: '오래된순' },
  { value: 'source', label: '출처별' },
  { value: 'title', label: '제목순' },
]

export default function LecturesPage() {
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState('date-desc')
  const [sourceFilter, setSourceFilter] = useState('all')

  const openLecture = useCallback(id => navigate(`/verify/${id}`), [navigate])

  const filters = useMemo(() => ({ query, sort, sourceFilter }), [query, sort, sourceFilter])

  return (
    <section className="lectures-page">
      <div className="page-header-row">
        <h2 className="list-heading">강의 목록</h2>
        <button type="button" className="btn" onClick={() => navigate('/')}>
          메인으로
        </button>
      </div>

      <div className="lectures-toolbar">
        <input
          className="lectures-search"
          type="search"
          value={query}
          placeholder="제목 검색"
          aria-label="강의 제목 검색"
          onChange={event => setQuery(event.target.value)}
        />
        <select
          className="lectures-select"
          value={sort}
          aria-label="정렬"
          onChange={event => setSort(event.target.value)}
        >
          {SORT_OPTIONS.map(option => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </select>
        <select
          className="lectures-select"
          value={sourceFilter}
          aria-label="출처 필터"
          onChange={event => setSourceFilter(event.target.value)}
        >
          <option value="all">전체 출처</option>
          {LECTURE_SOURCE_TAGS.map(tag => (
            <option key={tag.value} value={tag.value}>{tag.label}</option>
          ))}
        </select>
      </div>

      <LectureList onSelect={openLecture} filters={filters} />
    </section>
  )
}
