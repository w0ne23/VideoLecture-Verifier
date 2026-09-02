// 통계 화면 — 3개 보기(도메인별/출처별/강의 길이별) 탭 + 차트 + 인사이트 카드
// 데이터: GET /api/stats, 검증 완료 시 useJobStream 이 ['stats'] 캐시를 무효화해 자동 갱신

import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import DomainBubblePies from '../components/stats/DomainBubblePies'
import GroupedBarChart from '../components/stats/GroupedBarChart'
import IssueTypeLegend from '../components/stats/IssueTypeLegend'
import ProcessStageLegend from '../components/stats/ProcessStageLegend'
import StackedBarChart from '../components/stats/StackedBarChart'
import StatsInsight from '../components/stats/StatsInsight'
import { getStats } from '../api/stats'
import { DOMAIN_LABELS, SOURCE_TAG_LABELS, buildDomainInsight, buildInsight } from '../config/statsConfig'

const VIEWS = [
  { id: 'domain', label: '도메인별' },
  { id: 'tag', label: '출처별' },
  { id: 'duration', label: '강의 길이별' },
]

const EMPTY_STATS = { lecture_count: 0, by_tag: [], by_domain: [], by_duration: [] }

// API 행에 표시용 label 부착 (by_duration 은 이미 label 보유)
function decorate(view, rows) {
  if (view === 'duration') return rows
  const labels = view === 'tag' ? SOURCE_TAG_LABELS : DOMAIN_LABELS
  return rows.map(row => ({ ...row, label: labels[row.key] || row.key }))
}

export default function StatsPage() {
  const [view, setView] = useState('domain')
  const [animKey, setAnimKey] = useState(0)
  const [domainIndex, setDomainIndex] = useState(0)

  const { data = EMPTY_STATS, isLoading, error } = useQuery({
    queryKey: ['stats'],
    queryFn: getStats,
  })

  const rows = useMemo(() => {
    const raw = view === 'duration' ? data.by_duration : view === 'tag' ? data.by_tag : data.by_domain
    return decorate(view, raw || [])
  }, [view, data])

  const insight = useMemo(
    () => (view === 'domain' ? buildDomainInsight(rows[domainIndex]) : buildInsight(view, rows)),
    [view, rows, domainIndex],
  )

  // 보기 전환 시 animKey 를 올려 차트 재생, 도메인 캐러셀 인덱스 초기화
  function selectView(next) {
    setView(next)
    setAnimKey(key => key + 1)
    setDomainIndex(0)
  }

  const isEmpty = !isLoading && !error && data.lecture_count === 0

  return (
    <div className="stats-page">
      <div className="stats-page-head-row">
        <div className="stats-page-head">
          <h2>통계</h2>
        </div>

        <div className="stats-tabs" role="tablist" aria-label="통계 보기">
          {VIEWS.map(item => (
            <button
              key={item.id}
              type="button"
              role="tab"
              aria-selected={view === item.id}
              className={`stats-tab${view === item.id ? ' is-active' : ''}`}
              onClick={() => selectView(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {isLoading && <p className="stats-note">불러오는 중…</p>}
      {error && <p className="stats-note">통계를 불러오지 못했습니다: {error.message}</p>}
      {isEmpty && (
        <p className="stats-note">
          검증이 완료된 강의가 아직 없습니다. 강의를 업로드해 검증하면 여기에 집계됩니다.
        </p>
      )}

      {!isLoading && !error && !isEmpty && rows.length === 0 && (
        <p className="stats-note">이 보기에 표시할 데이터가 없습니다.</p>
      )}

      {!isLoading && !error && !isEmpty && rows.length > 0 && (
        <div className="stats-layout">
          <section className="stats-chart-panel">
            {view === 'duration' ? <ProcessStageLegend /> : <IssueTypeLegend />}
            {view === 'tag' && <GroupedBarChart rows={rows} animKey={animKey} />}
            {view === 'duration' && <StackedBarChart rows={rows} animKey={animKey} />}
            {view === 'domain' && (
              <DomainBubblePies rows={rows} animKey={animKey} onIndexChange={setDomainIndex} />
            )}
          </section>
          <StatsInsight title={insight.title} bullets={insight.bullets} />
        </div>
      )}
    </div>
  )
}
