import { useMemo, useState } from 'react'
import BeforeAfterCompare from '../components/stats/BeforeAfterCompare'
import DomainBubblePies from '../components/stats/DomainBubblePies'
import GroupedBarChart from '../components/stats/GroupedBarChart'
import IssueTypeLegend from '../components/stats/IssueTypeLegend'
import StackedBarChart from '../components/stats/StackedBarChart'
import StatsInsight from '../components/stats/StatsInsight'
import {
  MOCK_BEFORE_AFTER_PAIRS,
  MOCK_BY_DOMAIN,
  MOCK_BY_DURATION,
  MOCK_BY_TAG,
  buildCompareInsight,
  buildInsight,
} from '../data/mockStats'

const VIEWS = [
  {
    id: 'tag',
    label: '출처별',
  },
  {
    id: 'duration',
    label: '강의 길이별',
  },
  {
    id: 'domain',
    label: '도메인별',
  },
  {
    id: 'compare',
    label: '수정 전후',
  },
]

function rowsFor(view) {
  if (view === 'duration') return MOCK_BY_DURATION
  if (view === 'domain') return MOCK_BY_DOMAIN
  return MOCK_BY_TAG
}

export default function StatsPage() {
  const [view, setView] = useState('tag')
  const [animKey, setAnimKey] = useState(0)
  const [pairId, setPairId] = useState(MOCK_BEFORE_AFTER_PAIRS[0]?.id)
  const rows = useMemo(() => rowsFor(view), [view])
  const pair = useMemo(
    () => MOCK_BEFORE_AFTER_PAIRS.find(item => item.id === pairId) || MOCK_BEFORE_AFTER_PAIRS[0],
    [pairId],
  )
  const insight = useMemo(
    () => (view === 'compare' ? buildCompareInsight(pair) : buildInsight(view, rows)),
    [view, rows, pair],
  )

  function selectView(next) {
    setView(next)
    setAnimKey(key => key + 1)
  }

  function changePair(nextId) {
    setPairId(nextId)
    setAnimKey(key => key + 1)
  }

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

      <p className="stats-note">{VIEWS.find(item => item.id === view)?.blurb}</p>

      <div className="stats-layout">
        <section className={`stats-chart-panel${view === 'compare' ? ' stats-chart-panel--compare' : ''}`}>
          {view !== 'compare' && <IssueTypeLegend />}
          {view === 'tag' && <GroupedBarChart rows={rows} animKey={animKey} />}
          {view === 'duration' && <StackedBarChart rows={rows} animKey={animKey} />}
          {view === 'domain' && <DomainBubblePies rows={rows} animKey={animKey} />}
          {view === 'compare' && (
            <BeforeAfterCompare
              pairs={MOCK_BEFORE_AFTER_PAIRS}
              pairId={pair?.id}
              onPairChange={changePair}
              animKey={animKey}
            />
          )}
        </section>
        <StatsInsight title={insight.title} bullets={insight.bullets} />
      </div>
    </div>
  )
}
