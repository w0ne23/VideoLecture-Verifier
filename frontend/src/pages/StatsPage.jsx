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
    label: '태그별',
    blurb: '출처 태그마다 이슈 유형을 나란히 비교합니다.',
  },
  {
    id: 'duration',
    label: '강의 길이별',
    blurb: '길이 구간별 Issue 구성과 평균 소요시간을 봅니다.',
  },
  {
    id: 'domain',
    label: '도메인별',
    blurb: '원 크기는 Issue 수, 조각 색은 유형입니다.',
  },
  {
    id: 'compare',
    label: '수정 전후',
    blurb: '동일 강의의 수정 전·후 검증 결과를 비교합니다. (목업)',
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
      <div className="stats-page-head">
        <h2>통계</h2>
        <p>확정·교수확인 Issue만 집계합니다. 기각과 슬라이드 오류는 제외합니다.</p>
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

      <p className="stats-note">{VIEWS.find(item => item.id === view)?.blurb}</p>
      <IssueTypeLegend />

      <div className="stats-layout">
        <section className={`stats-chart-panel${view === 'compare' ? ' stats-chart-panel--compare' : ''}`}>
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
