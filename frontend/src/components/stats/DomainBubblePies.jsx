import { useEffect, useState } from 'react'
import { ISSUE_TYPES } from '../../data/mockStats'

function polar(cx, cy, r, angle) {
  const rad = ((angle - 90) * Math.PI) / 180
  return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)]
}

function arcPath(cx, cy, r, startAngle, endAngle) {
  if (endAngle - startAngle >= 359.9) {
    const [x1, y1] = polar(cx, cy, r, 0)
    const [x2, y2] = polar(cx, cy, r, 179.9)
    return [
      `M ${cx} ${cy}`,
      `L ${x1} ${y1}`,
      `A ${r} ${r} 0 1 1 ${x2} ${y2}`,
      `A ${r} ${r} 0 1 1 ${x1} ${y1}`,
      'Z',
    ].join(' ')
  }
  const [x1, y1] = polar(cx, cy, r, startAngle)
  const [x2, y2] = polar(cx, cy, r, endAngle)
  const large = endAngle - startAngle > 180 ? 1 : 0
  return `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z`
}

function slicesFor(typeDist, total) {
  let angle = 0
  return ISSUE_TYPES.map(type => {
    const value = typeDist[type.key] || 0
    if (value <= 0 || total <= 0) return null
    const sweep = (value / total) * 360
    const start = angle
    const end = angle + sweep
    angle = end
    return { ...type, value, start, end, mid: start + sweep / 2, sweep }
  }).filter(Boolean)
}

// 현재 도메인 하나를 중앙에 크게, 이전·다음 도메인은 옆에 반투명하게 보여주는 캐러셀.
// 화살표(이등변 삼각형) 버튼으로 도메인을 넘겨볼 수 있다.
function DomainPie({ row, radius, showLabels }) {
  const size = radius * 2 + 8
  const cx = size / 2
  const cy = size / 2
  const slices = slicesFor(row.typeDist, row.total)

  return (
    <figure className="stats-bubble">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <g className="stats-bubble-spin" style={{ transformOrigin: `${cx}px ${cy}px` }}>
          {slices.map(slice => (
            <path key={slice.key} d={arcPath(cx, cy, radius, slice.start, slice.end)} fill={slice.color}>
              <title>{`${row.label} · ${slice.label}: ${slice.value}`}</title>
            </path>
          ))}
        </g>
        {showLabels && slices.map(slice => {
          const [lx, ly] = polar(cx, cy, radius * 0.62, slice.mid)
          return (
            <text
              key={`${slice.key}-label`}
              className={`stats-data-label stats-data-label--on-bar stats-data-label--pie${slice.sweep < 24 ? ' stats-data-label--tight' : ''}`}
              x={lx}
              y={ly}
              textAnchor="middle"
              dominantBaseline="middle"
            >
              {slice.value}
            </text>
          )
        })}
      </svg>
      <figcaption>
        <strong>{row.label}</strong>
        <span>오류 {row.total}건</span>
      </figcaption>
    </figure>
  )
}

export default function DomainBubblePies({ rows, animKey, onIndexChange }) {
  const [index, setIndex] = useState(0)
  const count = rows.length
  const safeIndex = ((index % count) + count) % count

  const goPrev = () => setIndex(current => current - 1)
  const goNext = () => setIndex(current => current + 1)

  // 캐러셀이 넘어갈 때마다 지금 중앙에 있는 도메인을 부모(설명 패널)에도 알려서,
  // 그래프와 아래 설명이 항상 같은 도메인을 가리키게 한다.
  useEffect(() => {
    onIndexChange?.(safeIndex)
  }, [safeIndex, onIndexChange])

  const currentRow = rows[safeIndex]
  const prevRow = count > 1 ? rows[(safeIndex - 1 + count) % count] : null
  const nextRow = count > 1 ? rows[(safeIndex + 1) % count] : null

  return (
    <div key={animKey} className="stats-bubbles-carousel" role="img" aria-label="도메인별 이슈 원 그래프">
      <button
        type="button"
        className="stats-bubbles-nav stats-bubbles-nav--prev"
        onClick={goPrev}
        disabled={count <= 1}
        aria-label="이전 도메인"
      >
        <span className="stats-bubbles-nav-arrow" />
      </button>

      <div className="stats-bubbles-track">
        {prevRow && (
          <div className="stats-bubble-slot stats-bubble-slot--side">
            <DomainPie row={prevRow} radius={112} showLabels={false} />
          </div>
        )}
        <div className="stats-bubble-slot stats-bubble-slot--current">
          <DomainPie row={currentRow} radius={112} showLabels />
        </div>
        {nextRow && (
          <div className="stats-bubble-slot stats-bubble-slot--side">
            <DomainPie row={nextRow} radius={112} showLabels={false} />
          </div>
        )}
      </div>

      <button
        type="button"
        className="stats-bubbles-nav stats-bubbles-nav--next"
        onClick={goNext}
        disabled={count <= 1}
        aria-label="다음 도메인"
      >
        <span className="stats-bubbles-nav-arrow" />
      </button>
    </div>
  )
}
