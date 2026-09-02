// 도메인별 오류 유형 원 그래프 캐러셀 — 현재 도메인은 중앙에 크게, 이전/다음은 옆에 반투명

import { useEffect, useState } from 'react'
import { ISSUE_TYPES } from '../../config/statsConfig'

// 중심 (cx,cy) 기준 각도(deg, 12시 방향이 0)의 원주 좌표
function polar(cx, cy, r, angle) {
  const rad = ((angle - 90) * Math.PI) / 180
  return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)]
}

// 파이 조각 하나의 SVG path — 거의 원(360도)이면 두 개의 반원 arc 로 그림
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

// typeDist → 파이 조각 배열 (0건 유형 제외, 각도 누적)
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

// 파이 하나 — showLabels 면 각 조각 안에 값 라벨 표시
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

// rows: 도메인별 행 배열, onIndexChange: 중앙 도메인 인덱스를 부모에 통지
export default function DomainBubblePies({ rows, animKey, onIndexChange }) {
  const [index, setIndex] = useState(0)
  const count = rows.length
  // 양 끝에서 멈춤(순환 X) — 도메인 2개일 때 순환하면 양옆에 같은 게 보여 "같은 게 두 개"처럼 오해됨
  const safeIndex = Math.min(Math.max(index, 0), count - 1)

  const goPrev = () => setIndex(current => Math.max(0, current - 1))
  const goNext = () => setIndex(current => Math.min(count - 1, current + 1))

  // 넘길 때마다 중앙 도메인을 부모(설명 패널)에도 알려 그래프와 설명이 같은 도메인을 가리키게 함
  useEffect(() => {
    onIndexChange?.(safeIndex)
  }, [safeIndex, onIndexChange])

  const currentRow = rows[safeIndex]
  const prevRow = safeIndex > 0 ? rows[safeIndex - 1] : null
  const nextRow = safeIndex < count - 1 ? rows[safeIndex + 1] : null

  return (
    <div key={animKey} className="stats-bubbles-carousel" role="img" aria-label="도메인별 이슈 원 그래프">
      <button
        type="button"
        className="stats-bubbles-nav stats-bubbles-nav--prev"
        onClick={goPrev}
        disabled={safeIndex <= 0}
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
        disabled={safeIndex >= count - 1}
        aria-label="다음 도메인"
      >
        <span className="stats-bubbles-nav-arrow" />
      </button>
    </div>
  )
}
