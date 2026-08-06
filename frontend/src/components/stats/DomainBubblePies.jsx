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

export default function DomainBubblePies({ rows, animKey }) {
  const maxTotal = Math.max(1, ...rows.map(row => row.total))
  const minR = 52
  const maxR = 96

  return (
    <div key={animKey} className="stats-bubbles" role="img" aria-label="도메인별 이슈 원 그래프">
      {rows.map((row, index) => {
        const radius = minR + (row.total / maxTotal) * (maxR - minR)
        const size = radius * 2 + 8
        const cx = size / 2
        const cy = size / 2
        const slices = slicesFor(row.typeDist, row.total)

        return (
          <figure
            key={row.key}
            className="stats-bubble"
            style={{ animationDelay: `${index * 100}ms` }}
          >
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
              <g className="stats-bubble-spin" style={{ transformOrigin: `${cx}px ${cy}px` }}>
                {slices.map(slice => (
                  <path
                    key={slice.key}
                    d={arcPath(cx, cy, radius, slice.start, slice.end)}
                    fill={slice.color}
                  >
                    <title>{`${row.label} · ${slice.label}: ${slice.value}`}</title>
                  </path>
                ))}
              </g>
              {slices.map(slice => {
                // 모든 숫자는 원 안쪽 동일 비율 위치에 두어 밸런스를 맞춤
                const [lx, ly] = polar(cx, cy, radius * 0.62, slice.mid)
                return (
                  <text
                    key={`${slice.key}-label`}
                    className={`stats-data-label stats-data-label--on-bar${slice.sweep < 24 ? ' stats-data-label--tight' : ''}`}
                    x={lx}
                    y={ly}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    style={{ animationDelay: `${180 + index * 100}ms` }}
                  >
                    {slice.value}
                  </text>
                )
              })}
            </svg>
            <figcaption>
              <strong>{row.label}</strong>
              <span>Issue {row.total}건</span>
            </figcaption>
          </figure>
        )
      })}
    </div>
  )
}
