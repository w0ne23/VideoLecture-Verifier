import { ISSUE_TYPES } from '../../data/mockStats'

const W = 720
const H = 380
const PAD = { top: 36, right: 16, bottom: 56, left: 44 }

export default function GroupedBarChart({ rows, animKey }) {
  const maxValue = Math.max(
    1,
    ...rows.flatMap(row => ISSUE_TYPES.map(t => row.typeDist[t.key] || 0)),
  )
  const plotW = W - PAD.left - PAD.right
  const plotH = H - PAD.top - PAD.bottom
  const groupW = plotW / rows.length
  const barGap = 2
  const barW = Math.min(16, (groupW - 16) / ISSUE_TYPES.length - barGap)
  const ticks = [0, 0.25, 0.5, 0.75, 1].map(t => Math.round(maxValue * t))

  return (
    <svg
      key={animKey}
      className="stats-svg stats-svg--bars"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="태그별 이슈 유형 묶음 막대 그래프"
    >
      {ticks.map(tick => {
        const y = PAD.top + plotH - (tick / maxValue) * plotH
        return (
          <g key={tick}>
            <line className="stats-grid" x1={PAD.left} x2={W - PAD.right} y1={y} y2={y} />
            <text className="stats-axis" x={PAD.left - 8} y={y + 4} textAnchor="end">{tick}</text>
          </g>
        )
      })}

      {rows.map((row, groupIndex) => {
        const groupX = PAD.left + groupIndex * groupW + groupW / 2
        const startX = groupX - (ISSUE_TYPES.length * (barW + barGap)) / 2
        return (
          <g key={row.key}>
            {ISSUE_TYPES.map((type, typeIndex) => {
              const value = row.typeDist[type.key] || 0
              if (value <= 0) return null
              const h = (value / maxValue) * plotH
              const x = startX + typeIndex * (barW + barGap)
              const y = PAD.top + plotH - h
              const delay = `${groupIndex * 60 + typeIndex * 40}ms`
              return (
                <g key={type.key}>
                  <rect
                    className="stats-grow-bar"
                    x={x}
                    y={y}
                    width={barW}
                    height={h}
                    rx={3}
                    fill={type.color}
                    style={{ animationDelay: delay }}
                  >
                    <title>{`${row.label} · ${type.label}: ${value}`}</title>
                  </rect>
                  <text
                    className="stats-data-label"
                    x={x + barW / 2}
                    y={y - 5}
                    textAnchor="middle"
                    style={{ animationDelay: delay }}
                  >
                    {value}
                  </text>
                </g>
              )
            })}
            <text className="stats-axis stats-axis--x" x={groupX} y={H - 28} textAnchor="middle">
              {row.label}
            </text>
            <text className="stats-axis-sub" x={groupX} y={H - 12} textAnchor="middle">
              합계 {row.total}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
