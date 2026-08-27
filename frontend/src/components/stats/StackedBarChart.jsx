import { PROCESS_STAGES } from '../../data/mockStats'

const W = 720
const PAD = { top: 20, right: 24, bottom: 20, left: 108 }
const ROW_H = 64

export default function StackedBarChart({ rows, animKey }) {
  const maxTotal = Math.max(1, ...rows.map(row => row.total))
  const plotW = W - PAD.left - PAD.right - 110
  const height = PAD.top + PAD.bottom + rows.length * ROW_H

  return (
    <svg
      key={animKey}
      className="stats-svg stats-svg--stack"
      viewBox={`0 0 ${W} ${height}`}
      role="img"
      aria-label="강의 길이별 파이프라인 소요 시간 막대 그래프"
    >
      {rows.map((row, index) => {
        const y = PAD.top + index * ROW_H + 10
        const barH = 32
        let cursor = PAD.left
        const segments = PROCESS_STAGES.map(stage => {
          const value = row[`${stage.key}Min`] || 0
          const w = (value / maxTotal) * plotW
          const x = cursor
          cursor += w
          return { ...stage, value, w, x }
        }).filter(seg => seg.value > 0)

        return (
          <g key={row.key}>
            <text
              className="stats-axis stats-axis--y"
              x={PAD.left - 12}
              y={y + barH / 2}
              textAnchor="end"
              dominantBaseline="middle"
            >
              {row.label}
            </text>
            <g
              className="stats-stack-grow-x"
              style={{ animationDelay: `${index * 90}ms` }}
            >
              {segments.map(seg => (
                <rect
                  key={seg.key}
                  x={seg.x}
                  y={y}
                  width={Math.max(seg.w, 0)}
                  height={barH}
                  fill={seg.color}
                >
                  <title>{`${row.label} · ${seg.label}: ${seg.value}분`}</title>
                </rect>
              ))}
            </g>
            {segments.map(seg => (
              <text
                key={`${seg.key}-label`}
                className={`stats-data-label stats-data-label--on-bar${seg.w < 22 ? ' stats-data-label--tight' : ''}`}
                x={seg.x + seg.w / 2}
                y={y + barH / 2 + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                style={{ animationDelay: `${220 + index * 90}ms` }}
              >
                {seg.value}
              </text>
            ))}
            <text
              className="stats-side-label"
              x={cursor + 12}
              y={y + barH / 2 + 1}
              dominantBaseline="middle"
              style={{ animationDelay: `${200 + index * 90}ms` }}
            >
              총 소요 평균 {row.total}분
            </text>
          </g>
        )
      })}
    </svg>
  )
}
