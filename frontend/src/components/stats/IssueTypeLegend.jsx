// 지식 오류 유형 색상 범례 (묶음 막대·원 그래프 공용)

import { ISSUE_TYPES } from '../../config/statsConfig'

export default function IssueTypeLegend() {
  return (
    <ul className="stats-legend" aria-label="이슈 유형 색상">
      {ISSUE_TYPES.map(type => (
        <li key={type.key}>
          <span className="stats-legend-swatch" style={{ background: type.color }} />
          {type.label}
        </li>
      ))}
    </ul>
  )
}
