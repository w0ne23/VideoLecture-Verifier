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
