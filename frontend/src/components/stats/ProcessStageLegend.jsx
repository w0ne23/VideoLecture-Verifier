// 파이프라인 단계(전처리/검증) 색상 범례 (강의 길이별 누적 막대용)

import { PROCESS_STAGES } from '../../config/statsConfig'

export default function ProcessStageLegend() {
  return (
    <ul className="stats-legend" aria-label="파이프라인 단계 색상">
      {PROCESS_STAGES.map(stage => (
        <li key={stage.key}>
          <span className="stats-legend-swatch" style={{ background: stage.color }} />
          {stage.label}
        </li>
      ))}
    </ul>
  )
}
