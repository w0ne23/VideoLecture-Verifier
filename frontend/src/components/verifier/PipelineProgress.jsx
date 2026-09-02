import { PIPELINE_NODES } from './verifierConstants'

// 검증 파이프라인 단계를 한 줄로 표시하는 진행 바 (노드 + 커넥터)

// stages 배열에서 특정 stageKey 의 상태 추출 (없으면 wait)
function getStageStatus(stages, key) {
  return stages.find(s => s.stage === key)?.status ?? 'wait'
}

// 노드 상태 → 접근성 라벨용 한글
function getNodeStatusText(status) {
  if (status === 'done') return '완료'
  if (status === 'run') return '현재 단계'
  if (status === 'error') return '실패'
  return '대기'
}

// stages: [{ stage, status }] — PIPELINE_NODES 순서대로 렌더
export default function PipelineProgress({ stages, statusMessage }) {
  const nodes = PIPELINE_NODES.map(node => ({
    ...node,
    status: getStageStatus(stages, node.stageKey),
  }))
  const activeIndex = nodes.findIndex(node => node.status === 'run')

  return (
    <div className="vf-pipe">
      <div className="vf-progress-message">{String(statusMessage || '').trim() || '분석 준비 중...'}</div>
      <div className="vf-flow" role="list" aria-label="파이프라인 단계">
        {nodes.map((node, index) => {
          // done/run 노드는 앞 노드와 잇는 커넥터를 채움
          const connectorOn = index > 0 && (node.status === 'done' || node.status === 'run')
          return (
            <div
              key={node.id}
              className={`vf-flow-item vf-flow-item--${node.status}${index === activeIndex ? ' vf-flow-item--active' : ''}`}
              role="listitem"
              aria-current={node.status === 'run' ? 'step' : undefined}
              aria-label={`${index + 1}단계 ${node.label}: ${getNodeStatusText(node.status)}`}
              title={node.stageLabel}
            >
              {index > 0 && <span className={`vf-flow-connector${connectorOn ? ' vf-flow-connector--on' : ''}`} aria-hidden="true" />}
              <span className="vf-flow-node" aria-hidden="true" />
              <div className="vf-flow-label">{node.label}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
