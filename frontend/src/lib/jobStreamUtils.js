import { PHASES, VERIFY_STAGE_KEYS, normalizePipelineStages } from '../components/verifier/verifierConstants'

// VeriLec 백엔드는 verify 워크플로우만 지원: 상태 → 화면 phase 매핑도 verify 기준만 남긴다.
export function phaseFromStatus(status) {
  if (status === 'error') return PHASES.ERROR
  if (status === 'done' || status === 'waiting_approval') return PHASES.VERIFY_READY
  return PHASES.PIPELINE
}

export function isTerminalStatus(status) {
  return ['done', 'error', 'waiting_approval', 'rejected'].includes(status)
}

export function mergeStageStatus(current, incoming) {
  const known = new Set(VERIFY_STAGE_KEYS)
  const byStage = new Map(current.map(item => [item.stage, item.status]))

  incoming.forEach(item => {
    if (!item?.stage || !known.has(item.stage)) return
    byStage.set(item.stage, item.status)
  })
  return normalizePipelineStages(Array.from(byStage, ([stage, status]) => ({ stage, status })))
}

export function markAllStages(status) {
  return normalizePipelineStages(VERIFY_STAGE_KEYS.map(stage => ({ stage, status })))
}
