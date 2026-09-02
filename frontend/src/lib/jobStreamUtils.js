// job 스트림 상태값을 화면 표현으로 변환하는 순수 함수 모음 (useJobStream 에서 사용)

import { PHASES, VERIFY_STAGE_KEYS, normalizePipelineStages } from '../components/verifier/verifierConstants'

// job 상태 → 화면 phase
// VLVerifier 백엔드는 verify 워크플로우만 지원하므로 verify 기준 매핑만 유지
export function phaseFromStatus(status) {
  if (status === 'error') return PHASES.ERROR
  if (status === 'done' || status === 'waiting_approval') return PHASES.VERIFY_READY
  return PHASES.PIPELINE
}

// 더 이상 진행 이벤트가 오지 않는 종료 상태 여부
export function isTerminalStatus(status) {
  return ['done', 'error', 'waiting_approval', 'rejected'].includes(status)
}

// 기존 스테이지 배열에 새 이벤트를 병합 — 알려진 스테이지 키만 반영
export function mergeStageStatus(current, incoming) {
  const known = new Set(VERIFY_STAGE_KEYS)
  const byStage = new Map(current.map(item => [item.stage, item]))

  incoming.forEach(item => {
    if (!item?.stage || !known.has(item.stage)) return
    byStage.set(item.stage, { stage: item.stage, status: item.status, progress: item.progress ?? null })
  })
  return normalizePipelineStages(Array.from(byStage.values()))
}

// 모든 스테이지를 같은 상태로 세팅 (예: 완료 시 전체 done 처리)
export function markAllStages(status) {
  return normalizePipelineStages(VERIFY_STAGE_KEYS.map(stage => ({ stage, status })))
}
