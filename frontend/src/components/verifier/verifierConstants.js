// backend/app/worker.py의 PIPELINE_STAGE_KEYS / LABELS와 1:1 대응.
// stage 키가 백엔드와 달라지면 진행 표시가 '대기'로 멈추므로 여기만 고치면 된다.
export const PIPELINE_NODES = [
  { id: 'data_extract', label: '데이터 추출', stageKey: 'preprocess_extract_media', stageLabel: '슬라이드 추출 및 오디오 품질 분석' },
  { id: 'content_extract', label: '텍스트화', stageKey: 'preprocess_textualize_transcribe', stageLabel: '슬라이드 텍스트화 및 전체 전사' },
  { id: 'context_analysis', label: '맥락 구성', stageKey: 'preprocess_enrich_audio_annotation', stageLabel: '오디오 맥락 후처리' },
  { id: 'verifier_data', label: '검증 데이터 구성', stageKey: 'verifier_build_analyzer_input', stageLabel: '검증 입력 데이터 구성' },
  { id: 'claim_extraction', label: '주장 추출', stageKey: 'verifier_claim_extraction', stageLabel: '주장 후보 추출' },
  { id: 'issue_judge', label: '이슈 후보 판단', stageKey: 'verifier_issue_judge', stageLabel: '이슈 후보 판단' },
  { id: 'issue_classification', label: '이슈 유형 분류', stageKey: 'verifier_issue_classification', stageLabel: '이슈 유형 분류' },
  { id: 'final_verification', label: '멀티 LLM 검증', stageKey: 'verifier_final_verification', stageLabel: '멀티 LLM 검증' },
  { id: 'web_grounding', label: '웹 근거 검증', stageKey: 'verifier_web_grounding', stageLabel: '웹 근거 검증' },
  { id: 'slide_review', label: '슬라이드 오류', stageKey: 'verify_slide_errors', stageLabel: '슬라이드 오류 검사' },
]

export const VERIFY_STAGE_KEYS = PIPELINE_NODES.map(node => node.stageKey)

export const PHASES = {
  PIPELINE: 'pipeline',
  VERIFY_READY: 'verifyReady',
  ERROR: 'error',
}

export function createEmptyStages() {
  return VERIFY_STAGE_KEYS.map(stage => ({ stage, status: 'wait' }))
}

export function normalizePipelineStages(stages = []) {
  if (!Array.isArray(stages) || stages.length === 0) return createEmptyStages()

  const byStage = new Map(createEmptyStages().map(item => [item.stage, item.status]))
  stages.forEach(item => {
    if (!item?.stage) return
    if (byStage.has(item.stage)) byStage.set(item.stage, item.status)
  })
  return Array.from(byStage, ([stage, status]) => ({ stage, status }))
}
