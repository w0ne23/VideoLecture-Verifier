// backend/app/worker.py의 PIPELINE_STAGE_KEYS / LABELS와 1:1 대응.
// stage 키가 백엔드와 달라지면 진행 표시가 '대기'로 멈추므로 여기만 고치면 된다.
export const PIPELINE_NODES = [
  // 영상 레인(슬라이드 추출→분석)과 오디오 레인(품질 분석→음성 전사)은 백엔드에서 실제로
  // 서로 독립된 스레드로 진행되고(pipeline/orchestration/preprocess.py), 두 레인 다
  // 끝나야 다음 단계로 넘어간다. 이 선형 목록에서는 그냥 순서대로 나열하지만,
  // DiagramPipeline은 이 둘을 나란한 두 레인으로 그린다. 오디오 레인의 음성 전사
  // (preprocess_audio_transcribe)는 전사 자체뿐 아니라 그 뒤 오디오 맥락 후처리(P3)까지
  // 끝나야 done으로 보고된다 — 별도 노드를 두지 않고 음성 전사에 묶었다.
  { id: 'slide_extract', label: '슬라이드 추출', stageKey: 'preprocess_slide_extract', stageLabel: '슬라이드 추출' },
  { id: 'audio_quality', label: '오디오 품질 분석', stageKey: 'preprocess_audio_quality', stageLabel: '오디오 품질 분석' },
  { id: 'slide_analyze', label: '슬라이드 분석', stageKey: 'preprocess_slide_analyze', stageLabel: '슬라이드 분석' },
  { id: 'audio_transcribe', label: '음성 전사', stageKey: 'preprocess_audio_transcribe', stageLabel: '음성 전사' },
  { id: 'verifier_data', label: '검증 데이터 구성', stageKey: 'verifier_build_analyzer_input', stageLabel: '검증 입력 데이터 구성' },
  { id: 'claim_extraction', label: '주장 추출', stageKey: 'verifier_claim_extraction', stageLabel: '주장 후보 추출' },
  { id: 'issue_judge', label: '이슈 후보 판단', stageKey: 'verifier_issue_judge', stageLabel: '이슈 후보 판단' },
  { id: 'issue_classification', label: '이슈 유형 분류', stageKey: 'verifier_issue_classification', stageLabel: '이슈 유형 분류' },
  { id: 'final_verification', label: '멀티 LLM 검증', stageKey: 'verifier_final_verification', stageLabel: '멀티 LLM 검증' },
  { id: 'web_grounding', label: '웹 근거 검증', stageKey: 'verifier_web_grounding', stageLabel: '웹 근거 검증' },
  // 슬라이드 검사/문법·코드 오류 점검도 서로 독립된 검사라 백엔드가 각자 진행 상태를
  // 따로 보고한다(pipeline/verifier/classified_slide_error_checker.py).
  { id: 'slide_inspect', label: '슬라이드 검사', stageKey: 'verify_slide_inspect', stageLabel: '슬라이드 검사' },
  { id: 'slide_syntax', label: '문법/코드 오류 점검', stageKey: 'verify_slide_syntax', stageLabel: '문법/코드 오류 점검' },
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
