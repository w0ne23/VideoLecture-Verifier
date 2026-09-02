// 검증 파이프라인 단계 정의 + 진행 상태 유틸
// backend/app/worker.py 의 PIPELINE_STAGE_KEYS / LABELS 와 1:1 대응 —
// stageKey 가 백엔드와 어긋나면 해당 단계가 '대기'로 멈추므로 여기만 맞추면 됨

// 진행 표시용 단계 노드 목록 (순서 = 표시 순서)
// 영상 레인(슬라이드 추출→분석)과 오디오 레인(품질 분석→음성 전사)은 백엔드에서 서로 독립된
// 스레드로 진행되고(pipeline/orchestration/preprocess.py) 두 레인 다 끝나야 다음으로 넘어감
// 이 선형 목록은 순서대로 나열만 하고, DiagramPipeline 은 둘을 나란한 레인으로 그림
// 오디오 레인의 '음성 전사'는 전사 + 오디오 맥락 후처리(P3)까지 끝나야 done 으로 보고됨(별도 노드 없이 묶음)
export const PIPELINE_NODES = [
  { id: 'slide_extract', label: '슬라이드 추출', stageKey: 'preprocess_slide_extract', stageLabel: '슬라이드 추출' },
  { id: 'audio_quality', label: '오디오 품질 분석', stageKey: 'preprocess_audio_quality', stageLabel: '오디오 품질 분석' },
  { id: 'slide_analyze', label: '슬라이드 분석', stageKey: 'preprocess_slide_analyze', stageLabel: '슬라이드 분석' },
  { id: 'audio_transcribe', label: '음성 전사', stageKey: 'preprocess_audio_transcribe', stageLabel: '음성 전사' },
  { id: 'verifier_data', label: '검증 데이터 구성', stageKey: 'verifier_build_analyzer_input', stageLabel: '검증 입력 데이터 구성' },
  { id: 'claim_extraction', label: 'claim 추출', stageKey: 'verifier_claim_extraction', stageLabel: 'claim 추출' },
  { id: 'issue_judge', label: '오류 탐지', stageKey: 'verifier_issue_judge', stageLabel: '오류 탐지' },
  { id: 'issue_classification', label: '오류 유형 분류', stageKey: 'verifier_issue_classification', stageLabel: '오류 유형 분류' },
  // 웹 근거 수집으로 사실/시의성 오류 후보를 걸러내는 단계 (백엔드 실행 순서상 유형 분류 다음, 최종 판단 이전)
  { id: 'web_grounding', label: '오류 필터링', stageKey: 'verifier_web_grounding', stageLabel: '오류 필터링' },
  { id: 'final_verification', label: '오류 최종 판단', stageKey: 'verifier_final_verification', stageLabel: '오류 최종 판단' },
  // 슬라이드 검사/문법·코드 오류 점검도 독립 검사라 백엔드가 각자 진행 상태를 따로 보고
  // (pipeline/verifier/classified_slide_error_checker.py)
  { id: 'slide_inspect', label: '슬라이드 검사', stageKey: 'verify_slide_inspect', stageLabel: '슬라이드 검사' },
  { id: 'slide_syntax', label: '문법/코드 오류 점검', stageKey: 'verify_slide_syntax', stageLabel: '문법/코드 오류 점검' },
]

export const VERIFY_STAGE_KEYS = PIPELINE_NODES.map(node => node.stageKey)

// 화면 phase — 파이프라인 진행 중 / 검증 결과 준비됨 / 오류
export const PHASES = {
  PIPELINE: 'pipeline',
  VERIFY_READY: 'verifyReady',
  ERROR: 'error',
}

// 모든 단계를 wait 로 초기화한 배열
export function createEmptyStages() {
  return VERIFY_STAGE_KEYS.map(stage => ({ stage, status: 'wait', progress: null }))
}

// 임의의 스테이지 배열을 항상 전체 단계 + 고정 순서로 정규화 (알 수 없는 stage 는 무시)
export function normalizePipelineStages(stages = []) {
  if (!Array.isArray(stages) || stages.length === 0) return createEmptyStages()

  const byStage = new Map(createEmptyStages().map(item => [item.stage, item]))
  stages.forEach(item => {
    if (!item?.stage) return
    if (byStage.has(item.stage)) {
      byStage.set(item.stage, { stage: item.stage, status: item.status, progress: item.progress ?? null })
    }
  })
  return Array.from(byStage.values())
}
