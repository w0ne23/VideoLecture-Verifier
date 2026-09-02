// 다이어그램 파이프라인 노드 정의 + 데모용 스케줄 + SSE 상태 매핑
// 아키텍처 다이어그램(전처리 → 검증, 각 구간이 두 갈래로 나뉘었다 합쳐짐)을 그대로 옮긴 것
// 구조 설명용 라벨이며, 실제 백엔드 파이프라인(verifierConstants.js 의 PIPELINE_NODES)과는 별개

// 다이어그램 노드 — lane 이 없는 노드(video/integrated_text/error_output)는 합류·분기 지점
export const NODES = [
  { id: 'video', label: '강의 영상', icon: 'video' },
  { id: 'slide_extract', label: '슬라이드 추출', lane: 'video' },
  { id: 'slide_analyze', label: '슬라이드 분석', lane: 'video' },
  { id: 'audio_quality', label: '오디오 품질 분석', lane: 'audio' },
  { id: 'voice_transcribe', label: '음성 전사', lane: 'audio' },
  { id: 'integrated_text', label: '멀티모달 통합 텍스트', icon: 'text' },
  { id: 'claim_extract', label: 'claim 추출', lane: 'utterance' },
  { id: 'issue_detect', label: '오류 탐지', lane: 'utterance' },
  { id: 'issue_classify', label: '오류 분류', lane: 'utterance' },
  { id: 'issue_filter', label: '오류 필터링', lane: 'utterance' },
  { id: 'issue_judge', label: '오류 판단', lane: 'utterance' },
  { id: 'slide_inspect', label: '슬라이드 검사', lane: 'slide' },
  { id: 'syntax_verify', label: '문법/코드 오류 점검', lane: 'slide' },
  { id: 'error_output', label: '피드백', icon: 'stack' },
]

export const NODE_BY_ID = Object.fromEntries(NODES.map(node => [node.id, node]))
export const NODE_IDS = NODES.map(node => node.id)

// 틱 인덱스 기반 데모(useDemoDiagramFlow) 전용 — 한 틱에 동시 활성화되는 노드 묶음
// 영상/오디오, 발화검증/슬라이드검증처럼 병렬 구간은 같은 틱에 묶음
// 슬라이드검증(2단계)이 먼저 끝나면 이후 틱은 발화검증만 남아 단독 진행
// video 는 화면 진입 전(업로드 시점)에 끝난 단계라 별도 틱 없이 항상 done
export const TICKS = [
  ['slide_extract', 'audio_quality'],
  ['slide_analyze', 'voice_transcribe'],
  ['integrated_text'],
  ['claim_extract', 'slide_inspect'],
  ['issue_detect', 'syntax_verify'],
  ['issue_classify'],
  ['issue_filter'],
  ['issue_judge'],
  ['error_output'],
]

// 시간 기반 데모(useTimedDiagramFlow) 전용 노드별 소요 시간 — 이 숫자만 바꾸면
// 아래 NODE_SCHEDULE 의 시작·종료 시각이 자동 재계산됨
// - 전처리: 영상 레인(추출→분석)과 오디오 레인(품질→전사)의 총합을 맞춰 같은 시점에 합류시키되,
//   앞 단계에서 영상이 오래 걸리면 뒤 단계는 오디오가 오래 걸리도록 배분을 반대로 뒤집음
// - 검증: 슬라이드 검증(2단계)이 발화 검증(5단계)보다 훨씬 먼저 끝나도록 각 단계도 짧게 설정
const NODE_DURATIONS_MS = {
  slide_extract: 95_000,
  slide_analyze: 75_000,
  audio_quality: 65_000,
  voice_transcribe: 105_000,
  integrated_text: 30_000,
  claim_extract: 100_000,
  issue_detect: 110_000,
  issue_classify: 15_000,
  issue_filter: 15_000,
  issue_judge: 15_000,
  slide_inspect: 60_000,
  syntax_verify: 55_000,
  error_output: 5_000,
}

// startMs 부터 ids 를 순서대로 이어 붙인 레인 하나의 일정 생성 — { entries, endMs } 반환
function laneSchedule(startMs, ids) {
  let cursor = startMs
  const entries = {}
  for (const id of ids) {
    const duration = NODE_DURATIONS_MS[id]
    entries[id] = { start: cursor, duration }
    cursor += duration
  }
  return { entries, endMs: cursor }
}

// 레인별 일정 조립 — 전처리 두 레인은 0 에서 시작해 늦게 끝나는 쪽에 맞춰 통합 텍스트로 합류,
// 검증 두 레인은 통합 텍스트 종료 시점에 함께 시작
const videoLane = laneSchedule(0, ['slide_extract', 'slide_analyze'])
const audioLane = laneSchedule(0, ['audio_quality', 'voice_transcribe'])
const integratedLane = laneSchedule(Math.max(videoLane.endMs, audioLane.endMs), ['integrated_text'])
const verifyStartMs = integratedLane.endMs
const utteranceLane = laneSchedule(verifyStartMs, ['claim_extract', 'issue_detect', 'issue_classify', 'issue_filter', 'issue_judge'])
const slideLane = laneSchedule(verifyStartMs, ['slide_inspect', 'syntax_verify'])
const feedbackLane = laneSchedule(Math.max(utteranceLane.endMs, slideLane.endMs), ['error_output'])

// 노드 id → { start, duration }(ms). 상태(wait/run/done)는 elapsedMs 와 이 스케줄만 비교해 계산
export const NODE_SCHEDULE = {
  ...videoLane.entries,
  ...audioLane.entries,
  ...integratedLane.entries,
  ...utteranceLane.entries,
  ...slideLane.entries,
  ...feedbackLane.entries,
}

export const TOTAL_DURATION_MS = feedbackLane.endMs

// "멀티모달 강의 영상 분석(전처리)" / "지식 오류 탐지(검증)" 큰 분류 — 진행 텍스트에 표시
const PRE_PHASE_IDS = new Set(['slide_extract', 'slide_analyze', 'audio_quality', 'voice_transcribe', 'integrated_text'])

// 노드 id → 소속 대분류 문구
export function bigPhaseFor(id) {
  return PRE_PHASE_IDS.has(id) ? '멀티모달 강의 영상 분석' : '지식 오류 탐지'
}

// ms → mm:ss
export function formatDuration(ms) {
  const totalSeconds = Math.max(0, Math.round(ms / 1000))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

// 진행 바 색을 다이어그램 레인 색과 맞추기 위한 매핑 — 전처리(영상/오디오)는 pre,
// 검증은 발화(utterance)·슬라이드(slide) 로 diag-lane-capsule 과 동일 톤
const LANE_TONE = { video: 'pre', audio: 'pre', utterance: 'utterance', slide: 'slide' }

// 노드 id → 진행 바 톤 키
export function laneToneFor(id) {
  return LANE_TONE[NODE_BY_ID[id]?.lane] || 'pre'
}

// 백엔드 SSE stageKey → 다이어그램 노드 id
// 백엔드가 각 stage 를 다이어그램 노드 하나에 정확히 대응하도록 나눠 보고하므로 1:1 로 매핑
// (error_output 만 예외 — 아래 statusMapFromPipelineStages 참고)
const STAGE_KEY_TO_NODE_ID = {
  preprocess_slide_extract: 'slide_extract',
  preprocess_audio_quality: 'audio_quality',
  preprocess_slide_analyze: 'slide_analyze',
  preprocess_audio_transcribe: 'voice_transcribe',
  verifier_build_analyzer_input: 'integrated_text',
  verifier_claim_extraction: 'claim_extract',
  verifier_issue_judge: 'issue_detect',
  verifier_issue_classification: 'issue_classify',
  verifier_web_grounding: 'issue_filter',
  verifier_final_verification: 'issue_judge',
  verify_slide_inspect: 'slide_inspect',
  verify_slide_syntax: 'syntax_verify',
}

// 실제 검증 화면(VerifyProgressPage)이 SSE 로 받는 pipelineStages([{stage, status}])를
// 다이어그램 노드별 status 맵으로 변환
export function statusMapFromPipelineStages(pipelineStages = []) {
  const status = Object.fromEntries(NODE_IDS.map(id => [id, 'wait']))
  // video 는 화면 진입 전(업로드 시점)에 끝난 단계라 항상 done 고정
  status.video = 'done'

  for (const item of pipelineStages) {
    const nodeId = STAGE_KEY_TO_NODE_ID[item?.stage]
    if (nodeId) status[nodeId] = item.status
  }

  // error_output 전용 stage 는 백엔드에 없음 — 발화 체인(issue_judge)과 슬라이드 체인(syntax_verify)이
  // 둘 다 done 이어야 done 으로 간주
  status.error_output = status.issue_judge === 'done' && status.syntax_verify === 'done' ? 'done' : 'wait'

  return status
}

// 배치가 있는 stage 가 보낸 [완료, 전체] 진행도를 노드별로 추출
// 배치가 없는 stage(예: 검증 입력 데이터 구성)는 백엔드가 progress 를 안 보내므로 null 유지 —
// 그 노드는 프론트에서 시간 기반 흉내 애니메이션으로 대체
export function progressMapFromPipelineStages(pipelineStages = []) {
  const progress = Object.fromEntries(NODE_IDS.map(id => [id, null]))

  for (const item of pipelineStages) {
    const nodeId = STAGE_KEY_TO_NODE_ID[item?.stage]
    if (nodeId && Array.isArray(item.progress) && item.progress.length === 2) {
      progress[nodeId] = item.progress
    }
  }

  return progress
}
