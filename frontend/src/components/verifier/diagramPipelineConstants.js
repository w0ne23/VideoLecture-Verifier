// 첨부된 아키텍처 다이어그램(전처리 → 검증, 각 구간 안에서 두 갈래로 나뉘었다 합쳐짐)을
// 그대로 옮긴 데모 전용 단계 정의. 실제 백엔드 파이프라인(verifierConstants.js의
// PIPELINE_NODES, 9단계)과는 별개의, 구조 설명용 라벨이다 — /dev/verify-demo-diagram 전용.
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

// 한 틱에 동시에 활성화되는 노드들. 영상/오디오, 발화검증/슬라이드검증처럼 병렬로
// 진행되는 구간은 같은 틱에 묶고, 슬라이드검증(2단계)이 먼저 끝나면 그 뒤로는
// 발화검증(5단계)만 남아 혼자 진행된다.
// video는 파이프라인 화면에 들어오기 전(업로드 시점)에 이미 끝난 단계라 별도 틱 없이
// 항상 done 상태로 표시하고, 첫 틱부터 바로 슬라이드 추출·오디오 품질 분석이 진행된다.
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

// 발표용 데모(VerifyDemoLivePage) 노드별 소요 시간 — 여기 숫자만 바꾸면 각 레인의 시작·
// 종료 시각이 laneSchedule로 자동 재계산된다. 같은 구간에서 나란히 도는 두 노드(예: 슬라이드
// 추출/오디오 품질)는 서로 다른 시간을 줘도 되며, 먼저 끝난 쪽은 "완료" 상태로 남아 상대
// 노드를 기다린다.
// - 전처리: 영상 레인(슬라이드 추출→분석)과 오디오 레인(품질 분석→전사)의 "레인 총합"을
//   맞춰서 같은 시점에 통합 텍스트로 합류하게 하되, 앞 단계에서 영상이 오래 걸리면 뒤
//   단계에서는 오디오가 오래 걸리도록 배분을 반대로 뒤집었다.
// - 검증: 슬라이드 검증(slide_inspect·syntax_verify) 2단계는 발화 검증(claim_extract부터
//   issue_judge까지 5단계)보다 훨씬 먼저 끝나도록, 각 단계도 발화 검증의 짝보다 짧게 잡았다.
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

// startMs부터 시작해 ids를 순서대로 이어 붙인 레인 하나의 일정을 만든다.
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

const videoLane = laneSchedule(0, ['slide_extract', 'slide_analyze'])
const audioLane = laneSchedule(0, ['audio_quality', 'voice_transcribe'])
const integratedLane = laneSchedule(Math.max(videoLane.endMs, audioLane.endMs), ['integrated_text'])
const verifyStartMs = integratedLane.endMs
const utteranceLane = laneSchedule(verifyStartMs, ['claim_extract', 'issue_detect', 'issue_classify', 'issue_filter', 'issue_judge'])
const slideLane = laneSchedule(verifyStartMs, ['slide_inspect', 'syntax_verify'])
const feedbackLane = laneSchedule(Math.max(utteranceLane.endMs, slideLane.endMs), ['error_output'])

// 노드 id → { start, duration }(ms). 상태(wait/run/done)는 elapsedMs와 이 스케줄만
// 비교해서 구한다 — 틱 인덱스 같은 별도 진행 상태를 두지 않는다.
export const NODE_SCHEDULE = {
  ...videoLane.entries,
  ...audioLane.entries,
  ...integratedLane.entries,
  ...utteranceLane.entries,
  ...slideLane.entries,
  ...feedbackLane.entries,
}

export const TOTAL_DURATION_MS = feedbackLane.endMs
export const PREPROCESS_END_MS = integratedLane.endMs

// "멀티모달 강의 영상 분석(전처리)"/"지식 오류 탐지(검증)" 큰 분류 — 브래킷 대신
// 파이프라인 진행 텍스트 쪽에 표시한다.
const PRE_PHASE_IDS = new Set(['slide_extract', 'slide_analyze', 'audio_quality', 'voice_transcribe', 'integrated_text'])

export function bigPhaseFor(id) {
  return PRE_PHASE_IDS.has(id) ? '멀티모달 강의 영상 분석' : '지식 오류 탐지'
}

export function formatDuration(ms) {
  const totalSeconds = Math.max(0, Math.round(ms / 1000))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

// 진행 바 색을 다이어그램의 레인 색과 맞추기 위한 매핑 — 전처리(영상/오디오)는 primary,
// 검증 구간은 발화(rose)·슬라이드(amber)로 diag-lane-capsule과 동일한 톤을 쓴다.
const LANE_TONE = { video: 'pre', audio: 'pre', utterance: 'utterance', slide: 'slide' }

export function laneToneFor(id) {
  return LANE_TONE[NODE_BY_ID[id]?.lane] || 'pre'
}

// 실제 검증 진행 화면(VerifyProgressPage)이 SSE로 받는 pipelineStages([{stage, status}],
// verifierConstants.js의 PIPELINE_NODES와 동일한 stageKey 집합)를 이 다이어그램의 노드별
// status 맵으로 바꾼다. 백엔드가 이제 각 stage를 다이어그램 노드 하나에 정확히 대응하도록
// 나눠 보고하므로(preprocess.py/classified_slide_error_checker.py), "여러 노드가 상태를
// 공유"하는 임시방편 없이 그대로 1:1로 옮기면 된다. error_output만 예외로, 발화 체인
// (issue_judge = verifier_final_verification)과 슬라이드 체인(syntax_verify =
// verify_slide_syntax)이 둘 다 done이어야 done으로 본다 — 백엔드에 그 자체를 위한 stage가
// 따로 없다.
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

export function statusMapFromPipelineStages(pipelineStages = []) {
  const status = Object.fromEntries(NODE_IDS.map(id => [id, 'wait']))
  // video는 화면 진입 전(업로드 시점)에 이미 끝난 단계라 항상 done으로 고정한다.
  status.video = 'done'

  for (const item of pipelineStages) {
    const nodeId = STAGE_KEY_TO_NODE_ID[item?.stage]
    if (nodeId) status[nodeId] = item.status
  }

  status.error_output = status.issue_judge === 'done' && status.syntax_verify === 'done' ? 'done' : 'wait'

  return status
}
