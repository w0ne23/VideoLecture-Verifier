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
