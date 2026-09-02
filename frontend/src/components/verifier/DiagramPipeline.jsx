// 검증 파이프라인 전체 흐름을 한 장의 SVG 로 그리는 큰 다이어그램
// (/dev/verify-demo-diagram, VerifyDemoLivePage 공용)
//
// 전처리(영상 분석/오디오 분석 두 갈래 → 통합 텍스트)와 검증(발화 검증/슬라이드 검증 두 갈래
// → 오류) 두 구간으로 구성. video/integrated_text/error_output 3개만 그림 아이콘이고 나머지
// 11개는 원형 노드 + 연결선. 레인(영상/오디오/발화/슬라이드) 라벨은 알약 배경(LaneCapsule)이
// 자기 줄의 노드 범위를 감쌈. 라벨은 폭이 허용하면 한 줄, 아니면 첫 공백 기준 두 줄로 접음

import { NODES } from './diagramPipelineConstants'

const NODE_FONT_SIZE = 13

// 노드 id → SVG 좌표 (기본 = 오류 필터링 노드 포함)
const NODE_POS = {
  video: { x: 45, y: 160 },
  slide_extract: { x: 155, y: 68 },
  slide_analyze: { x: 261, y: 68 },
  audio_quality: { x: 155, y: 252 },
  voice_transcribe: { x: 261, y: 252 },
  integrated_text: { x: 367, y: 160 },
  claim_extract: { x: 489, y: 60 },
  issue_detect: { x: 566, y: 60 },
  issue_classify: { x: 644, y: 60 },
  issue_filter: { x: 721, y: 60 },
  issue_judge: { x: 799, y: 60 },
  // 슬라이드 검증 앞 두 단계를 발화 검증 앞 두 단계와 같은 x 좌표에 맞춰 두 레인을 세로로 정렬
  slide_inspect: { x: 489, y: 260 },
  syntax_verify: { x: 566, y: 260 },
  error_output: { x: 799, y: 260 },
}

// "웹그라운딩 미포함" 셋이면 백엔드가 verifier_web_grounding stage 를 시작 시점부터 'skip' 처리
// → 오류 필터링 노드를 아예 그리지 않고 오류 판단을 그 x 좌표(721)로 당김
// 피드백은 y 좌표(260)는 유지하고 x 좌표만 721 로 당겨 마지막 열 위아래 정렬을 유지
const NODE_POS_NO_GROUNDING = {
  issue_judge: { x: 721, y: 60 },
  error_output: { x: 721, y: 260 },
}

function posFor(id, groundingEnabled) {
  return (!groundingEnabled && NODE_POS_NO_GROUNDING[id]) || NODE_POS[id]
}

// 레인마다 노드 간격이 달라(발화검증 5개는 빽빽, 슬라이드검증 2개는 넉넉) 한 줄 표시 가능 최대 폭도 다름
const LANE_MAX_WIDTH = { video: 100, audio: 100, utterance: 68, slide: 68 }
const ICON_MAX_WIDTH = 150

// 발화 검증(rose)·슬라이드 검증(amber) 선/노드 색 — 여러 팔레트 비교 후 고정
const LANE_COLORS = { utterance: 'var(--rose)', slide: 'var(--amber)' }
// 강의 영상·통합 텍스트·피드백의 겉모양(박스/문서/스택)은 차콜 무채색, 안의 그림·음성 아이콘만 별도 색
const ICON_COLOR = 'var(--charcoal)'

// 연결선 정의
// 같은 레인 안 노드끼리 잇던 선은 알약 배경(LaneCapsule)이 대신하므로 제거, 레인을 넘나드는 선만 유지
// lane 값은 색 비교 실험(diffLine) 켰을 때 팔레트 선택에 사용 — 전처리 쪽 선은 실험 대상이 아니라 lane 없음
// 전처리 구간과 검증 진입부는 그라운딩 포함 여부와 무관하게 고정이라 공용
const COMMON_CONNECTORS = [
  { target: 'slide_extract', d: 'M79 138 C 110 100, 130 78, 146 70' },
  { target: 'audio_quality', d: 'M79 182 C 110 220, 126 245, 146 250' },
  { target: 'integrated_text', d: 'M270 68 C 297 85, 314 110, 328 138' },
  { target: 'integrated_text', d: 'M270 252 C 297 230, 314 205, 328 182' },
  { target: 'claim_extract', d: 'M406 138 C 432 100, 456 75, 480 64', lane: 'utterance' },
  { target: 'slide_inspect', d: 'M406 182 C 424 210, 452 245, 479 258', lane: 'slide' },
]

// 피드백으로 들어가는 선 — 오류 판단·피드백은 x 좌표만 799→721 로 당겨졌을 뿐 상대 위치는 동일하므로
// 그라운딩 있을 때 곡선을 x 축으로 78(=799-721) 평행이동한 형태
// 바로 아래로 내려가면 오류 판단 라벨 글자를 지나가므로, 노드 오른쪽으로 살짝 빠졌다가 완만한 곡선으로 하강
const FEEDBACK_CONNECTORS = {
  withGrounding: { target: 'error_output', d: 'M813 58 C 855 62, 855 190, 818 227', lane: 'utterance' },
  noGrounding: { target: 'error_output', d: 'M735 58 C 777 62, 777 190, 740 227', lane: 'utterance' },
}

// 슬라이드 검증(문법 검증)은 발화 검증보다 먼저 끝나므로, 이 선은 error_output 이 아니라
// syntax_verify 자신의 완료 여부로 상태를 잡아 미리 뻗어나가 있게 함
// error_output 의 y 좌표는 항상 260 이라 끝점 x 좌표만 765(그라운딩 있음) / 687(없음) 로 조정
const SYNTAX_TO_FEEDBACK_CONNECTORS = {
  withGrounding: { target: 'error_output', source: 'syntax_verify', d: 'M576 260 L765 260', lane: 'slide' },
  noGrounding: { target: 'error_output', source: 'syntax_verify', d: 'M576 260 L687 260', lane: 'slide' },
}

// source 가 target 보다 먼저 끝나는 선(예: 슬라이드 검증→피드백) 상태 판정
// source 완료 전: wait, source 완료 후 target 완료 전: run(점선, 도착해 대기), 둘 다 완료: done(실선)
function edgeStatus(sourceStatus, targetStatus) {
  if (targetStatus === 'done') return 'done'
  if (sourceStatus === 'done') return 'run'
  return 'wait'
}

function connectorStyle(status, lane, diffLine) {
  const classes = ['diag-connector']
  if (status === 'error') classes.push('diag-connector--error')
  else if (status === 'run') classes.push('diag-connector--run')
  else if (status === 'done') classes.push('diag-connector--done')
  else classes.push('diag-connector--wait')

  let style
  if (diffLine && lane && (status === 'run' || status === 'done')) {
    style = { stroke: LANE_COLORS[lane] }
  }
  return { className: classes.join(' '), style }
}

// 한 줄 표시 가능 여부 가늠용 폭 추정 — 한글/영문은 fontSize 기준 폭, 공백은 그보다 좁게
function estimateTextWidth(label, fontSize) {
  let width = 0
  for (const ch of label) width += ch === ' ' ? fontSize * 0.45 : fontSize * 1.0
  return width
}

// 노드 라벨 — maxWidth 를 넘거나 forceWrap 이면 첫 공백 기준 두 줄로 접음
function DiagLabel({ label, x, y, maxWidth = 100, forceWrap = false }) {
  if (!forceWrap && estimateTextWidth(label, NODE_FONT_SIZE) <= maxWidth) {
    return <text x={x} y={y} textAnchor="middle" className="diag-node-label">{label}</text>
  }
  const [first, ...rest] = label.split(' ')
  const second = rest.join(' ')
  return (
    <text x={x} y={y} textAnchor="middle" className="diag-node-label">
      <tspan x={x} dy="0">{first}</tspan>
      {second && <tspan x={x} dy="16">{second}</tspan>}
    </text>
  )
}

// 슬라이드 추출·슬라이드 분석은 폭에 여유가 있어도 "슬라이드"에서 줄바꿈해 두 줄로 고정
const FORCE_WRAP_IDS = new Set(['slide_extract', 'slide_analyze'])

// 원형 노드 + 라벨 (run 상태면 파동 링 추가)
function PlainNode({ id, label, lane, status, diffNode, pos }) {
  const { x, y } = pos
  let style
  if (diffNode && (lane === 'utterance' || lane === 'slide') && (status === 'run' || status === 'done')) {
    const color = LANE_COLORS[lane]
    style = { fill: color, stroke: color }
  }
  return (
    <g>
      {status === 'run' && (
        <circle cx={x} cy={y} r="12" className="diag-node-wave" style={style && { stroke: style.stroke }} />
      )}
      <circle cx={x} cy={y} r="12" className={`diag-node diag-node--${status}`} style={style} />
      <DiagLabel
        label={label}
        x={x}
        y={y + 35}
        maxWidth={LANE_MAX_WIDTH[lane]}
        forceWrap={FORCE_WRAP_IDS.has(id)}
      />
    </g>
  )
}

// 아이콘 활성(run/done) 여부에 따른 stroke/fill 오버라이드
function iconStyles(status) {
  const active = status === 'run' || status === 'done'
  return {
    box: active ? { stroke: ICON_COLOR } : undefined,
    fill: active ? { fill: ICON_COLOR } : undefined,
  }
}

// 강의 영상 아이콘 (재생 삼각형이 든 박스)
function VideoIcon({ status }) {
  const { x, y } = NODE_POS.video
  const s = iconStyles(status)
  return (
    <g transform={`translate(${x - 37},${y - 27})`}>
      <rect width="74" height="54" rx="9" className={`diag-icon-box diag-icon-box--${status}`} style={s.box} />
      <polygon points="29,16 29,38 49,27" className={`diag-icon-play diag-icon-play--${status}`} style={s.fill} />
      <DiagLabel label="강의 영상" x={37} y={82} maxWidth={ICON_MAX_WIDTH} />
    </g>
  )
}

// 오른쪽 위 모서리를 접은 문서 모양 — 접힌 삼각형도 본체와 같은 상태색 클래스라 테두리만 도드라짐
function DocumentIcon({ x, y, w, h, fold, status }) {
  const boxClass = `diag-icon-box diag-icon-box--${status}`
  return (
    <>
      <path d={`M${x} ${y} L${x + w - fold} ${y} L${x + w} ${y + fold} L${x + w} ${y + h} L${x} ${y + h} Z`} className={boxClass} />
      <path d={`M${x + w - fold} ${y} L${x + w - fold} ${y + fold} L${x + w} ${y + fold} Z`} className={boxClass} />
    </>
  )
}

// 통합 텍스트가 영상 + 음성 두 입력을 합친 것임을 보여주는 미니 아이콘
// 이미지 픽토그램: 액자 + 해 + 산 모양 (재생 버튼/필름 스트립은 알아보기 어려워 대체)
function MiniImageGlyph({ x, y, status, scale = 1 }) {
  const markClass = `diag-mini-mark diag-mini-mark--${status}`
  return (
    <g transform={`translate(${x},${y}) scale(${scale})`}>
      <rect width="20" height="14" rx="2" className={`diag-icon-frame diag-icon-frame--${status}`} />
      <circle cx="5.5" cy="4.5" r="1.8" className={markClass} />
      <polygon points="2,12 8,6 11.5,9.5 15,6.5 18,12" className={markClass} />
    </g>
  )
}

// 스피커 + 음파 미니 아이콘
function MiniAudioGlyph({ x, y, status, scale = 1 }) {
  return (
    <g transform={`translate(${x},${y}) scale(${scale})`}>
      <polygon points="0,4 5,4 10,0 10,14 5,10 0,10" className={`diag-mini-mark diag-mini-mark--${status}`} />
      <path d="M13,3 C16,7 16,7 13,11" className={`diag-icon-wave diag-icon-wave--${status}`} />
      <path d="M16,0 C20,7 20,7 16,14" className={`diag-icon-wave diag-icon-wave--${status}`} />
    </g>
  )
}

// 멀티모달 통합 텍스트 아이콘 (문서 + 이미지/음성 미니 아이콘 + 본문 줄)
function TextIcon({ status }) {
  const { x, y } = NODE_POS.integrated_text
  const s = iconStyles(status)
  const w = 84
  const h = 96
  const fold = 16
  return (
    <g transform={`translate(${x - w / 2},${y - h / 2})`}>
      <DocumentIcon x={0} y={0} w={w} h={h} fold={fold} status={status} />
      <MiniImageGlyph x={10} y={20} status={status} scale={1.3} />
      <MiniAudioGlyph x={48} y={21} status={status} scale={1.3} />
      <rect x="10" y="52" width="58" height="4" rx="2" className={`diag-icon-docline diag-icon-docline--${status}`} style={s.fill} />
      <rect x="10" y="62" width="42" height="4" rx="2" className={`diag-icon-docline diag-icon-docline--${status}`} style={s.fill} />
      <rect x="10" y="72" width="50" height="4" rx="2" className={`diag-icon-docline diag-icon-docline--${status}`} style={s.fill} />
      <DiagLabel label="멀티모달 통합 텍스트" x={w / 2} y={h + 28} maxWidth={ICON_MAX_WIDTH} forceWrap />
    </g>
  )
}

// 피드백 아이콘 — 여러 건임을 보이려 종이 두 장을 겹치고, 안에는 경고 삼각형으로 "오류"임을 표시
function StackIcon({ status, pos }) {
  const { x, y } = pos
  const s = iconStyles(status)
  const w = 62
  const h = 76
  return (
    <g transform={`translate(${x - (w + 12) / 2},${y - 46})`}>
      <rect x="12" y="12" width={w} height={h} rx="10" className={`diag-stack-sheet diag-stack-sheet--back diag-icon-box--${status}`} style={s.box} />
      <rect x="6" y="6" width={w} height={h} rx="10" className={`diag-stack-sheet diag-stack-sheet--mid diag-icon-box--${status}`} style={s.box} />
      <DocumentIcon x={0} y={0} w={w} h={h} fold={14} status={status} />
      <polygon points="31,10 43,32 19,32" className={`diag-warning-triangle diag-warning-triangle--${status}`} />
      <rect x="29.5" y="16" width="3" height="10" rx="1.5" className="diag-warning-mark" />
      <circle cx="31" cy="29" r="1.6" className="diag-warning-mark" />
      <rect x="10" y="42" width="38" height="4" rx="2" className={`diag-icon-docline diag-icon-docline--${status}`} style={s.fill} />
      <rect x="10" y="52" width="30" height="4" rx="2" className={`diag-icon-docline diag-icon-docline--${status}`} style={s.fill} />
      <rect x="10" y="62" width="34" height="4" rx="2" className={`diag-icon-docline diag-icon-docline--${status}`} style={s.fill} />
      <DiagLabel label="피드백(지식 오류)" x={w / 2} y={h + 40} maxWidth={ICON_MAX_WIDTH} />
    </g>
  )
}

const ICON_COMPONENTS = { video: VideoIcon, integrated_text: TextIcon, error_output: StackIcon }

function LaneLabel({ x, y, label }) {
  return <text x={x} y={y} textAnchor="start" className="diag-lane-label">{label}</text>
}

// 레인 안 노드들을 알약 배경으로 한 번에 감쌈 (반지름은 노드 반지름 12 보다 살짝 크게)
// 레인 안 노드가 하나라도 run/done 이 되면 진하게 살아남 — 색은 그대로 두고 투명도만 변경
function LaneCapsule({ x1, x2, y, r = 17, tone, active }) {
  return (
    <rect
      x={x1 - r}
      y={y - r}
      width={x2 - x1 + r * 2}
      height={r * 2}
      rx={r}
      ry={r}
      className={`diag-lane-capsule diag-lane-capsule--${tone}${active ? '' : ' diag-lane-capsule--dim'}`}
    />
  )
}

function isLaneActive(status, ids) {
  return ids.some(id => status[id] === 'run' || status[id] === 'done')
}

// status: 노드 id → wait/run/done/error/skip 맵 (statusMapFromPipelineStages 또는 데모 훅에서 생성)
// diffLine / diffNode: 레인 색 비교 실험용 토글, compact: 축소 표시
export default function DiagramPipeline({ status, diffLine = false, diffNode = false, compact = false }) {
  // "웹그라운딩 미포함" 잡은 verifier_web_grounding stage 가 시작부터 'skip' 이라,
  // 이 값 하나로 오류 필터링 노드 표시 여부 + 딸린 레이아웃(오류 판단·피드백 위치, 연결선)을 전부 결정
  const groundingEnabled = status.issue_filter !== 'skip'
  const visibleNodes = groundingEnabled ? NODES : NODES.filter(node => node.id !== 'issue_filter')
  const connectors = [
    ...COMMON_CONNECTORS,
    groundingEnabled ? FEEDBACK_CONNECTORS.withGrounding : FEEDBACK_CONNECTORS.noGrounding,
    groundingEnabled ? SYNTAX_TO_FEEDBACK_CONNECTORS.withGrounding : SYNTAX_TO_FEEDBACK_CONNECTORS.noGrounding,
  ]
  // 그라운딩 없으면 발화 검증 레인은 오류 판단(721)에서 끝남 — 피드백은 아래 별도 행이라 알약에 미포함
  const utteranceCapsuleEnd = groundingEnabled ? 799 : 721
  // 오류 필터링 노드가 빠지면 전체 그림이 왼쪽으로 쏠려 보이므로, 노드 좌표는 그대로 두고
  // viewBox 만 왼쪽으로 패닝해 줄어든 콘텐츠 폭이 다시 캔버스 한가운데 오게 함
  const viewBox = groundingEnabled ? '0 0 880 360' : '-52 0 880 360'

  const videoLaneActive = isLaneActive(status, ['slide_extract', 'slide_analyze'])
  const audioLaneActive = isLaneActive(status, ['audio_quality', 'voice_transcribe'])
  const utteranceLaneActive = isLaneActive(status, ['claim_extract', 'issue_detect', 'issue_classify', 'issue_filter', 'issue_judge'])
  const slideLaneActive = isLaneActive(status, ['slide_inspect', 'syntax_verify'])

  return (
    <svg
      viewBox={viewBox}
      className={compact ? 'diag-svg diag-svg--compact' : 'diag-svg'}
      aria-hidden="true"
    >
      <LaneLabel x={148} y={41} label="영상 분석" />
      <LaneLabel x={148} y={225} label="오디오 분석" />
      <LaneLabel x={482} y={33} label="발화 검증" />
      <LaneLabel x={482} y={233} label="슬라이드 검증" />

      <LaneCapsule x1={155} x2={261} y={68} tone="pre" active={videoLaneActive} />
      <LaneCapsule x1={155} x2={261} y={252} tone="pre" active={audioLaneActive} />
      <LaneCapsule x1={489} x2={utteranceCapsuleEnd} y={60} tone="utterance" active={utteranceLaneActive} />
      <LaneCapsule x1={489} x2={566} y={260} tone="slide" active={slideLaneActive} />

      {connectors.map((c, i) => {
        const rawStatus = c.source
          ? edgeStatus(status[c.source], status[c.target])
          : status[c.target]
        const { className, style } = connectorStyle(rawStatus, c.lane, diffLine)
        return <path key={i} d={c.d} className={className} style={style} />
      })}

      {visibleNodes.map(node => {
        const pos = posFor(node.id, groundingEnabled)
        if (node.icon) {
          const Icon = ICON_COMPONENTS[node.id]
          return <Icon key={node.id} status={status[node.id]} pos={pos} />
        }
        return (
          <PlainNode
            key={node.id}
            id={node.id}
            label={node.label}
            lane={node.lane}
            status={status[node.id]}
            diffNode={diffNode}
            pos={pos}
          />
        )
      })}
    </svg>
  )
}
