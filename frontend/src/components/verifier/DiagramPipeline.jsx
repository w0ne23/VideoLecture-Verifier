import { NODES } from './diagramPipelineConstants'

// /dev/verify-demo-diagram, VerifyDemoLivePage 공용 큰 다이어그램. 전처리(영상 분석/
// 오디오 분석 두 갈래 → 통합 텍스트)와 검증(발화 검증/슬라이드 검증 두 갈래 → 오류) 두
// 구간을 한 장의 SVG로 그린다. video/integrated_text/error_output 3개만 그림 아이콘이고
// 나머지 11개는 기존 데모 파이프라인과 같은 원형 노드+연결선 스타일이다. "① 멀티모달
// 강의 영상 분석 / ② 지식 오류 탐지" 구간 브래킷은 파이프라인 진행 텍스트·진행 바가
// 같은 정보를 이미 보여주므로 제거했다. 레인(영상/오디오/발화/슬라이드) 라벨은 알약
// 배경(LaneCapsule)이 자기 줄의 노드 범위를 가리킨다. 라벨은 폭이 허용하면 한 줄로,
// 안 되면 첫 공백 기준 두 줄로 접는다.

const NODE_FONT_SIZE = 13

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
  // 슬라이드 검증의 앞 두 단계를 발화 검증의 앞 두 단계(주장 추출·이슈 탐지)와 같은
  // x좌표에 맞춰서 두 레인이 세로로 나란히 정렬되게 한다.
  slide_inspect: { x: 489, y: 260 },
  syntax_verify: { x: 566, y: 260 },
  error_output: { x: 799, y: 260 },
}

// 적용된 LLM 셋이 "웹그라운딩 미포함"이면 백엔드가 오류 필터링(verifier_web_grounding)
// stage 자체를 건너뛴다(worker.py가 시작 시점에 바로 status: 'skip'으로 표시). 이 노드는
// 아예 그리지 않고 오류 판단이 그 x좌표(721)로 당겨온다. 피드백은 원래 y좌표(260, 슬라이드
// 검증과 같은 행)는 그대로 두고 x좌표만 오류 판단과 같은 721로 당겨서, 두 노드가 원래처럼
// 위아래로 정렬된 마지막 열을 그대로 유지한다(원래도 둘 다 x=799로 같은 열이었다).
const NODE_POS_NO_GROUNDING = {
  issue_judge: { x: 721, y: 60 },
  error_output: { x: 721, y: 260 },
}

function posFor(id, groundingEnabled) {
  return (!groundingEnabled && NODE_POS_NO_GROUNDING[id]) || NODE_POS[id]
}

// 레인마다 노드 사이 간격이 달라서(발화검증 5개는 빽빽, 슬라이드검증 2개는 넉넉) 한 줄로
// 표시 가능한 최대 폭도 다르게 잡는다.
const LANE_MAX_WIDTH = { video: 100, audio: 100, utterance: 68, slide: 68 }
const ICON_MAX_WIDTH = 150

// 발화 검증(로즈)·슬라이드 검증(틸) 선/노드 색 — 여러 팔레트를 만들어봤지만 비교해보니
// 이 조합이 제일 나아서 고정했다. "선만 다르게"/"노드만 다르게" 토글만 남긴다.
const LANE_COLORS = { utterance: 'var(--rose)', slide: 'var(--amber)' }
// 강의 영상·멀티모달 통합 텍스트·피드백의 겉모양(박스/문서/스택)은 차콜 무채색으로 —
// 안에 든 그림·음성 아이콘만 별도 색(diag-mini-mark)을 그대로 유지한다.
const ICON_COLOR = 'var(--charcoal)'

// 발화검증/슬라이드검증 구간에 속하는 연결선만 lane을 표시해 둔다 — 색 비교 실험(diffLine)
// 켰을 때 이 lane 값으로 팔레트에서 어느 색을 쓸지 정한다. 전처리 쪽 연결선은 실험 대상이
// 아니라 lane이 없다.
// 같은 레인 안에서 노드끼리 이어지던 연결선은 이제 알약 배경(LaneCapsule)이 그 역할을
// 대신하므로 제거했다. 레인을 넘나드는 연결선만 남긴다.
// 전처리 구간과 발화·슬라이드 검증 진입부는 그라운딩 포함 여부와 무관하게 항상 같은 자리에
// 있으므로 공용으로 둔다. 피드백(error_output)으로 들어가는 두 연결선만 그라운딩 여부에 따라
// error_output의 좌표가 달라져(y=260 vs y=60) 별도 버전이 필요하다.
const COMMON_CONNECTORS = [
  { target: 'slide_extract', d: 'M79 138 C 110 100, 130 78, 146 70' },
  { target: 'audio_quality', d: 'M79 182 C 110 220, 126 245, 146 250' },
  { target: 'integrated_text', d: 'M270 68 C 297 85, 314 110, 328 138' },
  { target: 'integrated_text', d: 'M270 252 C 297 230, 314 205, 328 182' },
  { target: 'claim_extract', d: 'M406 138 C 432 100, 456 75, 480 64', lane: 'utterance' },
  { target: 'slide_inspect', d: 'M406 182 C 424 210, 452 245, 479 258', lane: 'slide' },
]

// 오류 판단·피드백 모두 x좌표만 799→721로 당겨왔을 뿐 서로의 상대 위치(같은 열, 위/아래
// 200px 간격)는 그대로라, 그라운딩 있을 때 곡선을 x축으로 78(=799-721)만큼 그대로 평행이동
// 하면 된다.
const FEEDBACK_CONNECTORS = {
  // 곧바로 아래로 내려가면 이슈 판단 라벨 글자 위를 지나가므로, 노드 오른쪽으로 살짝
  // 빠져나와 라벨을 비켜간 뒤 완만한 곡선 하나로 내려온다.
  withGrounding: { target: 'error_output', d: 'M813 58 C 855 62, 855 190, 818 227', lane: 'utterance' },
  noGrounding: { target: 'error_output', d: 'M735 58 C 777 62, 777 190, 740 227', lane: 'utterance' },
}

// 슬라이드 검증(문법 검증)은 발화 검증보다 먼저 끝나므로, 이 엣지는 error_output이 아니라
// syntax_verify 자신의 완료 여부로 상태를 잡아 발화 검증이 끝나기 전에 먼저 뻗어나가 있게 한다.
// error_output의 y좌표는 그라운딩 여부와 무관하게 항상 260이므로, 이 연결선은 끝점 x좌표만
// 799 근방(765)에서 721 근방(687)으로 당기면 된다.
const SYNTAX_TO_FEEDBACK_CONNECTORS = {
  withGrounding: { target: 'error_output', source: 'syntax_verify', d: 'M576 260 L765 260', lane: 'slide' },
  noGrounding: { target: 'error_output', source: 'syntax_verify', d: 'M576 260 L687 260', lane: 'slide' },
}

// source가 target보다 먼저 끝나는 엣지(예: 슬라이드 검증→피드백)는 source가 끝나기 전엔
// wait(연결 전) 그대로 두고, source가 끝나면 target이 끝나기 전까지는 "이미 도착해서
// 기다리는" run(점선)으로, target까지 끝나야 비로소 done(실선)으로 바뀐다.
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

// 한글/영문 글자는 fontSize 기준 폭, 공백은 그보다 좁게 잡아 한 줄 표시 가능 여부를 가늠한다.
function estimateTextWidth(label, fontSize) {
  let width = 0
  for (const ch of label) width += ch === ' ' ? fontSize * 0.45 : fontSize * 1.0
  return width
}

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

// 슬라이드 추출·슬라이드 분석은 폭에 여유가 있어도 "슬라이드"에서 줄바꿈해 두 줄로 고정한다.
const FORCE_WRAP_IDS = new Set(['slide_extract', 'slide_analyze'])

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

function iconStyles(status) {
  const active = status === 'run' || status === 'done'
  return {
    box: active ? { stroke: ICON_COLOR } : undefined,
    fill: active ? { fill: ICON_COLOR } : undefined,
  }
}

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

// 오른쪽 위 모서리를 접은 문서 모양. 접힌 삼각형도 본체와 같은 상태색 클래스를 써서
// 배경색(카드색)은 본체와 같고 테두리만 도드라지게 해 "접힌 자국"처럼 보이게 한다.
function DocumentIcon({ x, y, w, h, fold, status }) {
  const boxClass = `diag-icon-box diag-icon-box--${status}`
  return (
    <>
      <path d={`M${x} ${y} L${x + w - fold} ${y} L${x + w} ${y + fold} L${x + w} ${y + h} L${x} ${y + h} Z`} className={boxClass} />
      <path d={`M${x + w - fold} ${y} L${x + w - fold} ${y + fold} L${x + w} ${y + fold} Z`} className={boxClass} />
    </>
  )
}

// 영상(그림/사진 아이콘)·음성(스피커+음파) 미니 아이콘 — 통합 텍스트가 두 입력을
// 합친 것임을 보여준다. 색은 문서 안 글자줄과 같은 상태색 클래스를 그대로 쓴다.
// 재생 버튼·필름 스트립은 알아보기 어려워서, 흔히 쓰는 "사진" 픽토그램(액자 + 해 +
// 산 모양)으로 바꿨다.
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

function MiniAudioGlyph({ x, y, status, scale = 1 }) {
  return (
    <g transform={`translate(${x},${y}) scale(${scale})`}>
      <polygon points="0,4 5,4 10,0 10,14 5,10 0,10" className={`diag-mini-mark diag-mini-mark--${status}`} />
      <path d="M13,3 C16,7 16,7 13,11" className={`diag-icon-wave diag-icon-wave--${status}`} />
      <path d="M16,0 C20,7 20,7 16,14" className={`diag-icon-wave diag-icon-wave--${status}`} />
    </g>
  )
}

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

// 피드백은 문서 한 장이 아니라 여러 건이므로 뒤에 종이 두 장을 더 겹쳐 "여러 개"임을
// 보여주고, 안에는 줄글 대신 경고 표시 하나로 "오류"라는 걸 바로 알아보게 한다.
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

// 레인 안의 노드들을 밑줄 대신 알약 모양 배경으로 한 번에 묶어서 보여준다.
// 노드 라벨 텍스트와 겹치지 않도록 반지름을 노드 반지름(12)보다 살짝만 크게 둔다.
// 레인 안 노드가 하나라도 run/done이 되기 전까지는 옅게 죽어있다가, 하나라도
// 불이 들어오는 순간 진하게 살아난다. 색 자체는 그대로 두고 투명도만 바꾼다.
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

export default function DiagramPipeline({ status, diffLine = false, diffNode = false, compact = false }) {
  // "웹그라운딩 미포함" 셋으로 실행된 잡은 worker.py가 verifier_web_grounding stage를
  // 시작 시점부터 'skip'으로 못박아 두므로, 이 값 하나로 오류 필터링 노드 표시 여부와
  // 그에 딸린 레이아웃(오류 판단·피드백 위치, 연결선)을 전부 결정할 수 있다.
  const groundingEnabled = status.issue_filter !== 'skip'
  const visibleNodes = groundingEnabled ? NODES : NODES.filter(node => node.id !== 'issue_filter')
  const connectors = [
    ...COMMON_CONNECTORS,
    groundingEnabled ? FEEDBACK_CONNECTORS.withGrounding : FEEDBACK_CONNECTORS.noGrounding,
    groundingEnabled ? SYNTAX_TO_FEEDBACK_CONNECTORS.withGrounding : SYNTAX_TO_FEEDBACK_CONNECTORS.noGrounding,
  ]
  // 그라운딩이 없으면 발화 검증 레인은 오류 판단(721)에서 끝난다. 피드백은 이제 그
  // 아래(y=260) 별도 행에 있으므로 "발화 검증" 알약에는 포함되지 않는다.
  const utteranceCapsuleEnd = groundingEnabled ? 799 : 721
  // 오류 필터링 노드 하나가 빠지면서 오류 판단·피드백이 78px 왼쪽으로 당겨진 만큼 전체
  // 그림이 캔버스 왼쪽으로 쏠려 보인다. 노드 좌표는 그대로 두고 보이는 창(viewBox)만
  // 왼쪽으로 패닝해서, 줄어든 콘텐츠 폭(getBBox 기준)이 다시 캔버스 한가운데 오도록 맞춘다.
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
