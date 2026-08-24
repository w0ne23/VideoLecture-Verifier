import { NODES } from './diagramPipelineConstants'

// /dev/verify-demo-diagram 전용 큰 다이어그램. 전처리(영상 분석/오디오 분석 두 갈래 →
// 통합 텍스트)와 검증(발화 검증/슬라이드 검증 두 갈래 → 오류) 두 구간을 한 장의 SVG로
// 그린다. video/integrated_text/error_output 3개만 그림 아이콘이고 나머지 11개는
// 기존 데모 파이프라인과 같은 원형 노드+연결선 스타일이다. 구간 구분은 사각 박스가
// 아니라 아래쪽 꺾은선(대괄호 모양 브래킷)으로 표시하고, 레인(영상/오디오/발화/슬라이드)
// 라벨도 밑줄로 자기 줄의 노드 범위를 가리키게 해서 "뭘 가리키는지 모르겠다"는 문제를
// 없앤다. 라벨은 폭이 허용하면 한 줄로, 안 되면 첫 공백 기준 두 줄로 접는다.

const NODE_FONT_SIZE = 13

const NODE_POS = {
  video: { x: 55, y: 160 },
  slide_extract: { x: 190, y: 68 },
  slide_analyze: { x: 320, y: 68 },
  audio_quality: { x: 190, y: 252 },
  voice_transcribe: { x: 320, y: 252 },
  integrated_text: { x: 450, y: 160 },
  claim_extract: { x: 600, y: 60 },
  issue_detect: { x: 695, y: 60 },
  issue_classify: { x: 790, y: 60 },
  issue_filter: { x: 885, y: 60 },
  issue_judge: { x: 980, y: 60 },
  // 슬라이드 검증의 앞 두 단계를 발화 검증의 앞 두 단계(주장 추출·이슈 탐지)와 같은
  // x좌표에 맞춰서 두 레인이 세로로 나란히 정렬되게 한다.
  slide_inspect: { x: 600, y: 260 },
  syntax_verify: { x: 695, y: 260 },
  error_output: { x: 980, y: 260 },
}

// 레인마다 노드 사이 간격이 달라서(발화검증 5개는 빽빽, 슬라이드검증 2개는 넉넉) 한 줄로
// 표시 가능한 최대 폭도 다르게 잡는다.
const LANE_MAX_WIDTH = { video: 100, audio: 100, utterance: 68, slide: 68 }
const ICON_MAX_WIDTH = 150

// 발화 검증(로즈)·슬라이드 검증(틸) 선/노드 색, 아이콘(강의 영상·통합 텍스트·오류 산출물)
// 블루 색 — 여러 팔레트를 만들어봤지만 비교해보니 이 조합이 제일 나아서 고정했다.
// "선만 다르게"/"노드만 다르게" 토글만 남긴다.
const LANE_COLORS = { utterance: 'var(--rose)', slide: 'var(--teal)' }
const ICON_COLOR = 'var(--info)'

// 발화검증/슬라이드검증 구간에 속하는 연결선만 lane을 표시해 둔다 — 색 비교 실험(diffLine)
// 켰을 때 이 lane 값으로 팔레트에서 어느 색을 쓸지 정한다. 전처리 쪽 연결선은 실험 대상이
// 아니라 lane이 없다.
const CONNECTORS = [
  { target: 'slide_extract', d: 'M97 138 C 135 100, 160 78, 179 70' },
  { target: 'audio_quality', d: 'M97 182 C 135 220, 155 245, 179 250' },
  { target: 'slide_analyze', d: 'M201 68 L309 68' },
  { target: 'voice_transcribe', d: 'M201 252 L309 252' },
  { target: 'integrated_text', d: 'M331 68 C 365 85, 385 110, 402 138' },
  { target: 'integrated_text', d: 'M331 252 C 365 230, 385 205, 402 182' },
  { target: 'claim_extract', d: 'M498 138 C 530 100, 560 75, 589 64', lane: 'utterance' },
  { target: 'slide_inspect', d: 'M498 182 C 520 210, 555 245, 588 258', lane: 'slide' },
  { target: 'issue_detect', d: 'M611 60 L684 60', lane: 'utterance' },
  { target: 'issue_classify', d: 'M706 60 L779 60', lane: 'utterance' },
  { target: 'issue_filter', d: 'M801 60 L874 60', lane: 'utterance' },
  { target: 'issue_judge', d: 'M896 60 L969 60', lane: 'utterance' },
  { target: 'syntax_verify', d: 'M612 260 L683 260', lane: 'slide' },
  { target: 'error_output', d: 'M980 72 L980 231', lane: 'utterance' },
  { target: 'error_output', d: 'M707 260 L939 260', lane: 'slide' },
]

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

function DiagLabel({ label, x, y, maxWidth = 100 }) {
  if (estimateTextWidth(label, NODE_FONT_SIZE) <= maxWidth) {
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

function PlainNode({ id, label, lane, status, diffNode }) {
  const { x, y } = NODE_POS[id]
  let style
  if (diffNode && (lane === 'utterance' || lane === 'slide') && (status === 'run' || status === 'done')) {
    const color = LANE_COLORS[lane]
    style = { fill: color, stroke: color }
  }
  return (
    <g>
      {status === 'run' && (
        <circle cx={x} cy={y} r="12" className="diag-node-halo" style={style && { stroke: style.stroke }} />
      )}
      <circle cx={x} cy={y} r="12" className={`diag-node diag-node--${status}`} style={style} />
      <DiagLabel label={label} x={x} y={y + 30} maxWidth={LANE_MAX_WIDTH[lane]} />
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
    <g transform={`translate(${x - 43},${y - 31})`}>
      <rect width="86" height="62" rx="10" className={`diag-icon-box diag-icon-box--${status}`} style={s.box} />
      <polygon points="34,18 34,44 57,31" className={`diag-icon-play diag-icon-play--${status}`} style={s.fill} />
      <DiagLabel label="강의 영상" x={43} y={84} maxWidth={ICON_MAX_WIDTH} />
    </g>
  )
}

function TextIcon({ status }) {
  const { x, y } = NODE_POS.integrated_text
  const s = iconStyles(status)
  return (
    <g transform={`translate(${x - 49},${y - 34})`}>
      <rect width="98" height="68" rx="10" className={`diag-icon-box diag-icon-box--${status}`} style={s.box} />
      <rect x="14" y="18" width="69" height="7" rx="3.5" className={`diag-icon-docline diag-icon-docline--${status}`} style={s.fill} />
      <rect x="14" y="32" width="55" height="7" rx="3.5" className={`diag-icon-docline diag-icon-docline--${status}`} style={s.fill} />
      <rect x="14" y="46" width="62" height="7" rx="3.5" className={`diag-icon-docline diag-icon-docline--${status}`} style={s.fill} />
      <DiagLabel label="통합 멀티모달 텍스트" x={49} y={90} maxWidth={ICON_MAX_WIDTH} />
    </g>
  )
}

function StackIcon({ status }) {
  const { x, y } = NODE_POS.error_output
  const s = iconStyles(status)
  return (
    <g transform={`translate(${x - 42},${y - 34})`}>
      <rect x="12" y="12" width="74" height="56" rx="10" className={`diag-stack-sheet diag-stack-sheet--back diag-icon-box--${status}`} style={s.box} />
      <rect x="6" y="6" width="74" height="56" rx="10" className={`diag-stack-sheet diag-stack-sheet--mid diag-icon-box--${status}`} style={s.box} />
      <rect width="74" height="56" rx="10" className={`diag-stack-sheet diag-icon-box--${status}`} style={s.box} />
      <rect x="12" y="17" width="47" height="6" rx="3" className={`diag-icon-docline diag-icon-docline--${status}`} style={s.fill} />
      <rect x="12" y="30" width="37" height="6" rx="3" className={`diag-icon-docline diag-icon-docline--${status}`} style={s.fill} />
      <rect x="12" y="43" width="42" height="6" rx="3" className={`diag-icon-docline diag-icon-docline--${status}`} style={s.fill} />
      <DiagLabel label="오류 산출물" x={43} y={90} maxWidth={ICON_MAX_WIDTH} />
    </g>
  )
}

const ICON_COMPONENTS = { video: VideoIcon, integrated_text: TextIcon, error_output: StackIcon }

function GroupBracket({ x1, x2, y, captionX, caption, variant }) {
  return (
    <g>
      <path
        d={`M${x1} ${y - 10} L${x1} ${y} L${x2} ${y} L${x2} ${y - 10}`}
        className={`diag-group-bracket diag-group-bracket--${variant}`}
      />
      <text x={captionX} y={y + 24} textAnchor="middle" className="diag-group-caption">{caption}</text>
    </g>
  )
}

// 레인 라벨: 그룹 브래킷과 같은 언어(작은 꺾은선)로 자기 줄의 노드 범위를 가리킨다.
// 그룹 색(전처리=주황, 검증=보라)을 그대로 물려받아 "이 레인이 어느 그룹 소속인지"까지
// 은은하게 드러낸다.
function LaneLabel({ x1, x2, labelY, underlineY, label, variant }) {
  const midX = (x1 + x2) / 2
  const tick = 5
  return (
    <g>
      <text x={midX} y={labelY} textAnchor="middle" className="diag-lane-label">{label}</text>
      <path
        d={`M${x1} ${underlineY - tick} L${x1} ${underlineY} L${x2} ${underlineY} L${x2} ${underlineY - tick}`}
        className={`diag-lane-bracket diag-lane-bracket--${variant}`}
      />
    </g>
  )
}

export default function DiagramPipeline({ status, diffLine = false, diffNode = false }) {
  return (
    <svg viewBox="0 0 1080 400" className="diag-svg" aria-hidden="true">
      <LaneLabel x1={170} x2={340} labelY={26} underlineY={38} label="영상 분석" variant="pre" />
      <LaneLabel x1={170} x2={340} labelY={210} underlineY={222} label="오디오 분석" variant="pre" />
      <LaneLabel x1={580} x2={1000} labelY={22} underlineY={34} label="발화 검증" variant="verify" />
      <LaneLabel x1={580} x2={715} labelY={210} underlineY={222} label="슬라이드 검증" variant="verify" />

      {CONNECTORS.map((c, i) => {
        const { className, style } = connectorStyle(status[c.target], c.lane, diffLine)
        return <path key={i} d={c.d} className={className} style={style} />
      })}

      {NODES.map(node => {
        if (node.icon) {
          const Icon = ICON_COMPONENTS[node.id]
          return <Icon key={node.id} status={status[node.id]} />
        }
        return (
          <PlainNode
            key={node.id}
            id={node.id}
            label={node.label}
            lane={node.lane}
            status={status[node.id]}
            diffNode={diffNode}
          />
        )
      })}

      <GroupBracket x1={140} x2={505} y={345} captionX={322} caption="① 전처리" variant="pre" />
      <GroupBracket x1={545} x2={1035} y={345} captionX={790} caption="② 검증" variant="verify" />
    </svg>
  )
}
