import { PIPELINE_NODES } from './verifierConstants'
import { DEMO_PHASES } from '../../hooks/useDemoPipelineFlow'

// /dev/verify-demo 전용 큰 삽화. 실제 검증 화면(PipelineProgress)에는 쓰이지 않는다.
// PIPELINE_NODES 9단계 전부에 전용 그림이 있고, GenericAnim은 새 단계가 추가됐을 때의
// 대체용 안전망이다. 색 언어: 청록=오디오, 보라=이미지, 골드=텍스트, 초록=완료/검증.
// 주의: SVG의 transform="translate(...)" 속성과 CSS animation의 transform은 같은
// 엘리먼트에 함께 걸면 CSS가 속성을 통째로 덮어써 위치가 틀어진다. 그래서 "위치 이동"은
// 항상 바깥 <g transform="...">, "애니메이션"은 항상 그 안의 클래스 있는 <g>에 건다.

function ImageIconGlyph() {
  return (
    <>
      <rect width="60" height="40" rx="6" className="demo-anim-box demo-anim-box--image" />
      <circle cx="16" cy="14" r="4" className="demo-anim-image-sun" />
      <path d="M8 32 L20 19 L29 27 L38 15 L52 32 Z" className="demo-anim-image-mountain" />
    </>
  )
}

function AudioIconGlyph() {
  return (
    <>
      <rect width="60" height="40" rx="6" className="demo-anim-box demo-anim-box--audio" />
      <g className="demo-anim-wave">
        <rect x="10" y="14" width="4" height="12" />
        <rect x="18" y="8" width="4" height="24" />
        <rect x="26" y="16" width="4" height="8" />
        <rect x="34" y="6" width="4" height="28" />
        <rect x="42" y="12" width="4" height="16" />
      </g>
    </>
  )
}

function TextIconGlyph() {
  return (
    <>
      <rect width="60" height="40" rx="6" className="demo-anim-box demo-anim-box--text" />
      <rect x="10" y="10" width="40" height="5" rx="2.5" className="demo-anim-doc-line" />
      <rect x="10" y="19" width="30" height="5" rx="2.5" className="demo-anim-doc-line" />
      <rect x="10" y="28" width="36" height="5" rx="2.5" className="demo-anim-doc-line" />
    </>
  )
}

function DataExtractAnim() {
  return (
    <div className="demo-anim">
      <svg viewBox="0 0 320 150" className="demo-anim-svg" aria-hidden="true">
        <g transform="translate(30,55)">
          <rect width="60" height="42" rx="6" className="demo-anim-box" />
          <polygon points="24,13 24,29 40,21" className="demo-anim-play" />
        </g>
        <path d="M95 68 C 150 68, 150 30, 230 30" className="demo-anim-line" />
        <path d="M95 84 C 150 84, 150 122, 230 122" className="demo-anim-line" />
        <g transform="translate(235,10)">
          <AudioIconGlyph />
        </g>
        <g transform="translate(235,102)">
          <ImageIconGlyph />
        </g>
      </svg>
      <p className="demo-anim-caption">영상을 오디오와 슬라이드 이미지로 분리하는 중</p>
    </div>
  )
}

function ContentExtractAnim() {
  return (
    <div className="demo-anim">
      <svg viewBox="0 0 320 150" className="demo-anim-svg" aria-hidden="true">
        <g transform="translate(20,20)">
          <AudioIconGlyph />
        </g>
        <g transform="translate(20,90)">
          <ImageIconGlyph />
        </g>
        <path d="M85 40 C 150 40, 150 75, 210 75" className="demo-anim-line" />
        <path d="M85 110 C 150 110, 150 75, 210 75" className="demo-anim-line" />
        <g transform="translate(210,30)">
          <rect width="90" height="90" rx="8" className="demo-anim-box" />
          <rect x="10" y="14" width="70" height="6" rx="3" className="demo-anim-text-line demo-anim-text-line--1" />
          <rect x="10" y="30" width="70" height="6" rx="3" className="demo-anim-text-line demo-anim-text-line--2" />
          <rect x="10" y="46" width="70" height="6" rx="3" className="demo-anim-text-line demo-anim-text-line--3" />
          <rect x="10" y="62" width="45" height="6" rx="3" className="demo-anim-text-line demo-anim-text-line--4" />
        </g>
      </svg>
      <p className="demo-anim-caption">음성·슬라이드를 텍스트로 변환하는 중</p>
    </div>
  )
}

function ContextAnalysisAnim() {
  return (
    <div className="demo-anim">
      <svg viewBox="0 0 320 150" className="demo-anim-svg" aria-hidden="true">
        {/* 텍스트 상자를 먼저 그려야 뒤이어 그리는 연결선이 그 위에 겹쳐 보인다(SVG는 그린 순서대로 쌓인다). */}
        <rect x="50" y="95" width="220" height="48" rx="8" className="demo-anim-box" />
        <rect x="64" y="106" width="90" height="5" rx="2.5" className="demo-anim-doc-line" />
        <rect x="64" y="118" width="140" height="5" rx="2.5" className="demo-anim-doc-line" />
        <rect x="64" y="130" width="110" height="5" rx="2.5" className="demo-anim-doc-line" />

        <path d="M70 32 C 75 60, 85 85, 100 106" className="demo-anim-link demo-anim-link--1" />
        <path d="M160 30 C 165 60, 180 95, 190 118" className="demo-anim-link demo-anim-link--2" />
        <path d="M250 32 C 230 70, 190 102, 150 130" className="demo-anim-link demo-anim-link--3" />
        <circle cx="100" cy="108" r="5" className="demo-anim-timeline-mark demo-anim-timeline-mark--1" />
        <circle cx="190" cy="120" r="5" className="demo-anim-timeline-mark demo-anim-timeline-mark--2" />
        <circle cx="150" cy="132" r="5" className="demo-anim-timeline-mark demo-anim-timeline-mark--3" />

        <g transform="translate(48,8)">
          <rect width="44" height="24" rx="12" className="demo-anim-tag-box" />
          <g className="demo-anim-tag-wave">
            <rect x="10" y="10" width="3" height="4" />
            <rect x="16" y="6" width="3" height="12" />
            <rect x="22" y="9" width="3" height="6" />
            <rect x="28" y="7" width="3" height="10" />
          </g>
        </g>
        <g transform="translate(138,6)">
          <rect width="44" height="24" rx="12" className="demo-anim-tag-box" />
          <g className="demo-anim-tag-wave">
            <rect x="10" y="10" width="3" height="4" />
            <rect x="16" y="6" width="3" height="12" />
            <rect x="22" y="9" width="3" height="6" />
            <rect x="28" y="7" width="3" height="10" />
          </g>
        </g>
        <g transform="translate(228,10)">
          <rect width="44" height="24" rx="12" className="demo-anim-tag-box" />
          <g className="demo-anim-tag-wave">
            <rect x="10" y="10" width="3" height="4" />
            <rect x="16" y="6" width="3" height="12" />
            <rect x="22" y="9" width="3" height="6" />
            <rect x="28" y="7" width="3" height="10" />
          </g>
        </g>
      </svg>
      <p className="demo-anim-caption">전사 텍스트에 오디오 맥락(어조·강조 등)을 연결하는 중</p>
    </div>
  )
}

function VerifierDataAnim() {
  return (
    <div className="demo-anim">
      <svg viewBox="0 0 320 165" className="demo-anim-svg" aria-hidden="true">
        <path d="M120 50 C 120 75, 130 90, 148 108" className="demo-anim-line" />
        <path d="M220 50 C 220 75, 200 90, 178 108" className="demo-anim-line" />

        <g transform="translate(90,8)">
          <TextIconGlyph />
        </g>
        <g transform="translate(190,8)">
          <ImageIconGlyph />
        </g>

        <g transform="translate(125,108)">
          <g className="demo-anim-package">
            <rect width="70" height="46" rx="8" className="demo-anim-box" />
            <rect x="10" y="10" width="36" height="4" rx="2" className="demo-anim-doc-line" />
            <rect x="10" y="18" width="26" height="4" rx="2" className="demo-anim-doc-line" />
            <rect x="10" y="26" width="30" height="4" rx="2" className="demo-anim-doc-line" />
            <circle cx="56" cy="34" r="12" className="demo-anim-package-badge" />
            <path d="M50 34 l4 4 l8 -8" className="demo-anim-package-check" />
          </g>
        </g>
      </svg>
      <p className="demo-anim-caption">텍스트(전사+맥락)와 슬라이드 이미지를 검증용 입력으로 묶는 중</p>
    </div>
  )
}

function ClaimExtractionAnim() {
  return (
    <div className="demo-anim">
      <svg viewBox="0 0 320 150" className="demo-anim-svg" aria-hidden="true">
        <g transform="translate(20,25)">
          <rect width="120" height="100" rx="8" className="demo-anim-box" />
          <rect x="12" y="14" width="96" height="6" rx="3" className="demo-anim-static-line" />
          <rect x="12" y="30" width="80" height="6" rx="3" className="demo-anim-static-line" />
          <rect x="12" y="46" width="90" height="6" rx="3" className="demo-anim-static-line" />
          <rect x="12" y="62" width="70" height="6" rx="3" className="demo-anim-static-line" />
          <rect x="12" y="78" width="85" height="6" rx="3" className="demo-anim-static-line" />
          <g transform="translate(66,20)">
            <g className="demo-anim-magnifier">
              <circle r="15" className="demo-anim-magnifier-glass" />
              <line x1="10" y1="10" x2="20" y2="20" className="demo-anim-magnifier-handle" />
            </g>
          </g>
        </g>

        <path d="M150 75 L 200 75" className="demo-anim-line" />

        <g transform="translate(210,45)">
          <g className="demo-anim-claim-card">
            <rect width="95" height="60" rx="8" className="demo-anim-box demo-anim-claim-box" />
            <rect x="10" y="12" width="70" height="6" rx="3" className="demo-anim-doc-line" />
            <rect x="10" y="26" width="55" height="6" rx="3" className="demo-anim-doc-line" />
            <rect x="10" y="40" width="40" height="14" rx="7" className="demo-anim-claim-tag" />
          </g>
        </g>
      </svg>
      <p className="demo-anim-caption">돋보기로 전사를 훑으며 검증할 주장 문장을 찾아내는 중</p>
    </div>
  )
}

function IssueJudgeAnim() {
  return (
    <div className="demo-anim">
      <svg viewBox="0 0 320 150" className="demo-anim-svg" aria-hidden="true">
        <g transform="translate(130,15)">
          <rect width="60" height="40" rx="8" className="demo-anim-box" />
          <rect x="8" y="10" width="44" height="5" rx="2.5" className="demo-anim-doc-line" />
          <rect x="8" y="22" width="30" height="5" rx="2.5" className="demo-anim-doc-line" />
        </g>
        <g transform="translate(184,4)">
          <circle cx="12" cy="12" r="12" className="demo-anim-magnifier-glass" />
          <line x1="20" y1="20" x2="29" y2="29" className="demo-anim-magnifier-handle" />
        </g>

        <line x1="160" y1="55" x2="160" y2="80" className="demo-anim-scale-stem" />
        <g transform="translate(160,80)">
          <g className="demo-anim-scale">
            <line x1="-55" y1="0" x2="55" y2="0" className="demo-anim-scale-beam" />
            <line x1="-55" y1="0" x2="-55" y2="18" className="demo-anim-scale-rope" />
            <line x1="55" y1="0" x2="55" y2="18" className="demo-anim-scale-rope" />
            <circle cx="-55" cy="24" r="12" className="demo-anim-scale-pan" />
            <circle cx="55" cy="24" r="12" className="demo-anim-scale-pan" />
            <path d="M-60 25 l4 3 l7 -9" className="demo-anim-judge-check" />
            <path d="M50 19 L60 29 M60 19 L50 29" className="demo-anim-judge-flag" />
          </g>
        </g>
      </svg>
      <p className="demo-anim-caption">돋보기로 살펴보고 이슈 후보인지 저울질하는 중</p>
    </div>
  )
}

function IssueClassificationAnim() {
  return (
    <div className="demo-anim">
      <svg viewBox="0 0 320 150" className="demo-anim-svg" aria-hidden="true">
        <path d="M152 24 C 130 55, 100 80, 85 104" className="demo-anim-line" />
        <path d="M168 24 C 190 55, 220 80, 235 104" className="demo-anim-line" />

        <rect x="30" y="106" width="110" height="34" rx="8" className="demo-anim-bin demo-anim-bin--a" />
        <text x="85" y="127" textAnchor="middle" className="demo-anim-bin-label">사실 오류</text>
        <rect x="180" y="106" width="110" height="34" rx="8" className="demo-anim-bin demo-anim-bin--b" />
        <text x="235" y="127" textAnchor="middle" className="demo-anim-bin-label">혼동 설명</text>

        <path d="M144 8 L176 8 L164 26 L156 26 Z" className="demo-anim-funnel" />
        <rect x="152" y="26" width="16" height="12" rx="3" className="demo-anim-chip demo-anim-chip--a" />
        <rect x="152" y="26" width="16" height="12" rx="3" className="demo-anim-chip demo-anim-chip--b" />
      </svg>
      <p className="demo-anim-caption">이슈를 깔때기로 걸러 유형별 통으로 분류하는 중</p>
    </div>
  )
}

function FinalVerificationAnim() {
  return (
    <div className="demo-anim">
      <svg viewBox="0 0 320 165" className="demo-anim-svg" aria-hidden="true">
        <path d="M86 83 C 105 83, 115 40, 140 21" className="demo-anim-line" />
        <path d="M86 83 C 100 83, 115 72, 140 65" className="demo-anim-line" />
        <path d="M86 83 C 105 83, 115 125, 140 139" className="demo-anim-line" />
        <path d="M174 21 C 220 21, 245 45, 268 68" className="demo-anim-line" />
        <path d="M174 65 C 210 65, 240 72, 266 78" className="demo-anim-line" />
        <path d="M174 139 C 220 139, 245 116, 268 92" className="demo-anim-line" />

        <g transform="translate(8,58)">
          <rect width="78" height="50" rx="8" className="demo-anim-box" />
          <rect x="9" y="10" width="58" height="5" rx="2.5" className="demo-anim-doc-line" />
          <rect x="9" y="21" width="46" height="5" rx="2.5" className="demo-anim-doc-line" />
          <rect x="9" y="32" width="52" height="5" rx="2.5" className="demo-anim-doc-line" />
        </g>

        <g transform="translate(140,4)" className="demo-anim-model demo-anim-model--1">
          <circle r="17" cx="17" cy="17" className="demo-anim-model-circle" />
          <text x="17" y="21" textAnchor="middle" className="demo-anim-model-label">LLM1</text>
          <path d="M9 17 l5 5 l9 -9" className="demo-anim-check" />
        </g>
        <g transform="translate(140,48)" className="demo-anim-model demo-anim-model--2">
          <circle r="17" cx="17" cy="17" className="demo-anim-model-circle" />
          <text x="17" y="21" textAnchor="middle" className="demo-anim-model-label">LLM2</text>
          <path d="M9 17 l5 5 l9 -9" className="demo-anim-check" />
        </g>
        {/* LLM이 더 있을 수 있다는 뜻의 세로 점 3개 */}
        <circle cx="157" cy="93" r="1.8" className="demo-anim-ellipsis-dot" />
        <circle cx="157" cy="100" r="1.8" className="demo-anim-ellipsis-dot" />
        <circle cx="157" cy="107" r="1.8" className="demo-anim-ellipsis-dot" />
        <g transform="translate(140,122)" className="demo-anim-model demo-anim-model--3">
          <circle r="17" cx="17" cy="17" className="demo-anim-model-circle" />
          <text x="17" y="21" textAnchor="middle" className="demo-anim-model-label">LLMn</text>
          <path d="M9 17 l5 5 l9 -9" className="demo-anim-check" />
        </g>

        <g transform="translate(280,80)">
          <circle r="15" className="demo-anim-magnifier-glass" />
          <line x1="10" y1="10" x2="20" y2="20" className="demo-anim-magnifier-handle" />
          <path d="M-7 0 l5 5 l9 -10" className="demo-anim-result-check" />
        </g>
      </svg>
      <p className="demo-anim-caption">텍스트 주장을 LLM1·2·n이 검증해 하나의 결과로 도출하는 중</p>
    </div>
  )
}

function SlideReviewAnim() {
  return (
    <div className="demo-anim">
      <svg viewBox="0 0 320 160" className="demo-anim-svg" aria-hidden="true">
        <g transform="translate(70,15)">
          <rect width="180" height="120" rx="8" className="demo-anim-box demo-anim-box--slide" />
          <rect x="16" y="16" width="120" height="8" rx="4" className="demo-anim-slide-line" />
          <rect x="16" y="34" width="90" height="8" rx="4" className="demo-anim-slide-line" />
          <rect x="16" y="52" width="100" height="8" rx="4" className="demo-anim-slide-line" />
          <rect x="16" y="80" width="60" height="30" rx="4" className="demo-anim-slide-block" />
          <rect x="90" y="80" width="60" height="30" rx="4" className="demo-anim-slide-block" />

          <rect x="8" y="8" width="14" height="104" className="demo-anim-scan" />
          <rect x="8" y="8" width="3" height="104" className="demo-anim-scan-edge" />
          <rect x="86" y="76" width="68" height="38" rx="6" className="demo-anim-error-box" />
          <text x="120" y="100" textAnchor="middle" className="demo-anim-error-badge">!</text>
        </g>
      </svg>
      <p className="demo-anim-caption">스캔 빔이 슬라이드를 훑으며 오류를 찾는 중</p>
    </div>
  )
}

function GenericAnim({ label }) {
  return (
    <div className="demo-anim">
      <svg viewBox="0 0 120 120" className="demo-anim-svg demo-anim-svg--sm" aria-hidden="true">
        <circle cx="60" cy="60" r="40" className="demo-anim-generic-ring" />
      </svg>
      <p className="demo-anim-caption">{label} 진행 중 (전용 애니메이션 준비 중)</p>
    </div>
  )
}

function DoneAnim() {
  return (
    <div className="demo-anim">
      <svg viewBox="0 0 120 120" className="demo-anim-svg demo-anim-svg--sm" aria-hidden="true">
        <circle cx="60" cy="60" r="44" className="demo-anim-done-ring" />
        <path d="M38 62 l16 16 l30 -34" className="demo-anim-done-check" />
      </svg>
      <p className="demo-anim-caption">모든 단계 완료</p>
    </div>
  )
}

function ErrorAnim() {
  return (
    <div className="demo-anim">
      <svg viewBox="0 0 120 120" className="demo-anim-svg demo-anim-svg--sm demo-anim-svg--shake" aria-hidden="true">
        <circle cx="60" cy="60" r="44" className="demo-anim-error-ring" />
        <line x1="60" y1="36" x2="60" y2="68" className="demo-anim-error-mark" />
        <circle cx="60" cy="84" r="3" className="demo-anim-error-mark demo-anim-error-dot" />
      </svg>
      <p className="demo-anim-caption">오류 발생 (데모)</p>
    </div>
  )
}

const CUSTOM_ANIMS = {
  data_extract: DataExtractAnim,
  content_extract: ContentExtractAnim,
  context_analysis: ContextAnalysisAnim,
  verifier_data: VerifierDataAnim,
  claim_extraction: ClaimExtractionAnim,
  issue_judge: IssueJudgeAnim,
  issue_classification: IssueClassificationAnim,
  final_verification: FinalVerificationAnim,
  slide_review: SlideReviewAnim,
}

export default function DemoStageAnimation({ stageId, phase }) {
  if (phase === DEMO_PHASES.DONE) return <DoneAnim />
  if (phase === DEMO_PHASES.ERROR) return <ErrorAnim />

  const Custom = CUSTOM_ANIMS[stageId]
  if (Custom) return <Custom />

  const label = PIPELINE_NODES.find(node => node.id === stageId)?.label || '단계'
  return <GenericAnim label={label} />
}
