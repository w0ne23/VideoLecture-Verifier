// 검증 결과 화면 — 총 오류 개수 + 유형 필터/정렬 + 지식 오류·슬라이드 오류 카드 목록
// verifier: GET /lectures/{id}/result 응답 (final_confirmed_claims / needs_review_claims / slide_errors)

import { useMemo, useState } from 'react'

// --- 라벨 매핑 (백엔드 enum → 한글) ---------------------------------------------

const TYPE_LABELS = {
  factual_error: '사실 오류',
  temporal_error: '오래된 내용',
  scope_overclaim: '과도한 일반화',
  confusing_explanation: '혼동 가능 설명',
  composite_issue: '슬라이드 오류',
}

const MODEL_DISPLAY_NAMES = {
  gpt: 'gpt-5.4',
  claude: 'claude-sonnet-5',
  grok: 'grok-4.5',
}

const STATUS_LABELS = {
  confirmed: '확정',
  professor_check: '검토 필요',
  review_needed: '검토 필요',
  rejected: '기각',
  supports_issue: '이슈 근거 있음',
  refutes_issue: '이슈 반박 근거',
  verified: '웹 근거 확인',
  insufficient_evidence: '근거 부족',
  grounding_unavailable: '웹 근거 확인 실패',
  not_applicable: '대상 아님',
}

// 파이프라인이 웹 근거 검색을 수행하는 유형 (classified_issue_grounder.GROUNDABLE_CATEGORIES)
// 나머지 유형(과도한 일반화·혼동 설명 등)은 웹 검색 대상 아님
const GROUNDABLE_TYPES = new Set(['factual_error', 'temporal_error'])

// 지식 오류 4개 카테고리 + 슬라이드 오류 = 5개 필터 버튼
const KNOWLEDGE_CATEGORY_KEYS = ['factual_error', 'temporal_error', 'scope_overclaim', 'confusing_explanation']
const CATEGORY_DEFS = [
  { key: 'factual_error', label: '사실 오류' },
  { key: 'temporal_error', label: '오래된 내용' },
  { key: 'scope_overclaim', label: '과도한 일반화' },
  { key: 'confusing_explanation', label: '혼동 가능 설명' },
  { key: 'slide', label: '슬라이드 오류' },
]
const SLIDE_ERROR_TYPE_LABELS = {
  text_error: '철자/표기 오류',
  numeric_unit: '숫자/단위 표기 오류',
  code_syntax: '코드/수식 문법 오류',
  visual_defect: '이미지 깨짐·텍스트 겹침 등 시각적 결함',
}

const SORT_OPTIONS = [
  { key: 'time', label: '시간순' },
  { key: 'severity', label: '심각도순' },
]

// --- 범용 유틸 ----------------------------------------------------------------

// 조건부 클래스명 결합 (falsy 제거)
function cx(...classNames) {
  return classNames.filter(Boolean).join(' ')
}

// 값을 항상 배열로 (단일 값은 1개짜리 배열, falsy 는 빈 배열)
function asArray(value) {
  if (!value) return []
  return Array.isArray(value) ? value : [value]
}

// keys 를 순서대로 보고 처음 만나는 비어있지 않은 문자열 반환
function pick(item, keys) {
  for (const key of keys) {
    const value = item?.[key]
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  return ''
}

// keys 를 순서대로 보고 처음 만나는 유한수 반환
function pickNumber(item, keys) {
  for (const key of keys) {
    const value = Number(item?.[key])
    if (Number.isFinite(value)) return value
  }
  return null
}

function compactText(value, fallback = '') {
  if (value === undefined || value === null) return fallback
  const text = String(value).replace(/\s+/g, ' ').trim()
  return text || fallback
}

function formatTime(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0))
  const hh = Math.floor(total / 3600)
  const mm = Math.floor((total % 3600) / 60)
  const ss = total % 60
  if (hh > 0) return `${hh}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`
  return `${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`
}

// 점수 값(0~1 또는 0~100) → 백분율 문자열 (10 이상은 소수 1자리, 미만은 2자리)
function formatScore(value) {
  const num = Number(value)
  if (!Number.isFinite(num)) return ''
  const percent = num <= 1 ? num * 100 : num
  return percent.toFixed(percent >= 10 ? 1 : 2)
}

function formatPoint(value) {
  const num = Number(value)
  if (!Number.isFinite(num)) return ''
  return num.toFixed(num >= 10 ? 1 : 2)
}

function formatRatio(value) {
  const num = Number(value)
  if (!Number.isFinite(num)) return ''
  return `${(num * 100).toFixed(1)}%`
}

function labelForType(key) {
  return TYPE_LABELS[key] || key || ''
}

function normalizeCandidateKey(candidate) {
  if (typeof candidate === 'string') return candidate
  return candidate?.category || candidate?.issue_type || candidate?.type || candidate?.key || ''
}

// --- 항목 해석 헬퍼 (백엔드 응답 형태가 유형마다 달라 필드 후보를 넓게 훑음) --------

// 복합 오류(여러 유형이 얽힌 오류) 여부
function isCompositeItem(item) {
  const type = item.feedback_type || item.category || item.issue_type || item.type
  return type === 'composite_issue' || item.scored_as_composite || item.classified_issue_verifier?.scored_as_composite
}

// 항목이 속하는 카테고리 목록
// 단일 유형이면 그 유형 하나, 복합 오류면 구성 후보 카테고리 전부 (태그·필터 매칭에 공용)
function itemCategories(item) {
  const type = item.feedback_type || item.category || item.issue_type || item.type
  if (!isCompositeItem(item)) {
    return type ? [type] : []
  }

  const verifier = item.classified_issue_verifier || {}
  const scoring = item.composite_scoring || verifier.composite_scoring || {}
  const candidateKeys = [
    ...asArray(item.composite_candidate_categories),
    ...asArray(verifier.composite_candidate_categories),
    ...Object.keys(scoring.normalized_probabilities || {}),
    ...Object.keys(scoring.raw_probabilities || {}),
  ]
    .map(normalizeCandidateKey)
    .filter(Boolean)

  const known = [...new Set(candidateKeys)].filter(key => KNOWLEDGE_CATEGORY_KEYS.includes(key))
  return known.length ? known : ['composite_issue']
}

// 지식 오류 최종 심각도 (%) — percent 필드 우선, 없으면 score(0~1)에 100 곱함
function getSeverity(item) {
  const verifier = item.classified_issue_verifier || {}
  const percent = pickNumber(
    {
      severity_score_percent: item.severity_score_percent,
      final_severity_percent: item.final_severity_percent,
      verifier_final_severity_percent: verifier.final_severity_percent,
    },
    ['severity_score_percent', 'final_severity_percent', 'verifier_final_severity_percent']
  )
  if (percent != null) return percent
  const score = pickNumber(
    {
      severity_score: item.severity_score,
      final_severity_score: verifier.final_severity_score,
    },
    ['severity_score', 'final_severity_score']
  )
  return score == null ? null : score * 100
}

// 슬라이드 오류 심각도 (%)
function getSlideSeverity(item) {
  const num = Number(item.severity_score)
  if (!Number.isFinite(num)) return null
  return num <= 1 ? num * 100 : num
}

// 웹 근거 객체 추출 — 응답 위치가 여러 곳이라 후보를 순서대로 확인
function getGrounding(item) {
  const evidence = item.evidence || {}
  const verifier = item.classified_issue_verifier || {}
  return (
    verifier.web_evidence
    || evidence.web_evidence
    || item.web_evidence
    || verifier.web_grounding
    || item.web_grounding
    || evidence.web_grounding
    || {}
  )
}

function sourceUrl(source) {
  if (typeof source === 'string') return source
  return source?.url || source?.source_url || ''
}

function sourceLabel(source, index) {
  if (typeof source === 'string') return source
  return source?.title || source?.domain || source?.url || source?.source_url || `근거 ${index + 1}`
}

// 백엔드 storage 경로 → nginx 가 서빙하는 /files/ URL
function fileUrlFromStoragePath(path) {
  if (!path) return ''
  const value = String(path).replace(/\\/g, '/')
  if (value.startsWith('http://') || value.startsWith('https://') || value.startsWith('/files/')) return value
  if (value.startsWith('storage/')) return `/files/${value.slice('storage/'.length)}`
  const marker = '/storage/'
  const index = value.indexOf(marker)
  if (index >= 0) return `/files/${value.slice(index + marker.length)}`
  return ''
}

// 항목의 위치 정보 (슬라이드 번호 + 영상 재생 시각)
function locationOf(item) {
  const location = item.location || {}
  return {
    slideNumber: item.slide_number || location.slide_number,
    startTime: pickNumber({ ...item, ...location }, ['start_time', 'start', 'timestamp']),
    endTime: pickNumber({ ...item, ...location }, ['end_time', 'end']),
  }
}

function timeRangeLabel(startTime, endTime) {
  if (startTime == null) return ''
  if (endTime == null || endTime <= startTime) return formatTime(startTime)
  return `${formatTime(startTime)} - ${formatTime(endTime)}`
}

// --- 표시용 하위 컴포넌트 ----------------------------------------------------

// 정의 목록 한 행 — value 가 비면 렌더 안 함
function DetailRow({ label, value, wide = false }) {
  if (value === undefined || value === null || value === '') return null
  return (
    <div className={cx('claim-detail-row', wide && 'claim-detail-row--wide')}>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  )
}

// 긴 근거 문단(모델별 판단·웹 검색 결과)을 1~2줄로 자르고, 클릭하면 펼침
function ClampedText({ children, lines = 2 }) {
  const [expanded, setExpanded] = useState(false)
  if (!children) return null
  return (
    <p
      className={cx('claim-detail-text', 'clamped-text-clickable', !expanded && 'clamped-text')}
      style={!expanded ? { WebkitLineClamp: lines } : undefined}
      onClick={() => setExpanded(prev => !prev)}
      role="button"
      tabIndex={0}
      title={expanded ? '접으려면 클릭' : '전체 보기'}
    >
      {children}
    </p>
  )
}

function DetailGroup({ title, children, noDivider = false }) {
  if (!children) return null
  return (
    <div className={cx('claim-detail-group', noDivider && 'claim-detail-group--no-divider')}>
      <h4>{title}</h4>
      {children}
    </div>
  )
}

// LLM 이 "원문 → 수정문" 형태로 쓰면 화살표 뒤 수정문만 남김 (원문은 '원 발화' 행에 이미 있음)
function stripArrowPrefix(text) {
  if (!text) return text
  const idx = text.indexOf('→')
  return idx === -1 ? text : text.slice(idx + 1).trim()
}

// 복합 오류의 세부 유형별 확률·점수·기여도 표
function CompositeScoringPanel({ item }) {
  const verifier = item.classified_issue_verifier || {}
  const scoring = item.composite_scoring || verifier.composite_scoring
  if (!scoring) return null

  const normalized = scoring.normalized_probabilities || {}
  const scores = scoring.candidate_scores || {}
  const contributions = scoring.candidate_contributions || {}
  const keys = Object.keys(normalized).length
    ? Object.keys(normalized)
    : [...new Set([...Object.keys(scores), ...Object.keys(contributions)])]

  return (
    <DetailGroup title="복합 오류 점수" noDivider>
      <div className="composite-summary">
        <span>방식: {scoring.method || 'weighted_expected_severity'}</span>
        {scoring.primary_issue_type && <span>대표 유형: {labelForType(scoring.primary_issue_type)}</span>}
        {scoring.weighted_score !== undefined && <span>가중 점수: {formatScore(scoring.weighted_score)}</span>}
      </div>
      {keys.length > 0 && (
        <div className="composite-table">
          <div className="composite-row composite-row--head">
            <span>세부 유형</span>
            <span>보정 확률</span>
            <span>검증 점수</span>
            <span>기여도</span>
          </div>
          {keys.map(key => (
            <div className="composite-row" key={key}>
              <span>{labelForType(key)}</span>
              <span>{formatRatio(normalized[key])}</span>
              <span>{formatScore(scores[key])}</span>
              <span>{formatScore(contributions[key])}</span>
            </div>
          ))}
        </div>
      )}
    </DetailGroup>
  )
}

// 웹 근거로 실제 판정이 난 상태 — 나머지(근거 부족·검증 실패·빈 객체)는 모두 "판정 못 함" 하나로 합침
const GROUNDING_CONCLUSIVE = new Set(['verified', 'supports_issue', 'refutes_issue'])

// 웹 검색 결과 패널 — 3상태: 판정 남 / 판정 못 함(대상 유형) / 대상 아님(비대상 유형)
function WebGroundingPanel({ item }) {
  const grounding = getGrounding(item)
  const evidence = item.evidence || {}
  const verifier = item.classified_issue_verifier || {}

  // 웹 근거 수집 단계 자체가 안 돈 경우(파이프라인 비활성 등)만 섹션을 숨김
  const evidenceRan =
    Object.keys(grounding).length > 0 ||
    'web_evidence' in evidence || 'web_evidence' in item || 'web_evidence' in verifier
  if (!evidenceRan) return null

  const sources = grounding.evidence_sources?.length ? grounding.evidence_sources : evidence.web_sources
  const sourceItems = asArray(sources).filter(sourceUrl)
  const reasonText = grounding.reason || grounding.evidence_summary || ''

  const conclusive =
    GROUNDING_CONCLUSIVE.has(grounding.status) || sourceItems.length > 0 || !!reasonText

  if (conclusive) {
    const statusLabel = STATUS_LABELS[grounding.status] || grounding.status || ''
    const sourceValue = sourceItems.length > 0
      ? sourceItems.map((source, index) => {
          const url = sourceUrl(source)
          return (
            <span key={`${url}-${index}`}>
              {index > 0 && ' · '}
              <a href={url} target="_blank" rel="noreferrer">{sourceLabel(source, index)}</a>
            </span>
          )
        })
      : null
    return (
      <DetailGroup title="웹 검색 결과" noDivider>
        <div className="grounding-card">
          <dl className="claim-detail-list">
            <DetailRow label="검증 상태" value={!reasonText && !sourceValue ? statusLabel : null} wide />
            <DetailRow label="판정 이유" value={reasonText ? <ClampedText lines={2}>{reasonText}</ClampedText> : null} wide />
            <DetailRow label="근거 출처" value={sourceValue ? <ClampedText lines={1}>{sourceValue}</ClampedText> : null} wide />
          </dl>
        </div>
      </DetailGroup>
    )
  }

  // 판정 못 난 경우 — 대상 유형이면 "판정 불가", 비대상 유형이면 "대상 아님"
  const groundable = itemCategories(item).some(cat => GROUNDABLE_TYPES.has(cat))
  return (
    <DetailGroup title="웹 검색 결과" noDivider>
      <div className="grounding-card grounding-card--na">
        <p className="claim-detail-text">
          {groundable
            ? '웹 근거로는 판정하지 못했습니다.'
            : `${STATUS_LABELS.not_applicable} — 이 유형은 웹 검색 대상이 아닙니다.`}
        </p>
      </div>
    </DetailGroup>
  )
}

// 모델별 판단 목록 (model_judgments 우선, 없으면 severity 체크의 model_results 사용)
function ModelJudgments({ item }) {
  const verifier = item.classified_issue_verifier || {}
  const judgments = asArray(verifier.model_judgments || item.model_judgments)
  const modelResults = item.checks?.severity?.model_results
  const results = judgments.length
    ? judgments
    : Object.entries(modelResults || {}).map(([model, result]) => ({ model, ...result }))

  if (!results.length) return null
  return (
    <DetailGroup title="모델별 판단" noDivider>
      <div className="model-judgment-list">
        {results.map((judgment, index) => (
          <div className="model-judgment" key={`${judgment.model || 'model'}-${judgment.category || ''}-${index}`}>
            <div className="model-judgment-name">
              <strong>{MODEL_DISPLAY_NAMES[judgment.model] || judgment.model || judgment.provider || `model ${index + 1}`}</strong>
              <div className="model-judgment-meta">
                {judgment.category && <span>{labelForType(judgment.category)}</span>}
                {judgment.judgment && <span>{judgment.judgment}</span>}
                {judgment.final_model_score !== undefined && <span>{formatScore(judgment.final_model_score)}</span>}
              </div>
            </div>
            <ClampedText lines={2}>{judgment.reason || judgment.explanation || judgment.summary}</ClampedText>
          </div>
        ))}
      </div>
    </DetailGroup>
  )
}

// 지식 오류 카드 펼침 영역 — 문제 요약 + 복합 점수 + 모델별 판단 + 웹 검색 결과
function ClaimDetail({ item }) {
  const problem = item.problem || {}
  const correctInfo = problem.correct_info

  return (
    <div className="claim-detail-body">
      <DetailGroup title="문제 요약">
        <div className="claim-detail-card">
          <dl className="claim-detail-list">
            <DetailRow label="원 발화" value={item.claim_text} wide />
            <DetailRow label="수정 제안" value={stripArrowPrefix(correctInfo)} wide />
          </dl>
        </div>
      </DetailGroup>
      <CompositeScoringPanel item={item} />
      <ModelJudgments item={item} />
      <WebGroundingPanel item={item} />
    </div>
  )
}

// 지식 오류 카드 — 요약 행(유형 칩·슬라이드·시각·심각도) + 펼침 상세
// onSeek(startTime): 시각 버튼 클릭 시 영상 해당 지점으로 이동
function ClaimCard({ item, index, categories, onSeek }) {
  const [open, setOpen] = useState(false)
  const claimText = pick(item, ['resolved_claim', 'claim_text', 'claim', 'statement', 'text', 'content'])
  const severity = getSeverity(item)
  const { slideNumber, startTime, endTime } = locationOf(item)
  const timeLabel = timeRangeLabel(startTime, endTime)
  const primaryCategory = categories[0] || 'factual_error'

  const toggle = () => setOpen(prev => !prev)
  const handleKeyDown = event => {
    if (event.key !== 'Enter' && event.key !== ' ') return
    event.preventDefault()
    toggle()
  }

  return (
    <article className={cx(`claim-card claim-card--${primaryCategory}`, open && 'claim-card--open')}>
      <div
        className="claim-card-summary"
        role="button"
        tabIndex={0}
        aria-expanded={open}
        onClick={toggle}
        onKeyDown={handleKeyDown}
      >
        <div className="claim-card-main">
          <div className="claim-card-head">
            <span className="claim-card-index">#{index + 1}</span>
            {categories.map(catKey => (
              <span key={catKey} className={cx('claim-tag', 'claim-tag--type', `claim-tag--type-${catKey}`)}>
                {labelForType(catKey)}
              </span>
            ))}
            {slideNumber && <span className="claim-location-chip">slide {slideNumber}</span>}
            {startTime != null && (
              <button
                type="button"
                className="claim-time"
                onClick={event => {
                  event.stopPropagation()
                  onSeek?.(startTime)
                }}
              >
                ▶ {timeLabel}
              </button>
            )}
          </div>
          {claimText && <p className="claim-text">{claimText}</p>}
        </div>
        <div className="claim-card-side">
          {severity != null && (
            <div className="claim-card-score">
              <strong>{formatPoint(severity)}</strong>
            </div>
          )}
          <span className={cx('claim-card-toggle', open && 'claim-card-toggle--open')} aria-hidden="true">▾</span>
        </div>
      </div>
      {open && (
        <div className="claim-detail">
          <ClaimDetail item={item} />
        </div>
      )}
    </article>
  )
}

// 슬라이드 오류 카드 — 요약 행(오류 유형·슬라이드·원문→수정문) + 펼침 상세(슬라이드 이미지 포함)
function SlideErrorCard({ item, index }) {
  const [open, setOpen] = useState(false)
  const imageUrl = item.slide_image_url || item.image_url || fileUrlFromStoragePath(item.slide_image_path || item.image_path)
  const slideTypeKey = item.error_type || 'slide_error'
  const toggle = () => setOpen(prev => !prev)
  const handleKeyDown = event => {
    if (event.key !== 'Enter' && event.key !== ' ') return
    event.preventDefault()
    toggle()
  }

  return (
    <article className={cx('claim-card claim-card--slide slide-error-card', open && 'claim-card--open')}>
      <div
        className="claim-card-summary"
        role="button"
        tabIndex={0}
        aria-expanded={open}
        onClick={toggle}
        onKeyDown={handleKeyDown}
      >
        <div className="claim-card-main">
          <div className="claim-card-head">
            <span className="claim-card-index">#{index + 1}</span>
            <span className={cx('claim-tag', 'claim-tag--type', `claim-tag--type-${slideTypeKey}`)}>{SLIDE_ERROR_TYPE_LABELS[slideTypeKey] || '슬라이드 오류'}</span>
            {item.slide_number && <span className="claim-location-chip">slide {item.slide_number}</span>}
          </div>
          <p className="slide-error-change slide-error-change--compact">
            <span>{compactText(item.problematic_text, '-')}</span>
            <strong>→</strong>
            <span>{compactText(item.corrected_text, '-')}</span>
          </p>
        </div>
        <div className="claim-card-side">
          {item.severity_score !== undefined && (
            <div className="claim-card-score">
              <strong>{formatScore(item.severity_score)}</strong>
            </div>
          )}
          <span className={cx('claim-card-toggle', open && 'claim-card-toggle--open')} aria-hidden="true">▾</span>
        </div>
      </div>
      {open && (
        <div className="slide-error-detail">
          {imageUrl && (
            <div className="slide-error-image slide-error-image--large">
              <img src={imageUrl} alt={`slide ${item.slide_number || index + 1}`} loading="lazy" />
            </div>
          )}
          <dl className="claim-detail-list slide-error-detail-list">
            <DetailRow label="슬라이드 번호" value={item.slide_number} />
            <DetailRow label="슬라이드 제목" value={item.slide_title} wide />
            <DetailRow label="오류 유형" value={SLIDE_ERROR_TYPE_LABELS[item.error_type] || item.error_type} />
            <DetailRow label="판단 이유" value={item.reason} wide />
          </dl>
        </div>
      )}
    </article>
  )
}

// --- 목록 정렬 --------------------------------------------------------------

// 지식 오류 + 슬라이드 오류를 정렬용 공통 엔트리로 변환
function buildEntries(knowledgeItems, slideErrors) {
  const claimEntries = knowledgeItems.map((item, sourceIndex) => ({
    kind: 'claim',
    item,
    sourceIndex,
    categories: itemCategories(item),
    severity: getSeverity(item),
    // 지식 오류의 정렬 키 = 영상 재생 시각(초)
    time: locationOf(item).startTime,
  }))
  const slideEntries = slideErrors.map((item, sourceIndex) => ({
    kind: 'slide',
    item,
    sourceIndex,
    categories: ['slide'],
    severity: getSlideSeverity(item),
    // 슬라이드 오류의 정렬 키 = 슬라이드 번호 (초 단위 시각 없음, 척도 다름)
    time: item.slide_number != null ? Number(item.slide_number) : null,
  }))
  return [...claimEntries, ...slideEntries]
}

// 지식 오류(초)와 슬라이드 오류(슬라이드 번호)는 척도가 달라 직접 비교 불가
// 각 그룹 안에서 0~1 상대 위치로 정규화한 뒤 섞어야 하나의 시간순 목록처럼 보임
function buildTimeRank(entries) {
  const claimTimes = entries.filter(entry => entry.kind === 'claim' && entry.time != null).map(entry => entry.time)
  const slideTimes = entries.filter(entry => entry.kind === 'slide' && entry.time != null).map(entry => entry.time)
  const claimMin = claimTimes.length ? Math.min(...claimTimes) : 0
  const claimMax = claimTimes.length ? Math.max(...claimTimes) : 0
  const slideMin = slideTimes.length ? Math.min(...slideTimes) : 0
  const slideMax = slideTimes.length ? Math.max(...slideTimes) : 0

  return entry => {
    if (entry.time == null) return null
    if (entry.kind === 'slide') return slideMax === slideMin ? 0 : (entry.time - slideMin) / (slideMax - slideMin)
    return claimMax === claimMin ? 0 : (entry.time - claimMin) / (claimMax - claimMin)
  }
}

// mode: 'severity'(심각도 내림차순) | 'time'(정규화 시간순) — 값 없는 항목은 뒤로
function sortEntries(entries, mode) {
  const sorted = [...entries]
  if (mode === 'severity') {
    sorted.sort((a, b) => {
      if (a.severity == null && b.severity == null) return 0
      if (a.severity == null) return 1
      if (b.severity == null) return -1
      return b.severity - a.severity
    })
  } else {
    const rankOf = buildTimeRank(sorted)
    sorted.sort((a, b) => {
      const ar = rankOf(a)
      const br = rankOf(b)
      if (ar == null && br == null) return 0
      if (ar == null) return 1
      if (br == null) return -1
      return ar - br
    })
  }
  return sorted
}

// --- 화면 컴포넌트 ---------------------------------------------------------

// 유형 필터 버튼 + 정렬 토글 + 카드 목록
function IssueExplorer({ knowledgeItems, slideErrors, onSeek }) {
  const [activeCategory, setActiveCategory] = useState(null)
  const [sortMode, setSortMode] = useState('time')

  const entries = useMemo(() => buildEntries(knowledgeItems, slideErrors), [knowledgeItems, slideErrors])

  const counts = useMemo(() => {
    const next = Object.fromEntries(CATEGORY_DEFS.map(def => [def.key, 0]))
    entries.forEach(entry => {
      entry.categories.forEach(key => {
        if (next[key] !== undefined) next[key] += 1
      })
    })
    return next
  }, [entries])

  const filtered = useMemo(
    () => (activeCategory ? entries.filter(entry => entry.categories.includes(activeCategory)) : entries),
    [entries, activeCategory]
  )
  const sorted = useMemo(() => sortEntries(filtered, sortMode), [filtered, sortMode])

  return (
    <div className="issue-explorer">
      <div className="issue-filter-bar" role="group" aria-label="오류 유형 필터">
        {CATEGORY_DEFS.map(def => (
          <button
            key={def.key}
            type="button"
            className={cx(
              'issue-filter-btn',
              `issue-filter-btn--${def.key}`,
              activeCategory === def.key && 'issue-filter-btn--active'
            )}
            aria-pressed={activeCategory === def.key}
            onClick={() => setActiveCategory(prev => (prev === def.key ? null : def.key))}
          >
            <span className="issue-filter-count">{counts[def.key]}</span>
            <span className="issue-filter-label">{def.label}</span>
          </button>
        ))}
      </div>

      <div className="issue-list-toolbar">
        <span className="issue-list-count">{sorted.length}건</span>
        <div className="issue-sort-group" role="radiogroup" aria-label="정렬 기준">
          {SORT_OPTIONS.map(option => (
            <button
              key={option.key}
              type="button"
              role="radio"
              aria-checked={sortMode === option.key}
              className={cx('issue-sort-btn', sortMode === option.key && 'issue-sort-btn--active')}
              onClick={() => setSortMode(option.key)}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      {sorted.length === 0 ? (
        <p className="result-empty">항목이 없습니다.</p>
      ) : (
        <div className="claim-list">
          {sorted.map(entry =>
            entry.kind === 'slide'
              ? (
                <SlideErrorCard
                  key={entry.item.slide_error_id || `slide-${entry.sourceIndex}`}
                  item={entry.item}
                  index={entry.sourceIndex}
                />
              )
              : (
                <ClaimCard
                  key={entry.item.feedback_id || entry.item.issue_id || `claim-${entry.sourceIndex}`}
                  item={entry.item}
                  index={entry.sourceIndex}
                  categories={entry.categories}
                  onSeek={onSeek}
                />
              )
          )}
        </div>
      )}
    </div>
  )
}

export default function VerifierResults({ verifier, onSeek }) {
  if (!verifier) return null

  const confirmed = verifier.final_confirmed_claims || []
  const needsReview = verifier.needs_review_claims || []
  const slideErrors = verifier.slide_errors || []

  // 기각된 주장은 오류가 아니므로 총 개수·필터·목록 어디에도 미포함
  const knowledgeItems = [...confirmed, ...needsReview]
  const totalCount = knowledgeItems.length + slideErrors.length

  return (
    <div className="verifier-results">
      <div className="result-total">
        <span className="result-total-label">총 오류 개수</span>
        <strong className="result-total-value">{totalCount}</strong>
      </div>

      <IssueExplorer knowledgeItems={knowledgeItems} slideErrors={slideErrors} onSeek={onSeek} />
    </div>
  )
}
