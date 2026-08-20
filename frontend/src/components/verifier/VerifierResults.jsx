import { useMemo, useState } from 'react'

const TYPE_LABELS = {
  factual_error: '사실 오류',
  temporal_error: '오래된 내용',
  scope_overclaim: '과도한 일반화',
  confusing_explanation: '혼동 가능 설명',
  composite_issue: '복합 오류',
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

const RELATION_LABELS = {
  supports_claim: '강의 주장 뒷받침',
  contradicts_claim: '강의 주장 반박',
  irrelevant: '직접 관련 없음',
}

const VERDICT_LABELS = {
  claim_false: '주장 틀림',
  claim_true: '주장 맞음',
  unclear: '불명확',
}

const SOURCE_PRIORITY_LABELS = {
  official_docs: '공식 문서',
  standards_or_government: '표준/정부 자료',
  academic: '학술 자료',
  educational: '교육 자료',
  encyclopedia: '백과사전',
}

// 지식 오류로 분류되는 4개 카테고리 + 슬라이드 오류를 합쳐 5개 필터 버튼을 구성한다.
const KNOWLEDGE_CATEGORY_KEYS = ['factual_error', 'temporal_error', 'scope_overclaim', 'confusing_explanation']
const CATEGORY_DEFS = [
  { key: 'factual_error', label: '사실 오류' },
  { key: 'temporal_error', label: '오래된 내용' },
  { key: 'scope_overclaim', label: '과도한 일반화' },
  { key: 'confusing_explanation', label: '혼동 가능 설명' },
  { key: 'slide', label: '슬라이드 오류' },
]
const SORT_OPTIONS = [
  { key: 'time', label: '시간순' },
  { key: 'severity', label: '심각도순' },
]

function cx(...classNames) {
  return classNames.filter(Boolean).join(' ')
}

function asArray(value) {
  if (!value) return []
  return Array.isArray(value) ? value : [value]
}

function isObject(value) {
  return value && typeof value === 'object' && !Array.isArray(value)
}

function pick(item, keys) {
  for (const key of keys) {
    const value = item?.[key]
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  return ''
}

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

function uniqueText(value, existing = []) {
  const text = compactText(value)
  if (!text) return ''
  const normalized = text.replace(/\s+/g, ' ').trim()
  const duplicate = existing.some(item => {
    const other = compactText(item).replace(/\s+/g, ' ').trim()
    return other && other === normalized
  })
  return duplicate ? '' : text
}

function formatTime(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0))
  const hh = Math.floor(total / 3600)
  const mm = Math.floor((total % 3600) / 60)
  const ss = total % 60
  if (hh > 0) return `${hh}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`
  return `${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`
}

function formatPercent(value) {
  const num = Number(value)
  if (!Number.isFinite(num)) return ''
  return `${num.toFixed(num >= 10 ? 1 : 2)}%`
}

function formatScore(value) {
  const num = Number(value)
  if (!Number.isFinite(num)) return ''
  const percent = num <= 1 ? num * 100 : num
  return `${percent.toFixed(percent >= 10 ? 1 : 2)}점`
}

function formatPoint(value) {
  const num = Number(value)
  if (!Number.isFinite(num)) return ''
  return `${num.toFixed(num >= 10 ? 1 : 2)}점`
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

function isCompositeItem(item) {
  const type = item.feedback_type || item.category || item.issue_type || item.type
  return type === 'composite_issue' || item.scored_as_composite || item.classified_issue_verifier?.scored_as_composite
}

// 지식 오류 항목이 속하는 카테고리 목록을 반환한다.
// 단일 유형이면 그 유형 하나, 복합 오류면 구성 후보 카테고리 전부를 반환해
// 태그로도, 필터 버튼 매칭에도 그대로 쓸 수 있게 한다.
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

function getSlideSeverity(item) {
  const num = Number(item.severity_score)
  if (!Number.isFinite(num)) return null
  return num <= 1 ? num * 100 : num
}

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

function DetailRow({ label, value, wide = false }) {
  if (value === undefined || value === null || value === '') return null
  return (
    <div className={cx('claim-detail-row', wide && 'claim-detail-row--wide')}>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  )
}

function TextBlock({ children }) {
  if (!children) return null
  return <p className="claim-detail-text">{children}</p>
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

// LLM이 "원문 → 수정문" 형태로 쓰는 경우가 있는데, 원문은 바로 위 '원 발화' 행에 이미 있으므로
// 화살표 뒤 수정문만 남긴다.
function stripArrowPrefix(text) {
  if (!text) return text
  const idx = text.indexOf('→')
  return idx === -1 ? text : text.slice(idx + 1).trim()
}

function ScoreGrid({ item }) {
  const verifier = item.classified_issue_verifier || {}
  const rows = [
    ['최종 심각도', formatPoint(getSeverity(item))],
    ['유효 이슈 평균', formatRatio(verifier.average_is_valid_issue)],
    ['유형 심각도 평균', formatRatio(verifier.average_category_severity)],
    ['문맥 해소', formatRatio(verifier.average_context_resolution)],
    ['문맥 미해소', formatRatio(verifier.average_context_unresolved)],
    ['모델 불일치', formatRatio(verifier.model_disagreement)],
  ].filter(([, value]) => value)

  if (!rows.length) return null
  return (
    <details className="claim-score-details">
      <summary>점수</summary>
      <p className="claim-detail-text">{rows.map(([label, value]) => `${label} ${value}`).join(' · ')}</p>
    </details>
  )
}

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
        {scoring.primary_issue_type_label && <span>대표 유형: {scoring.primary_issue_type_label}</span>}
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

function PassageList({ passages }) {
  const items = asArray(passages)
  if (!items.length) return null
  return (
    <div className="grounding-subblock">
      <strong>근거 문단</strong>
      <div className="grounding-passage-list">
        {items.map((passage, index) => {
          const url = sourceUrl(passage)
          return (
            <div className="grounding-passage" key={`${url || passage.id || 'passage'}-${index}`}>
              <div className="grounding-passage-head">
                {passage.stance && <span>{STATUS_LABELS[passage.stance] || passage.stance}</span>}
                {passage.relation_to_claim && (
                  <span>{RELATION_LABELS[passage.relation_to_claim] || passage.relation_to_claim}</span>
                )}
                {passage.match_status && <span>match: {passage.match_status}</span>}
                {passage.match_score !== undefined && <span>{formatRatio(passage.match_score)}</span>}
                {passage.relation_confidence !== undefined && (
                  <span>관계 {formatRatio(passage.relation_confidence)}</span>
                )}
              </div>
              {url && <a href={url} target="_blank" rel="noreferrer">{url}</a>}
              <TextBlock>{passage.key_sentence || passage.quote_or_paragraph || passage.matched_text}</TextBlock>
              {(passage.why_relevant || passage.relation_reason) && (
                <p className="grounding-relevance">{passage.why_relevant || passage.relation_reason}</p>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function VerifiedSources({ trials }) {
  const verifiedSources = asArray(trials).flatMap(trial => asArray(trial.verified_sources))
  if (!verifiedSources.length) return null
  return (
    <details className="grounding-diagnostics">
      <summary>출처 fetch/match 진단 {verifiedSources.length}건</summary>
      <div className="grounding-diagnostic-list">
        {verifiedSources.map((source, index) => (
          <div className="grounding-diagnostic" key={`${source.url || 'source'}-${index}`}>
            <div className="grounding-diagnostic-head">
              <strong>{source.domain || source.url || `source ${index + 1}`}</strong>
              {source.fetch_status && <span>{source.fetch_status}</span>}
              {source.source_priority_label && <span>{SOURCE_PRIORITY_LABELS[source.source_priority_label] || source.source_priority_label}</span>}
              {source.trust_score !== undefined && <span>trust {formatRatio(source.trust_score)}</span>}
            </div>
            {source.url && <a href={source.url} target="_blank" rel="noreferrer">{source.url}</a>}
            {source.error && <p>{source.error}</p>}
            <PassageList passages={source.matched_passages || source.verified_model_passages} />
          </div>
        ))}
      </div>
    </details>
  )
}

function WebGroundingPanel({ item }) {
  const grounding = getGrounding(item)
  const evidence = item.evidence || {}
  if (!grounding || !Object.keys(grounding).length) return null

  const trials = asArray(grounding.trials)
  const trialPassages = trials.flatMap(trial => asArray(trial.evidence_passages))
  const compactEvidence = asArray(grounding.evidence)
  const passages = grounding.evidence_passages?.length
    ? grounding.evidence_passages
    : compactEvidence.length
      ? compactEvidence
      : trialPassages
  const sources = grounding.evidence_sources?.length
    ? grounding.evidence_sources
    : compactEvidence.length
      ? compactEvidence
      : evidence.web_sources

  const judgmentParts = [
    grounding.status && (STATUS_LABELS[grounding.status] || grounding.status),
    grounding.claim_verdict && (VERDICT_LABELS[grounding.claim_verdict] || grounding.claim_verdict),
    grounding.selected_source_priority_label
      ? `${SOURCE_PRIORITY_LABELS[grounding.selected_source_priority_label] || grounding.selected_source_priority_label} 선정 ${grounding.selected_source_count ?? 0}건`
      : grounding.selected_source_count !== undefined && `선정 ${grounding.selected_source_count}건`,
  ].filter(Boolean)

  const passageText = asArray(passages)
    .map(passage => passage.key_sentence || passage.quote_or_paragraph || passage.matched_text)
    .filter(Boolean)
    .join(' / ')

  const queryRows = isObject(grounding.search_queries)
    ? Object.entries(grounding.search_queries).flatMap(([model, values]) => asArray(values).map(value => ({ model, value })))
    : asArray(grounding.search_queries).map(value => ({ model: '', value }))

  const sourceItems = asArray(sources).filter(sourceUrl)

  const queryValue = queryRows.length > 0
    ? queryRows.map((row, index) => (
        <span key={`${row.model}-${row.value}-${index}`}>
          {index > 0 && ', '}
          {row.model && `${row.model}: `}
          <code>{row.value}</code>
        </span>
      ))
    : null

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
          <DetailRow label="판단 결과" value={judgmentParts.join(', ')} wide />
          <DetailRow label="근거 문단" value={passageText} wide />
          <DetailRow label="자료 요약" value={grounding.evidence_summary} wide />
          <DetailRow label="판정 이유" value={grounding.reason} wide />
          <DetailRow label="검색어" value={queryValue} wide />
          <DetailRow label="근거 출처" value={sourceValue} wide />
        </dl>
      </div>
      <VerifiedSources trials={trials} />
    </DetailGroup>
  )
}

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
            <div className="model-judgment-head">
              <strong>{judgment.model || judgment.provider || `model ${index + 1}`}</strong>
              {judgment.category && <span>{labelForType(judgment.category)}</span>}
              {judgment.judgment && <span>{judgment.judgment}</span>}
              {judgment.final_model_score !== undefined && <span>{formatScore(judgment.final_model_score)}</span>}
            </div>
            <TextBlock>{judgment.reason || judgment.explanation || judgment.summary}</TextBlock>
          </div>
        ))}
      </div>
    </DetailGroup>
  )
}

function ClaimDetail({ item }) {
  const problem = item.problem || {}
  const feedback = item.professor_feedback || {}
  const evidence = item.evidence || {}
  const summary = problem.summary || feedback.summary
  const rawWhyWrong = problem.why_wrong || feedback.why_wrong
  const rawRecommendation = problem.recommendation || feedback.suggested_rephrase
  const correctInfo = problem.correct_info
  const evidenceInContext = uniqueText(evidence.evidence_in_context, [
    summary,
    rawWhyWrong,
    rawRecommendation,
    correctInfo,
    feedback.teaching_note,
  ])

  return (
    <div className="claim-detail-body">
      <DetailGroup title="문제 요약">
        <dl className="claim-detail-list">
          <DetailRow label="원 발화" value={item.claim_text} wide />
          <DetailRow label="판단 대상 주장" value={item.resolved_claim} wide />
          <DetailRow label="수정 제안" value={stripArrowPrefix(correctInfo)} wide />
          <DetailRow label="문맥 근거" value={evidenceInContext} wide />
        </dl>
      </DetailGroup>
      <CompositeScoringPanel item={item} />
      <ModelJudgments item={item} />
      <WebGroundingPanel item={item} />
      <ScoreGrid item={item} />
      <details className="claim-debug-json">
        <summary>디버그 JSON</summary>
        <pre>{JSON.stringify(item, null, 2)}</pre>
      </details>
    </div>
  )
}

function ClaimCard({ item, index, categories, onSeek }) {
  const [open, setOpen] = useState(false)
  const claimText = pick(item, ['resolved_claim', 'claim_text', 'claim', 'statement', 'text', 'content'])
  const status = item.status || item.severity_status || pick(item, ['verdict', 'final_verdict', 'decision'])
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
            {status && <span className="claim-tag claim-tag--verdict">{STATUS_LABELS[status] || status}</span>}
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
              <span>최종 심각도</span>
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
            <span className={cx('claim-tag', 'claim-tag--type', `claim-tag--type-${slideTypeKey}`)}>{item.error_type_label || '슬라이드 오류'}</span>
            {item.slide_number && <span className="claim-location-chip">slide {item.slide_number}</span>}
            {item.confidence !== undefined && <span className="claim-tag">신뢰도 {formatPercent(Number(item.confidence) * 100)}</span>}
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
              <span>심각도</span>
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
            <DetailRow label="슬라이드 제목" value={item.slide_title} wide />
            <DetailRow label="오류 유형" value={item.error_type_label || item.error_type} />
            <DetailRow label="오류 텍스트" value={item.problematic_text} />
            <DetailRow label="수정 텍스트" value={item.corrected_text} />
            <DetailRow label="판단 이유" value={item.reason} wide />
          </dl>
          <details className="claim-debug-json">
            <summary>디버그 JSON</summary>
            <pre>{JSON.stringify(item, null, 2)}</pre>
          </details>
        </div>
      )}
    </article>
  )
}

function buildEntries(knowledgeItems, slideErrors) {
  const claimEntries = knowledgeItems.map((item, sourceIndex) => ({
    kind: 'claim',
    item,
    sourceIndex,
    categories: itemCategories(item),
    severity: getSeverity(item),
    // 지식 오류는 실제 영상 재생 시간(초)을 정렬 키로 쓴다.
    time: locationOf(item).startTime,
  }))
  const slideEntries = slideErrors.map((item, sourceIndex) => ({
    kind: 'slide',
    item,
    sourceIndex,
    categories: ['slide'],
    severity: getSlideSeverity(item),
    // 슬라이드 오류에는 초 단위 재생 시간이 없고 슬라이드 번호만 있다(스케일이 다름).
    time: item.slide_number != null ? Number(item.slide_number) : null,
  }))
  return [...claimEntries, ...slideEntries]
}

// 지식 오류(초)와 슬라이드 오류(슬라이드 번호)는 척도가 달라 값을 그대로 비교할 수 없다.
// 각자의 그룹 안에서 0~1 사이 상대적 위치로 정규화한 뒤 섞어야 하나의 시간순 목록처럼 보인다.
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

  // 기각된 주장은 오류가 아니므로 총 개수·필터·목록 어디에도 포함하지 않는다.
  const knowledgeItems = [...confirmed, ...needsReview]
  const totalCount = knowledgeItems.length + slideErrors.length

  return (
    <div className="verifier-results">
      <div className="result-total">
        <span className="result-total-label">이 영상의 총 오류 개수</span>
        <strong className="result-total-value">{totalCount}</strong>
      </div>

      <IssueExplorer knowledgeItems={knowledgeItems} slideErrors={slideErrors} onSeek={onSeek} />
    </div>
  )
}
