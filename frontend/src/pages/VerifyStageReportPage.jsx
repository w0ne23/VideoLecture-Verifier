import { Fragment, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getLectureArtifact } from '../api/pipeline'
import { PIPELINE_NODES } from '../components/verifier/verifierConstants'

const STAGE_IDS = ['claim_extraction', 'issue_judge', 'issue_classification', 'final_verification', 'slide_review']
const STAGE_TABS = STAGE_IDS.map(id => PIPELINE_NODES.find(node => node.id === id)).filter(Boolean)

const BASIS_CODE_LABELS = {
  currentness: '시의성',
  causal: '인과관계',
  relationship: '관계 진술',
  numeric: '수치/단위',
  definition: '정의',
  comparison: '비교',
  procedure: '절차',
}

const FLOW_ISSUE_TYPE_KEYS = ['factual_error', 'temporal_error', 'scope_overclaim', 'confusing_explanation']
const FLOW_ISSUE_TYPE_LABELS = {
  factual_error: '사실 오류',
  temporal_error: '오래된 내용',
  scope_overclaim: '과도한 일반화',
  confusing_explanation: '혼동 가능 설명',
}

const TYPE_COLOR_VARS = {
  factual_error: 'var(--red)',
  temporal_error: 'var(--amber)',
  scope_overclaim: 'var(--info)',
  confusing_explanation: '#c084fc',
  composite_issue: 'var(--teal)',
  text_error: 'var(--sky)',
  numeric_unit: 'var(--orange)',
  code_syntax: '#c084fc',
}

function asArray(value) {
  return Array.isArray(value) ? value : []
}

function basisCodeLabel(code) {
  if (!code) return '미분류'
  return BASIS_CODE_LABELS[code] || code
}

function formatTime(seconds) {
  const value = Number(seconds)
  if (!Number.isFinite(value)) return ''
  const total = Math.max(0, Math.floor(value))
  const mm = Math.floor(total / 60)
  const ss = total % 60
  return `${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`
}

function formatPercent(value) {
  const num = Number(value)
  if (!Number.isFinite(num)) return ''
  const percent = num <= 1 ? num * 100 : num
  return `${percent.toFixed(percent >= 10 ? 0 : 1)}%`
}

function formatScoreMap(map) {
  if (!map || typeof map !== 'object') return ''
  return Object.entries(map)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .map(([key, value]) => `${key} ${formatPercent(value)}`)
    .join(' · ')
}

function isReviewTarget(issue) {
  return Boolean(issue?.needs_manual_review) || Boolean(issue?.model_disagreement_needs_review)
}

function countByFixedTypes(items, typeFn) {
  const counts = new Map(FLOW_ISSUE_TYPE_KEYS.map(key => [key, 0]))
  asArray(items).forEach(item => {
    const key = typeFn(item)
    if (counts.has(key)) counts.set(key, counts.get(key) + 1)
  })
  return FLOW_ISSUE_TYPE_KEYS.map(key => ({ key, label: FLOW_ISSUE_TYPE_LABELS[key], value: counts.get(key) }))
}

function modelWeightRows(weights) {
  return Object.entries(weights || {}).map(([label, value]) => ({
    key: label,
    label,
    value: Number.isFinite(Number(value)) ? Number(value).toFixed(2) : String(value ?? '-'),
  }))
}

function buildClaimColumn(data) {
  const ready = Boolean(data)
  const claims = asArray(data?.claims)
  const typeCounts = new Map()
  claims.forEach(claim => {
    const key = claim.claim_type || 'unknown'
    typeCounts.set(key, (typeCounts.get(key) || 0) + 1)
  })
  return {
    key: 'claim_extraction',
    title: '주장 추출',
    ready,
    mainLabel: 'claims',
    mainValue: claims.length,
    detailRows: [...typeCounts.entries()].map(([key, value]) => ({ key, label: basisCodeLabel(key), value })),
  }
}

function buildIssueJudgeColumn(judgeData, summaryData) {
  const ready = Boolean(judgeData)
  const issues = asArray(judgeData?.issues)
  const s = summaryData?.summary || {}
  const detailRows = [
    { key: 'agreed', label: '모델 전원 일치', value: s.all_models_agreed_count ?? '-' },
    { key: 'partial', label: '절반 이상 일치', value: s.majority_agreement_count ?? s.partial_agreement_count ?? '-' },
    { key: 'single', label: '단일 강한 확신 통과', value: s.single_model_only_count ?? '-' },
    { key: 'none', label: '이슈 없음', value: s.no_issue_claim_count ?? '-' },
    { key: 'reject', label: '합의 기준 미달 기각', value: s.consensus_rejected_count ?? s.rejected_single_model_low_confidence_count ?? '-' },
    { divider: true },
    ...Object.entries(s.issue_counts_by_model || {}).map(([model, count]) => ({ key: model, label: model, value: count })),
  ]
  return {
    key: 'issue_judge',
    title: '이슈 후보 판단',
    ready,
    mainLabel: 'issues',
    mainValue: issues.length,
    detailRows,
  }
}

function buildIssueTypeColumn(data) {
  const ready = Boolean(data)
  const branchRows = countByFixedTypes(data?.classifications, item => item.final_issue_type)
  return {
    key: 'issue_classification',
    title: '이슈 유형 분류',
    ready,
    branchRows,
    detailRows: [
      { key: 'weight-head', label: '모델 가중치', value: '' },
      ...modelWeightRows(data?.model_weights),
    ],
  }
}

function buildFinalColumn(data) {
  const ready = Boolean(data)
  const issues = asArray(data?.all_issues)
  const reviewTargets = issues.filter(isReviewTarget)
  const branchRows = countByFixedTypes(reviewTargets, item => item.category)
  const problemThresholdCount = issues.filter(item => Number(item.final_severity_score) > 0.2).length
  const disagreementCount = issues.filter(item => item.model_disagreement_needs_review).length
  return {
    key: 'final_verification',
    title: '멀티 LLM 검증',
    ready,
    mainLabel: 'final',
    mainValue: reviewTargets.length,
    branchRows,
    detailRows: [
      { key: 'threshold', label: '문제 기준 초과', value: problemThresholdCount },
      { key: 'disagreement', label: '모델 의견 불합치', value: disagreementCount },
      { divider: true },
      { key: 'weight-head', label: '모델 가중치', value: '' },
      ...modelWeightRows(data?.model_weights),
    ],
  }
}

function FlowColumn({ title, ready, mainLabel, mainValue, branchRows, detailRows }) {
  const [open, setOpen] = useState(false)
  const hasDetail = asArray(detailRows).length > 0
  return (
    <div className="flow-column">
      <div className="flow-column-head">
        <span className="flow-column-title">{title}</span>
        {hasDetail && (
          <button type="button" className="flow-detail-toggle" aria-expanded={open} onClick={() => setOpen(prev => !prev)}>
            {open ? '접기' : '상세'}
          </button>
        )}
      </div>
      <div className="flow-node">
        {mainValue !== undefined && (
          <div className="flow-node-row">
            <span>{mainLabel}</span>
            <strong>{ready ? mainValue : '-'}</strong>
          </div>
        )}
        {branchRows && (
          <div className="flow-branch-rows">
            {branchRows.map(row => (
              <div className="flow-branch-row" key={row.key}>
                <span>{row.label}</span>
                <strong>{ready ? row.value : '-'}</strong>
              </div>
            ))}
          </div>
        )}
        {open && (
          <div className="flow-node-detail">
            {detailRows.map((row, index) => (
              row.divider
                ? <div className="flow-detail-divider" key={`div-${index}`} />
                : <div className="flow-detail-row" key={row.key || index}>
                    <span>{row.label}</span>
                    <strong>{ready ? row.value : '-'}</strong>
                  </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function useArtifact(lectureId, stage) {
  return useQuery({
    queryKey: ['lecture-artifact', lectureId, stage],
    queryFn: () => getLectureArtifact(lectureId, stage),
  })
}

function FlowSummaryPanel({ lectureId }) {
  const claimQ = useArtifact(lectureId, 'claim_extraction')
  const judgeQ = useArtifact(lectureId, 'issue_judge')
  const judgeSummaryQ = useArtifact(lectureId, 'issue_judge_summary')
  const typeQ = useArtifact(lectureId, 'issue_classification')
  const finalQ = useArtifact(lectureId, 'final_verification')

  const columns = [
    buildClaimColumn(claimQ.data),
    buildIssueJudgeColumn(judgeQ.data, judgeSummaryQ.data),
    buildIssueTypeColumn(typeQ.data),
    buildFinalColumn(finalQ.data),
  ]

  return (
    <section className="flow-summary" aria-label="흐름 보고서">
      <div className="flow-summary-head"><span>흐름 보고서</span></div>
      <div className="flow-summary-row">
        {columns.map(({ key, ...column }, index) => (
          <Fragment key={key}>
            {index > 0 && (
              <span className="flow-summary-arrow" aria-hidden="true">
                <svg viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
              </span>
            )}
            <FlowColumn {...column} />
          </Fragment>
        ))}
      </div>
    </section>
  )
}

function groupRows(rows) {
  const order = []
  const map = new Map()
  rows.forEach(row => {
    const key = row.groupKey || 'unknown'
    if (!map.has(key)) {
      map.set(key, { key, label: row.groupLabel || key, rows: [] })
      order.push(key)
    }
    map.get(key).rows.push(row)
  })
  return order.map(key => map.get(key))
}

function DetailRow({ label, value }) {
  if (value === undefined || value === null || value === '') return null
  return (
    <div className="claim-detail-row">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  )
}

function ModelBreakdown({ rows }) {
  const items = asArray(rows).filter(row => row?.text)
  if (!items.length) return null
  return (
    <div className="claim-detail-group">
      <h4>모델별 판단</h4>
      <div className="model-judgment-list">
        {items.map((row, index) => (
          <div className="model-judgment" key={`${row.model}-${index}`}>
            <div className="model-judgment-head">
              <strong>{row.model}</strong>
              {row.tag && <span>{row.tag}</span>}
            </div>
            <p className="claim-detail-text">{row.text}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

function ArtifactCard({ index, title, chips, detailRows, modelRows }) {
  const [open, setOpen] = useState(false)
  const toggle = () => setOpen(prev => !prev)
  const handleKeyDown = event => {
    if (event.key !== 'Enter' && event.key !== ' ') return
    event.preventDefault()
    toggle()
  }

  return (
    <article className={`claim-card claim-card--compact${open ? ' claim-card--open' : ''}`}>
      <div className="claim-card-summary" role="button" tabIndex={0} aria-expanded={open} onClick={toggle} onKeyDown={handleKeyDown}>
        <div className="claim-card-main">
          <div className="claim-card-head">
            <span className="claim-card-index">#{index + 1}</span>
            {asArray(chips).filter(Boolean).map((chip, chipIndex) => (
              <span key={chipIndex} className={`claim-tag${chip.className ? ` ${chip.className}` : ''}`}>{chip.text}</span>
            ))}
          </div>
          {title && <p className="claim-text">{title}</p>}
        </div>
        <span className={`claim-card-toggle${open ? ' claim-card-toggle--open' : ''}`} aria-hidden="true">▾</span>
      </div>
      {open && (
        <div className="claim-detail">
          <div className="claim-detail-body">
            {asArray(detailRows).some(row => row?.value) && (
              <div className="claim-detail-group">
                <h4>상세</h4>
                <dl className="claim-detail-list">
                  {asArray(detailRows).map((row, rowIndex) => (
                    <DetailRow key={rowIndex} label={row.label} value={row.value} />
                  ))}
                </dl>
              </div>
            )}
            <ModelBreakdown rows={modelRows} />
          </div>
        </div>
      )}
    </article>
  )
}

function SlideReviewCard({ index, item }) {
  const [open, setOpen] = useState(false)
  const toggle = () => setOpen(prev => !prev)
  return (
    <article className={`claim-card claim-card--compact claim-card--slide slide-error-card${open ? ' claim-card--open' : ''}`}>
      <div className="claim-card-summary" role="button" tabIndex={0} aria-expanded={open} onClick={toggle}>
        <div className="claim-card-main">
          <div className="claim-card-head">
            <span className="claim-card-index">#{index + 1}</span>
            {item.slide_number != null && <span className="claim-location-chip">slide {item.slide_number}</span>}
            {item.confidence !== undefined && <span className="claim-tag">신뢰도 {formatPercent(item.confidence)}</span>}
          </div>
          <p className="slide-error-change slide-error-change--compact">
            <span>{item.problematic_text || '-'}</span>
            <strong>→</strong>
            <span>{item.corrected_text || '-'}</span>
          </p>
        </div>
        <span className={`claim-card-toggle${open ? ' claim-card-toggle--open' : ''}`} aria-hidden="true">▾</span>
      </div>
      {open && (
        <div className="slide-error-detail">
          {item.slide_image_url && (
            <div className="slide-error-image slide-error-image--large">
              <img src={item.slide_image_url} alt={`slide ${item.slide_number || index + 1}`} loading="lazy" />
            </div>
          )}
          <dl className="claim-detail-list slide-error-detail-list">
            <DetailRow label="오류 텍스트" value={item.problematic_text} />
            <DetailRow label="수정 텍스트" value={item.corrected_text} />
            <DetailRow label="판단 이유" value={item.reason} />
            <DetailRow label="모델" value={item.model} />
          </dl>
        </div>
      )}
    </article>
  )
}

function StageFilterBar({ groups, active, onSelect }) {
  return (
    <div className="stage-filter-bar" role="group" aria-label="유형 필터">
      {groups.map(group => {
        const accent = TYPE_COLOR_VARS[group.key]
        const isActive = active === group.key
        return (
          <button
            key={group.key}
            type="button"
            className={`stage-filter-btn${isActive ? ' stage-filter-btn--active' : ''}`}
            style={accent ? { '--accent': accent } : undefined}
            aria-pressed={isActive}
            onClick={() => onSelect(prev => (prev === group.key ? null : group.key))}
          >
            <span className="stage-filter-count">{group.rows.length}</span>
            <span className="stage-filter-label">{group.label}</span>
          </button>
        )
      })}
    </div>
  )
}

function buildClaimExtractionRows(data) {
  return asArray(data?.claims).map(claim => ({
    key: claim.claim_id,
    title: claim.resolved_claim || claim.claim_text,
    groupKey: claim.claim_type || 'unknown',
    groupLabel: basisCodeLabel(claim.claim_type),
    chips: [
      claim.context_id && { text: claim.context_id },
    ],
    detailRows: [
      { label: '원 발화', value: claim.claim_text },
      { label: '정규화 주장', value: claim.resolved_claim },
      { label: '유형', value: claim.claim_type },
    ],
  }))
}

function buildIssueJudgeRows(data) {
  return asArray(data?.issues).map(issue => ({
    key: issue.issue_id,
    title: issue.issue,
    groupKey: issue.basis_code || 'unknown',
    groupLabel: basisCodeLabel(issue.basis_code),
    chips: [
      issue.confidence !== undefined && { text: `신뢰도 ${formatPercent(issue.confidence)}`, className: 'claim-tag--score' },
      ...asArray(issue.detected_by_models).map(model => ({ text: model })),
    ],
    detailRows: [
      { label: '대표 모델', value: issue.representative_model },
      { label: '슬라이드', value: issue.slide_number },
      { label: '시간', value: formatTime(issue.start_time) },
      { label: '원 발화', value: issue.claim_text },
    ],
    modelRows: asArray(issue.source_model_issues).map(row => ({
      model: row.model,
      tag: formatPercent(row.confidence),
      text: row.issue,
    })),
  }))
}

function buildIssueClassificationRows(data) {
  return asArray(data?.classifications).map(item => ({
    key: item.id,
    title: item.issue,
    groupKey: item.final_issue_type || 'unknown',
    groupLabel: item.final_issue_type_label || item.final_issue_type || '미분류',
    chips: [
      item.ensemble_confidence !== undefined && { text: `신뢰도 ${formatPercent(item.ensemble_confidence)}`, className: 'claim-tag--score' },
      item.low_margin && { text: '근소한 차이', className: 'claim-tag--verdict' },
    ],
    detailRows: [
      { label: '근거 코드', value: item.basis_code },
      { label: '가중 점수', value: formatScoreMap(item.weighted_scores) },
      { label: '모델 수', value: item.model_count },
      { label: '원 발화', value: item.claim_text },
    ],
    modelRows: asArray(item.model_classifications).map(row => ({
      model: row.model,
      tag: `${row.top_issue_type_label || row.top_issue_type} ${formatPercent(row.top_probability)}`,
      text: row.reason,
    })),
  }))
}

function buildFinalVerificationRows(data) {
  const judgmentsByIssue = new Map()
  Object.entries(data?.model_results || {}).forEach(([model, result]) => {
    asArray(result?.judgments).forEach(judgment => {
      const issueId = judgment.id
      if (!issueId) return
      if (!judgmentsByIssue.has(issueId)) judgmentsByIssue.set(issueId, [])
      judgmentsByIssue.get(issueId).push({ model, ...judgment })
    })
  })

  return asArray(data?.all_issues).map(issue => ({
    key: issue.id,
    title: issue.resolved_claim || issue.claim_text,
    groupKey: issue.category || 'unknown',
    groupLabel: issue.category_label || issue.category || '미분류',
    chips: [
      issue.location?.slide_number != null && { text: `slide ${issue.location.slide_number}` },
    ],
    detailRows: [
      { label: '시간', value: formatTime(issue.location?.start_time) },
      { label: '원 발화', value: issue.claim_text },
    ],
    modelRows: (judgmentsByIssue.get(issue.id) || []).map(row => ({
      model: row.model,
      tag: `${row.judgment || ''} ${formatPercent(row.final_model_score)}`.trim(),
      text: row.reason,
    })),
  }))
}

function buildSlideReviewRows(data) {
  return asArray(data?.slide_errors).map((item, index) => ({
    key: item.slide_error_id || index,
    item,
    groupKey: item.error_type || 'unknown',
    groupLabel: item.error_type_label || item.error_type || '미분류',
  }))
}

function StageSection({ stage }) {
  const { lectureId } = useParams()
  const [activeFilter, setActiveFilter] = useState(null)
  const { data, isLoading, error } = useQuery({
    queryKey: ['lecture-artifact', lectureId, stage],
    queryFn: () => getLectureArtifact(lectureId, stage),
  })

  if (isLoading) return <p className="list-note">불러오는 중...</p>
  if (error) return <p className="error-text">불러오기 실패: {String(error?.message || error)}</p>

  const isSlideReview = stage === 'slide_review'
  const rows = isSlideReview
    ? buildSlideReviewRows(data)
    : ({
        claim_extraction: buildClaimExtractionRows,
        issue_judge: buildIssueJudgeRows,
        issue_classification: buildIssueClassificationRows,
        final_verification: buildFinalVerificationRows,
      }[stage]?.(data) || [])

  if (!rows.length) return <p className="list-note">데이터가 없습니다.</p>

  const groups = groupRows(rows)
  const visibleRows = activeFilter ? groups.find(group => group.key === activeFilter)?.rows || [] : rows

  return (
    <div className="issue-explorer">
      {groups.length > 1 && <StageFilterBar groups={groups} active={activeFilter} onSelect={setActiveFilter} />}
      <div className="issue-list-toolbar">
        <span className="issue-list-count">{visibleRows.length}건</span>
      </div>
      <div className="claim-list claim-list--compact">
        {isSlideReview
          ? visibleRows.map((row, index) => <SlideReviewCard key={row.key} index={index} item={row.item} />)
          : visibleRows.map((row, index) => (
              <ArtifactCard
                key={row.key || index}
                index={index}
                title={row.title}
                chips={row.chips}
                detailRows={row.detailRows}
                modelRows={row.modelRows}
              />
            ))}
      </div>
    </div>
  )
}

function StagePipeline({ activeStage, onSelect }) {
  return (
    <div className="stage-pipeline" role="tablist" aria-label="검증 파이프라인 단계">
      {STAGE_TABS.map((tab, index) => (
        <Fragment key={tab.id}>
          {index > 0 && (
            <span className="stage-pipeline-arrow" aria-hidden="true">
              <svg viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
            </span>
          )}
          <button
            type="button"
            role="tab"
            aria-selected={activeStage === tab.id}
            className={`stage-pipeline-step${activeStage === tab.id ? ' is-active' : ''}`}
            onClick={() => onSelect(tab.id)}
          >
            <span className="stage-pipeline-index">{index + 1}</span>
            <span className="stage-pipeline-label">{tab.label}</span>
          </button>
        </Fragment>
      ))}
    </div>
  )
}

export default function VerifyStageReportPage() {
  const { lectureId } = useParams()
  const navigate = useNavigate()
  const [activeStage, setActiveStage] = useState(STAGE_TABS[0]?.id || 'claim_extraction')

  return (
    <div className="detail">
      <div className="detail-head">
        <button type="button" className="btn" onClick={() => navigate(`/result/${lectureId}`)}>← 결과로</button>
        <h2>검증 과정 보기</h2>
      </div>

      <StagePipeline activeStage={activeStage} onSelect={setActiveStage} />

      {activeStage !== 'slide_review' && <FlowSummaryPanel lectureId={lectureId} />}

      <StageSection key={activeStage} stage={activeStage} />
    </div>
  )
}
