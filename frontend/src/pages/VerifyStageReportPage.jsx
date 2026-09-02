// 검증 과정 보기 화면 — 상단 흐름 패널(5단계 탭 + 요약 수치) + 선택 단계의 항목 카드 목록
// 각 단계의 중간 산출물(raw JSON)을 GET /lectures/{id}/artifacts/{stage} 로 받아 렌더

import { useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getLectureArtifact, getLectureResult } from '../api/pipeline'
import { PIPELINE_NODES } from '../components/verifier/verifierConstants'

// --- 라벨/색 매핑 -----------------------------------------------------------

const STAGE_IDS = ['claim_extraction', 'issue_judge', 'issue_classification', 'web_grounding', 'final_verification']
const STAGE_TABS = STAGE_IDS.map(id => PIPELINE_NODES.find(node => node.id === id)).filter(Boolean)

const GROUNDING_STATUS_LABELS = {
  verified: '웹 근거 확인',
  supports_issue: '이슈 근거 있음',
  refutes_issue: '이슈 반박 근거',
  insufficient_evidence: '근거 부족',
  grounding_unavailable: '웹 근거 확인 실패',
  not_applicable: '대상 아님',
}

// claim_extractor 가 부여하는 5개 claim_type (없어도 0 으로 항상 표시)
const CLAIM_TYPE_ORDER = ['definition', 'numeric', 'causal', 'relationship', 'currentness']
const BASIS_CODE_LABELS = {
  // claim 유형
  definition: '정의',
  numeric: '수치/단위',
  causal: '인과관계',
  relationship: '관계 진술',
  currentness: '시의성',
  // 오류 탐지 basis_code (issue_detector)
  definition_relation: '정의·관계',
  mechanism_actor: '메커니즘·주체',
  scope_condition: '범위·조건',
  terminology: '용어',
}

const MODEL_COLORS = {
  gpt: '#10a37f',
  claude: '#d97757',
  grok: '#8b5cf6',
  gemini: '#4285f4',
}

function modelColor(name) {
  const key = String(name || '').toLowerCase().split(/[-.]/)[0]
  return MODEL_COLORS[key] || '#94a3b8'
}

const FLOW_ISSUE_TYPE_KEYS = ['factual_error', 'temporal_error', 'scope_overclaim', 'confusing_explanation']
const FLOW_ISSUE_TYPE_LABELS = {
  factual_error: '사실 오류',
  temporal_error: '오래된 내용',
  scope_overclaim: '과도한 일반화',
  confusing_explanation: '혼동 가능 설명',
  composite_issue: '복합 오류',
}

// 최종 판단에서 각 모델이 낸 판정
const JUDGMENT_LABELS = {
  valid_issue: '오류로 판단',
  not_issue: '오류 아님',
  insufficient_context: '문맥 부족',
}

// 웹 근거 판정
const WEB_VERDICT_LABELS = {
  true: '주장 맞음', supported: '주장 맞음', claim_true: '주장 맞음', verified_true: '주장 맞음',
  false: '주장 틀림', refuted: '주장 틀림', claim_false: '주장 틀림', verified_false: '주장 틀림',
  uncertain: '불명확', unclear: '불명확', unknown: '불명확', inconclusive: '불명확',
  not_applicable: '대상 아님',
}

function issueTypeLabel(key) {
  return FLOW_ISSUE_TYPE_LABELS[key] || key || ''
}

// 분류 칩 색 — 단계별 taxonomy 마다 색 "계열"이 다르고, 계열 안에서도 값별로 다름
//  - 오류 유형(유형 분류·필터링·최종 판단): 파랑 계열
//  - basis code(오류 탐지): 붉은 계열
//  - claim 유형(claim 추출): 초록 계열
const CAT_FAMILIES = {
  errtype: {
    factual_error: '#38bdf8', temporal_error: '#2563eb', scope_overclaim: '#1e40af',
    confusing_explanation: '#7c3aed', composite_issue: '#0e7490',
    _fallback: ['#38bdf8', '#2563eb', '#1e40af', '#7c3aed', '#0e7490'],
  },
  basis: {
    // 핑크 → 빨강 → 주황 → 자주 → 로즈
    definition_relation: '#db2777', mechanism_actor: '#dc2626', scope_condition: '#ea580c',
    terminology: '#a21caf', currentness: '#e11d48',
    _fallback: ['#db2777', '#dc2626', '#ea580c', '#a21caf', '#e11d48'],
  },
  claim: {
    // 초록 → 노랑 → 연두 → 청록 → 에메랄드
    definition: '#16a34a', numeric: '#ca8a04', causal: '#65a30d',
    relationship: '#0d9488', currentness: '#047857',
    _fallback: ['#16a34a', '#ca8a04', '#65a30d', '#0d9488', '#047857'],
  },
}

// (분류 키, 단계) → 칩 색 — 매핑에 없으면 문자열 해시로 계열 폴백 색 선택
function chipColor(key, stage) {
  const fam = stage === 'claim_extraction'
    ? CAT_FAMILIES.claim
    : stage === 'issue_judge'
      ? CAT_FAMILIES.basis
      : CAT_FAMILIES.errtype
  if (fam[key]) return fam[key]
  const s = String(key || '')
  if (!s || s === 'unknown') return 'var(--muted)'
  let h = 0
  for (let i = 0; i < s.length; i += 1) h = (h * 31 + s.charCodeAt(i)) >>> 0
  return fam._fallback[h % fam._fallback.length]
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

// --- 흐름 패널 컬럼 빌더 (단계별 요약 수치) --------------------------------

// 가중 점수 맵 → "사실 오류 60% · 오래된 내용 40%" 문자열 (유형 키 한국어화, 합산 1 → 100%)
function formatWeightedScores(map) {
  if (!map || typeof map !== 'object') return ''
  return Object.entries(map)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .map(([key, value]) => `${issueTypeLabel(key)} ${formatPercent(value)}`)
    .join(' · ')
}

// 항목들을 고정 4개 오류 유형으로 집계 (그 외 유형은 무시)
function countByFixedTypes(items, typeFn) {
  const counts = new Map(FLOW_ISSUE_TYPE_KEYS.map(key => [key, 0]))
  asArray(items).forEach(item => {
    const key = typeFn(item)
    if (counts.has(key)) counts.set(key, counts.get(key) + 1)
  })
  return FLOW_ISSUE_TYPE_KEYS.map(key => ({ key, label: FLOW_ISSUE_TYPE_LABELS[key], value: counts.get(key) }))
}

// claim 추출 단계 요약 — 추출된 claim 수 + 유형별 분포(5종 고정)
function buildClaimColumn(data) {
  const ready = Boolean(data)
  const claims = asArray(data?.claims)
  const typeCounts = new Map()
  claims.forEach(claim => {
    const key = claim.claim_type || 'unknown'
    typeCounts.set(key, (typeCounts.get(key) || 0) + 1)
  })
  // 5개 claim_type 을 항상 고정 순서로, 없으면 0
  const detailRows = CLAIM_TYPE_ORDER.map(key => ({
    key,
    label: BASIS_CODE_LABELS[key],
    value: typeCounts.get(key) || 0,
  }))
  const extra = [...typeCounts.keys()].filter(key => !CLAIM_TYPE_ORDER.includes(key))
  extra.forEach(key => detailRows.push({ key, label: basisCodeLabel(key), value: typeCounts.get(key) }))
  return {
    key: 'claim_extraction',
    title: 'claim 수',
    ready,
    mainLabel: '추출된 claim 수',
    mainValue: claims.length,
    alwaysOpen: true,
    detailRows,
  }
}

// 오류 탐지 단계 요약 — 탐지된 오류 후보 수 + 모델별 후보 수
function buildIssueJudgeColumn(judgeData, summaryData) {
  const ready = Boolean(judgeData)
  const issues = asArray(judgeData?.issues)
  const s = summaryData?.summary || {}
  const counts = s.issue_counts_by_model || {}
  // 검증 시점에 실제 적용된 모델 목록 (파이프라인이 쓴 셋을 그대로 기록 — 프론트에서 고정 안 함)
  const models = asArray(judgeData?.models || summaryData?.models)
  const modelList = models.length ? models : Object.keys(counts)
  return {
    key: 'issue_judge',
    title: '오류 후보 수',
    ready,
    mainLabel: '탐지된 오류 후보 수',
    mainValue: issues.length,
    alwaysOpen: true,
    detailRows: modelList.map(model => ({ key: model, label: model, value: counts[model] ?? 0 })),
  }
}

// 오류 유형 분류 단계 요약 — 최종 유형별 분류 결과
function buildIssueTypeColumn(data) {
  const ready = Boolean(data)
  const branchRows = countByFixedTypes(data?.classifications, item => item.final_issue_type)
  return {
    key: 'issue_classification',
    title: '유형 분류 결과',
    ready,
    branchRows,
  }
}

// 오류 필터링(웹 근거) 단계 요약 — 검색 대상 + 결과 분류(각 대상은 셋 중 하나로 종료)
function buildWebGroundingColumn(data) {
  const ready = Boolean(data)
  const s = data?.summary || {}
  const target = asArray(data?.evidence_items).length || Number(s.target_count) || 0
  return {
    key: 'web_grounding',
    title: '웹 근거 수',
    ready,
    mainLabel: '검색 대상',
    mainValue: target,
    mainDivider: true,
    branchRows: [
      { key: 'verified', label: '근거 확인', value: s.verified_count ?? 0 },
      { key: 'insufficient', label: '근거 부족', value: s.insufficient_evidence_count ?? 0 },
      { key: 'unavailable', label: '검색 실패', value: s.grounding_unavailable_count ?? 0 },
    ],
  }
}

// 최종 오류 수 기준 = 결과 페이지·통계와 동일: 확정 + 교수확인 (기각 제외)
// 원본은 verification_final.json(GET /result) — classified_issue_verifier 아티팩트엔 확정/기각 판정이 없음
const FINAL_ERROR_STATUSES = new Set(['confirmed', 'professor_check'])

// 최종 판단 단계 요약 — 최종 오류 수 + 모델별 valid_issue 수 + 기각 수
function buildFinalColumn(resultData, finalData) {
  const ready = Boolean(resultData)
  const items = asArray(resultData?.feedback_items)
  const kept = items.filter(it => FINAL_ERROR_STATUSES.has(it.status))
  const rejected = resultData?.summary?.rejected_feedback_count
    ?? items.filter(it => it.status === 'rejected').length
  // 각 모델이 최종 검증에서 "오류다"(valid_issue)로 판단한 건수
  const modelRows = Object.values(finalData?.model_results || {}).map(r => {
    const name = r.resolved_model || asArray(r.judgments)[0]?.model || '모델'
    return {
      key: name,
      label: name,
      value: asArray(r.judgments).filter(j => j.judgment === 'valid_issue').length,
    }
  })
  return {
    key: 'final_verification',
    title: '최종 오류 개수',
    ready,
    mainLabel: '최종 오류 수',
    mainValue: kept.length,
    mainDivider: modelRows.length > 0,
    branchRows: modelRows.length ? modelRows : undefined,
    detailRows: [{ key: 'rejected', label: '기각 오류 수', value: rejected }],
  }
}

// 흐름 패널의 단계 요약 노드 — mainValue / 분기 행 / 상세 행을 순서대로 렌더 (ready 아니면 '-')
function FlowNode({ ready, mainLabel, mainValue, branchRows, detailRows, mainDivider = false }) {
  const detail = asArray(detailRows)
  return (
    <div className="flow-node">
      {mainValue !== undefined && (
        <div className="flow-node-row">
          <span>{mainLabel}</span>
          <strong>{ready ? mainValue : '-'}</strong>
        </div>
      )}
      {mainDivider && mainValue !== undefined && branchRows && <div className="flow-detail-divider" />}
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
      {detail.length > 0 && (
        <div className="flow-node-detail">
          {detail.map((row, index) => (
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
  )
}

function useArtifact(lectureId, stage) {
  return useQuery({
    queryKey: ['lecture-artifact', lectureId, stage],
    queryFn: () => getLectureArtifact(lectureId, stage),
  })
}

// --- 화면 컴포넌트 -------------------------------------------------------------

// 상단 흐름 패널 — 5단계 탭 + 각 단계 요약 노드 한 줄
// 모든 단계 아티팩트를 한 번에 조회해 요약 수치 계산
function StageFlowPanel({ lectureId, activeStage, onSelect }) {
  const claimQ = useArtifact(lectureId, 'claim_extraction')
  const judgeQ = useArtifact(lectureId, 'issue_judge')
  const judgeSummaryQ = useArtifact(lectureId, 'issue_judge_summary')
  const typeQ = useArtifact(lectureId, 'issue_classification')
  const groundingQ = useArtifact(lectureId, 'web_grounding')
  const finalQ = useArtifact(lectureId, 'final_verification')
  const resultQ = useQuery({
    queryKey: ['lecture-result', lectureId],
    queryFn: () => getLectureResult(lectureId),
  })

  const columns = [
    buildClaimColumn(claimQ.data),
    buildIssueJudgeColumn(judgeQ.data, judgeSummaryQ.data),
    buildIssueTypeColumn(typeQ.data),
    buildWebGroundingColumn(groundingQ.data),
    buildFinalColumn(resultQ.data, finalQ.data),
  ]

  return (
    <section className="stage-flow" aria-label="검증 단계별 흐름">
      <div className="stage-flow-tabs" role="tablist">
        {STAGE_TABS.map((tab, index) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={activeStage === tab.id}
            className={`stage-flow-tab${activeStage === tab.id ? ' is-active' : ''}`}
            onClick={() => onSelect(tab.id)}
          >
            <span className="stage-flow-index">{index + 1}</span>
            <span className="stage-flow-label">{tab.label}</span>
          </button>
        ))}
      </div>
      <div className="stage-flow-nodes">
        {STAGE_TABS.map((tab, index) => {
          const { key, title, alwaysOpen, ...node } = columns[index] || {}
          return <FlowNode key={tab.id} {...node} />
        })}
      </div>
    </section>
  )
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

// 카드 펼침 영역의 "모델별 판단" 목록 — tag(판정) 볼드 + 이유 본문
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
            </div>
            <p className="claim-detail-text">
              {row.tag && <><strong>{row.tag}</strong> - </>}
              {row.text}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}

// 항목 카드 (claim/오류) — 요약 행(카테고리 칩 + 부가 칩 + 제목) + 펼침 상세(상세 행 + 모델별 판단)
function ArtifactCard({ index, title, categoryChip, chips, detailRows, modelRows }) {
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
            {categoryChip && (
              <span className="claim-tag claim-tag--cat" style={{ '--tag-color': categoryChip.color }}>
                {categoryChip.text}
              </span>
            )}
            {asArray(chips).filter(Boolean).map((chip, chipIndex) => (
              <span
                key={chipIndex}
                className={['claim-tag', chip.className].filter(Boolean).join(' ')}
                style={chip.color ? { '--tag-color': chip.color } : undefined}
              >
                {chip.text}
              </span>
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
              <div className="claim-detail-group claim-detail-group--no-divider">
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

// 슬라이드 검사 항목 카드 — 원문→수정문 요약 + 펼침 상세(슬라이드 이미지 포함)
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

// --- 단계별 항목 카드 행 빌더 (아티팩트 JSON → ArtifactCard props) -----------

function buildClaimExtractionRows(data) {
  return asArray(data?.claims).map(claim => ({
    key: claim.claim_id,
    title: claim.resolved_claim || claim.claim_text,
    groupKey: claim.claim_type || 'unknown',
    groupLabel: basisCodeLabel(claim.claim_type),
    chips: [],
    detailRows: [
      { label: '원 발화', value: claim.claim_text },
      { label: '정규화 주장', value: claim.resolved_claim },
    ],
  }))
}

function buildIssueJudgeRows(data) {
  return asArray(data?.issues).map(issue => ({
    key: issue.issue_id,
    title: issue.issue,
    groupKey: issue.basis_code || 'unknown',
    groupLabel: basisCodeLabel(issue.basis_code),
    chips: asArray(issue.detected_by_models).map(model => ({
      text: model,
      className: 'claim-tag--model',
      color: modelColor(model),
    })),
    detailRows: [
      { label: '슬라이드', value: issue.slide_number },
      { label: '시간', value: formatTime(issue.start_time) },
      { label: '원 발화', value: issue.claim_text },
    ],
    modelRows: asArray(issue.source_model_issues).map(row => ({
      model: row.model,
      tag: basisCodeLabel(row.basis_code),
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
    chips: [],
    detailRows: [
      { label: '근거 코드', value: basisCodeLabel(item.basis_code) },
      { label: '가중 점수', value: formatWeightedScores(item.weighted_scores) },
      { label: '원 발화', value: item.claim_text },
    ],
    modelRows: asArray(item.model_classifications).map(row => ({
      model: row.resolved_model || row.model,
      tag: row.top_issue_type_label || issueTypeLabel(row.top_issue_type),
      text: row.reason,
    })),
  }))
}

// 각 후보가 어떤 모델 합의로 걸러졌는지(=기각 사유)를 칩 하나로 — 의미색(빨강=기각 / 초록=인정 / 앰버=애매)
function verdictChip(issue, judgments) {
  const j = asArray(judgments)
  const valid = j.filter(x => x.judgment === 'valid_issue').length
  const notIssue = j.filter(x => x.judgment === 'not_issue').length
  const insufficient = j.filter(x => x.judgment === 'insufficient_context').length
  const total = j.length
  if (issue.model_disagreement_needs_review) return { text: '모델 판단 엇갈림', className: 'claim-tag--review' }
  if (total > 0 && notIssue === total) return { text: '모델 전원 기각', className: 'claim-tag--reject' }
  if (notIssue > valid) return { text: '모델 다수 기각', className: 'claim-tag--reject' }
  if (insufficient > 0 && valid <= notIssue) return { text: '문맥 부족', className: 'claim-tag--review' }
  if (valid > notIssue) return { text: '모델 다수 인정', className: 'claim-tag--accept' }
  return { text: '심각도 미달', className: 'claim-tag--review' }
}

function buildFinalVerificationRows(data) {
  // issue id 별로 모델 판단을 모음
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
      verdictChip(issue, judgmentsByIssue.get(issue.id)),
      issue.location?.slide_number != null && {
        text: `슬라이드 ${issue.location.slide_number}`,
        className: 'claim-tag--muted',
      },
    ],
    detailRows: [
      { label: '시간', value: formatTime(issue.location?.start_time) },
      { label: '원 발화', value: issue.claim_text },
    ],
    modelRows: (judgmentsByIssue.get(issue.id) || []).map(row => ({
      model: row.resolved_model || row.model,
      tag: JUDGMENT_LABELS[row.judgment] || row.judgment,
      text: row.reason,
    })),
  }))
}

function buildWebGroundingRows(data) {
  return asArray(data?.evidence_items).map((item, index) => ({
    key: item.candidate_id || item.issue_id || index,
    title: item.suspected_error || item.verification_question || item.claim_id || `근거 ${index + 1}`,
    groupKey: item.category || 'unknown',
    groupLabel: FLOW_ISSUE_TYPE_LABELS[item.category] || item.category || '미분류',
    chips: [
      item.status && {
        text: GROUNDING_STATUS_LABELS[item.status] || item.status,
        className: item.status === 'verified'
          ? 'claim-tag--accept'
          : item.status === 'not_applicable'
            ? 'claim-tag--muted'
            : 'claim-tag--review',
      },
    ],
    detailRows: [
      { label: '검증 질문', value: item.verification_question },
      { label: '검색 질의', value: asArray(item.search_queries).join(' · ') },
      { label: '웹 판정', value: WEB_VERDICT_LABELS[item.web_claim_verdict] || item.web_claim_verdict },
      { label: '판정 이유', value: item.web_verdict_reason || item.partial_evidence },
    ],
    modelRows: [],
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

// 선택된 단계의 항목 카드 목록 — 단계별 행 빌더로 아티팩트를 변환해 렌더
function StageSection({ stage }) {
  const { lectureId } = useParams()
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
        web_grounding: buildWebGroundingRows,
        final_verification: buildFinalVerificationRows,
      }[stage]?.(data) || [])

  if (!rows.length) return <p className="list-note">데이터가 없습니다.</p>

  return (
    <div className="issue-explorer">
      <div className="issue-list-toolbar">
        <span className="issue-list-count">{rows.length}건</span>
      </div>
      <div className="claim-list claim-list--compact">
        {isSlideReview
          ? rows.map((row, index) => <SlideReviewCard key={row.key} index={index} item={row.item} />)
          : rows.map((row, index) => (
              <ArtifactCard
                key={row.key || index}
                index={index}
                title={row.title}
                categoryChip={{ text: row.groupLabel, color: chipColor(row.groupKey, stage) }}
                chips={row.chips}
                detailRows={row.detailRows}
                modelRows={row.modelRows}
              />
            ))}
      </div>
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

      <StageFlowPanel lectureId={lectureId} activeStage={activeStage} onSelect={setActiveStage} />

      <StageSection key={activeStage} stage={activeStage} />
    </div>
  )
}
