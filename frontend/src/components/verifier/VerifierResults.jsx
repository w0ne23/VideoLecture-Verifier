import { useState } from 'react'

// 검증 결과 JSON의 claim 항목은 파이프라인 버전에 따라 필드가 조금씩 다르다.
// 자주 쓰는 필드는 골라서 보여주고, 전체 내용은 <details>로 항상 확인 가능하게 한다.

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
    if (Number.isFinite(value) && value > 0) return value
  }
  return null
}

function formatTime(seconds) {
  const total = Math.floor(seconds)
  const mm = String(Math.floor(total / 60)).padStart(2, '0')
  const ss = String(total % 60).padStart(2, '0')
  return `${mm}:${ss}`
}

function ClaimCard({ item, index, tone, onSeek }) {
  const claimText = pick(item, ['claim', 'claim_text', 'statement', 'text', 'content'])
  const verdict = pick(item, ['verdict', 'final_verdict', 'decision', 'status'])
  const issueType = pick(item, ['issue_type', 'issue_category', 'category', 'type'])
  const explanation = pick(item, ['explanation', 'reason', 'reasoning', 'feedback', 'description', 'summary'])
  const startTime = pickNumber(item, ['start', 'start_time', 'start_sec', 'timestamp'])

  return (
    <div className={`claim-card claim-card--${tone}`}>
      <div className="claim-card-head">
        <span className="claim-card-index">#{index + 1}</span>
        {issueType && <span className="claim-tag">{issueType}</span>}
        {verdict && <span className="claim-tag claim-tag--verdict">{verdict}</span>}
        {startTime != null && (
          <button type="button" className="claim-time" onClick={() => onSeek?.(startTime)}>
            ▶ {formatTime(startTime)}
          </button>
        )}
      </div>
      {claimText && <p className="claim-text">{claimText}</p>}
      {explanation && <p className="claim-explanation">{explanation}</p>}
      <details className="claim-raw">
        <summary>원본 JSON</summary>
        <pre>{JSON.stringify(item, null, 2)}</pre>
      </details>
    </div>
  )
}

function Section({ title, items, tone, onSeek, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <section className="result-section">
      <button type="button" className="result-section-head" onClick={() => setOpen(prev => !prev)}>
        <span>{title}</span>
        <span className={`count-badge count-badge--${tone}`}>{items.length}</span>
        <span className="result-section-toggle">{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        items.length === 0
          ? <p className="result-empty">항목이 없습니다.</p>
          : <div className="claim-list">
              {items.map((item, index) => (
                <ClaimCard key={index} item={item} index={index} tone={tone} onSeek={onSeek} />
              ))}
            </div>
      )}
    </section>
  )
}

export default function VerifierResults({ verifier, onSeek }) {
  if (!verifier) return null

  const counts = verifier.counts || {}
  const confirmed = verifier.final_confirmed_claims || []
  const needsReview = verifier.needs_review_claims || []
  const rejected = verifier.verifier_rejected_claims || []
  const slideErrors = verifier.slide_errors || []

  return (
    <div className="verifier-results">
      <div className="count-grid">
        <div className="count-cell count-cell--confirmed">
          <div className="count-value">{counts.final_confirmed ?? confirmed.length}</div>
          <div className="count-label">확정 이슈</div>
        </div>
        <div className="count-cell count-cell--review">
          <div className="count-value">{counts.needs_review ?? needsReview.length}</div>
          <div className="count-label">검토 필요</div>
        </div>
        <div className="count-cell count-cell--slide">
          <div className="count-value">{counts.slide_errors ?? slideErrors.length}</div>
          <div className="count-label">슬라이드 오류</div>
        </div>
        <div className="count-cell count-cell--rejected">
          <div className="count-value">{counts.verifier_rejected ?? rejected.length}</div>
          <div className="count-label">기각</div>
        </div>
      </div>

      {verifier.verification_date && (
        <p className="result-meta">
          검증 시각: {verifier.verification_date}
          {Array.isArray(verifier.models) && verifier.models.length > 0 && ` · 모델: ${verifier.models.join(', ')}`}
        </p>
      )}

      <Section title="확정 이슈" items={confirmed} tone="confirmed" onSeek={onSeek} />
      <Section title="검토 필요" items={needsReview} tone="review" onSeek={onSeek} />
      <Section title="슬라이드 오류" items={slideErrors} tone="slide" onSeek={onSeek} />
      <Section title="기각된 주장" items={rejected} tone="rejected" onSeek={onSeek} defaultOpen={false} />
    </div>
  )
}
