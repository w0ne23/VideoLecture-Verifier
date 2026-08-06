import { ISSUE_TYPES } from '../../data/mockStats'

function TypeBars({ typeDist, animKey, maxValue }) {
  const ceiling = maxValue || Math.max(1, ...Object.values(typeDist))

  return (
    <ul key={animKey} className="stats-compare-type-bars">
      {ISSUE_TYPES.map((type, index) => {
        const value = typeDist[type.key] || 0
        return (
          <li key={type.key}>
            <span className="stats-compare-type-name">{type.label}</span>
            <span className="stats-compare-type-track">
              <span
                className="stats-compare-type-fill"
                style={{
                  width: `${(value / ceiling) * 100}%`,
                  background: type.color,
                  animationDelay: `${index * 50}ms`,
                }}
              />
            </span>
            <span className="stats-data-label stats-compare-type-value" style={{ animationDelay: `${80 + index * 50}ms` }}>
              {value}
            </span>
          </li>
        )
      })}
    </ul>
  )
}

export default function BeforeAfterCompare({ pairs, pairId, onPairChange, animKey }) {
  const pair = pairs.find(item => item.id === pairId) || pairs[0]
  if (!pair) return null

  const maxType = Math.max(
    1,
    ...ISSUE_TYPES.flatMap(t => [
      pair.before.typeDist[t.key] || 0,
      pair.after.typeDist[t.key] || 0,
    ]),
  )
  const delta = pair.after.total - pair.before.total

  return (
    <div className="stats-compare">
      <label className="field stats-compare-select">
        <span>비교 페어</span>
        <select value={pair.id} onChange={event => onPairChange(event.target.value)}>
          {pairs.map(item => (
            <option key={item.id} value={item.id}>{item.title}</option>
          ))}
        </select>
      </label>

      <div className="stats-compare-meta">
        <span>{pair.tagLabel}</span>
        <span>{pair.domainLabel}</span>
        <span>{pair.durationMin}분</span>
        <span className={delta <= 0 ? 'stats-delta--down' : 'stats-delta--up'}>
          Issue {delta > 0 ? '+' : ''}{delta}
        </span>
      </div>

      <div className="stats-compare-grid">
        {[pair.before, pair.after].map(side => (
          <article key={side.label} className="stats-compare-card">
            <header className="stats-compare-card-head">
              <h3>{side.label}</h3>
              <span>소요 {side.processMin}분 · Issue {side.total}건</span>
            </header>
            <TypeBars typeDist={side.typeDist} animKey={`${animKey}-${side.label}`} maxValue={maxType} />
          </article>
        ))}
      </div>
    </div>
  )
}
