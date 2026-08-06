export default function StatsInsight({ title, bullets }) {
  return (
    <aside className="stats-insight">
      <h3>{title}</h3>
      <p className="stats-insight-kicker">그래프 해석</p>
      <ul>
        {bullets.map(text => (
          <li key={text}>{text}</li>
        ))}
      </ul>
    </aside>
  )
}
