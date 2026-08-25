// 불릿 문장 안에서 **강조할 부분**(오류 유형/출처/개수 등)만 색을 입혀 보여준다.
function renderBullet(text) {
  return text.split('**').map((part, index) => (
    index % 2 === 1
      ? <span className="stats-highlight" key={index}>{part}</span>
      : part
  ))
}

export default function StatsInsight({ title, bullets }) {
  return (
    <aside className="stats-insight">
      <h3>{title}</h3>
      <ul>
        {bullets.map((text, index) => (
          <li key={index}>{renderBullet(text)}</li>
        ))}
      </ul>
    </aside>
  )
}
