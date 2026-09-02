// 인사이트 카드 — buildInsight / buildDomainInsight 가 만든 { title, bullets[] } 렌더

// bullet 문자열에서 **강조** 로 감싼 부분(오류 유형·출처·개수 등)만 색을 입힘
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
