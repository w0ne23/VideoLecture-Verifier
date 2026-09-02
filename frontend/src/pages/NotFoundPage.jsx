// 404 — 매칭되는 라우트가 없을 때 (App.jsx 의 path="*")

import { useNavigate } from 'react-router-dom'

export default function NotFoundPage() {
  const navigate = useNavigate()

  return (
    <section className="not-found">
      <p className="not-found-code">404</p>
      <h2>페이지를 찾을 수 없습니다</h2>
      <p className="list-note">요청하신 주소에 해당하는 화면이 없습니다.</p>
      <button type="button" className="btn" onClick={() => navigate('/')}>
        메인으로
      </button>
    </section>
  )
}
