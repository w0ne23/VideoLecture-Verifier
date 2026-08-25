import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

export default function SiteHeader() {
  const navigate = useNavigate()
  const [isScrolled, setIsScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setIsScrolled(window.scrollY > 8)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <div className={`site-header${isScrolled ? ' site-header--scrolled' : ''}`}>
      <div className="site-header-inner">
        <button type="button" className="site-header-brand" onClick={() => navigate('/')}>
          <span className="site-header-brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <path d="M9 12l2 2 4-4" />
              <circle cx="12" cy="12" r="9" />
            </svg>
          </span>
          <span className="site-header-brand-word">VeriLec</span>
        </button>
        <div className="site-header-links">
          <button type="button" onClick={() => navigate('/model-setup/sets')}>Multi-LLM 설정</button>
          <button type="button" onClick={() => navigate('/lectures')}>강의 목록</button>
          <button type="button" onClick={() => navigate('/stats')}>통계</button>
        </div>
      </div>
    </div>
  )
}
