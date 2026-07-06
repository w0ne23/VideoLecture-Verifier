import { useCallback, useEffect, useState } from 'react'
import LectureDetail from './components/LectureDetail'
import LectureList from './components/LectureList'
import UploadForm from './components/UploadForm'

// 테스트용 초경량 해시 라우팅: '#/lecture/<id>' ↔ 홈. 새로고침해도 화면이 유지된다.
function lectureIdFromHash() {
  const match = window.location.hash.match(/^#\/lecture\/([\w-]+)/)
  return match ? match[1] : ''
}

export default function App() {
  const [lectureId, setLectureId] = useState(lectureIdFromHash)
  const [refreshKey, setRefreshKey] = useState(0)

  useEffect(() => {
    function onHashChange() {
      setLectureId(lectureIdFromHash())
    }
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const openLecture = useCallback(id => {
    window.location.hash = `#/lecture/${id}`
  }, [])

  const goHome = useCallback(() => {
    window.location.hash = ''
    setRefreshKey(key => key + 1)
  }, [])

  return (
    <div className="app">
      <header className="app-header">
        <h1 onClick={goHome}>VeriLec <span className="app-badge">test</span></h1>
        <p>강의 영상 검증 파이프라인 테스트 콘솔</p>
      </header>
      <main>
        {lectureId
          ? <LectureDetail lectureId={lectureId} onExit={goHome} />
          : (
            <>
              <UploadForm onUploaded={openLecture} />
              <h2 className="list-heading">강의 목록</h2>
              <LectureList onSelect={openLecture} refreshKey={refreshKey} />
            </>
          )}
      </main>
    </div>
  )
}
