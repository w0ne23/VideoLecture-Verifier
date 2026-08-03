import { useCallback } from 'react'
import { BrowserRouter, Routes, Route, useNavigate, useParams } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import LectureDetail from './components/LectureDetail'
import LectureList from './components/LectureList'
import UploadForm from './components/UploadForm'

const queryClient = new QueryClient()

function HomePage() {
  const navigate = useNavigate()
  const openLecture = useCallback(id => navigate(`/lecture/${id}`), [navigate])

  return (
    <>
      <UploadForm onUploaded={openLecture} />
      <h2 className="list-heading">강의 목록</h2>
      <LectureList onSelect={openLecture} />
    </>
  )
}

function LectureDetailPage() {
  const { lectureId } = useParams()
  const navigate = useNavigate()
  const goHome = useCallback(() => navigate('/'), [navigate])

  return <LectureDetail lectureId={lectureId} onExit={goHome} />
}

function AppShell() {
  const navigate = useNavigate()
  const goHome = useCallback(() => navigate('/'), [navigate])

  return (
    <div className="app">
      <header className="app-header">
        <h1 onClick={goHome}>VeriLec <span className="app-badge">test</span></h1>
        <p>강의 영상 검증 파이프라인 테스트 콘솔</p>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/lecture/:lectureId" element={<LectureDetailPage />} />
        </Routes>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppShell />
      </BrowserRouter>
    </QueryClientProvider>
  )
}
