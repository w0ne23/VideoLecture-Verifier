import { useCallback } from 'react'
import { BrowserRouter, Routes, Route, useNavigate, useParams } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import LectureDetail from './components/LectureDetail'
import MainPage from './pages/MainPage'
import UploadPage from './pages/UploadPage'

const queryClient = new QueryClient()

function LectureDetailPage() {
  const { lectureId } = useParams()
  const navigate = useNavigate()
  const goToList = useCallback(() => navigate('/upload'), [navigate])

  return <LectureDetail lectureId={lectureId} onExit={goToList} />
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
          <Route path="/" element={<MainPage />} />
          <Route path="/upload" element={<UploadPage />} />
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
