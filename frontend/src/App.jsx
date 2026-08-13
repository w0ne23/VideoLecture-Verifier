import { useCallback } from 'react'
import { BrowserRouter, Routes, Route, Navigate, useNavigate, useParams } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import MainPage from './pages/MainPage'
import VerifyEntryPage from './pages/VerifyEntryPage'
import UploadPage from './pages/UploadPage'
import LecturesPage from './pages/LecturesPage'
import VerifyProgressPage from './pages/VerifyProgressPage'
import ResultPage from './pages/ResultPage'
import StatsPage from './pages/StatsPage'
import ModelSetupLayout from './pages/model-setup/ModelSetupLayout'
import ModelSetupEntryPage from './pages/model-setup/ModelSetupEntryPage'
import ModelSetsPage from './pages/model-setup/ModelSetsPage'
import ModelRegistryPage from './pages/model-setup/ModelRegistryPage'
import ModelSetupFab from './components/model-setup/ModelSetupFab'

const queryClient = new QueryClient()

function RedirectPresetEdit() {
  const { profileId } = useParams()
  return <Navigate to={`/model-setup/sets?edit=${profileId}`} replace />
}

function AppShell() {
  const navigate = useNavigate()
  const goHome = useCallback(() => navigate('/'), [navigate])

  return (
    <>
      <header className="app-header">
        <div className="app-header-inner">
          <h1 onClick={goHome}>
            <span className="app-logo-mark" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none">
                <path d="M12 2.5l3.4 6.4-3.4 1.1-3.4-1.1 3.4-6.4z" fill="currentColor" />
                <rect x="9" y="9.5" width="6" height="9.5" rx="1.4" fill="currentColor" opacity="0.85" />
                <path d="M6.5 21h11" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
              </svg>
            </span>
            LID
          </h1>
          <p>강의 영상 속 오류를 찾아 품질 향상으로 이끈다!</p>
        </div>
      </header>
      <div className="app">
        <main>
          <Routes>
            <Route path="/" element={<MainPage />} />
            <Route path="/verify" element={<VerifyEntryPage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/lectures" element={<LecturesPage />} />
            <Route path="/stats" element={<StatsPage />} />
            <Route path="/verify/:lectureId" element={<VerifyProgressPage />} />
            <Route path="/result/:lectureId" element={<ResultPage />} />
            <Route path="/model-setup" element={<ModelSetupLayout />}>
              <Route index element={<ModelSetupEntryPage />} />
              <Route path="models" element={<ModelRegistryPage />} />
              <Route path="sets" element={<ModelSetsPage />} />
              {/* 이전 경로 호환 */}
              <Route path="sets/new" element={<Navigate to="/model-setup/sets" replace />} />
              <Route path="sets/:profileId/edit" element={<RedirectPresetEdit />} />
              <Route path="presets" element={<Navigate to="/model-setup/sets" replace />} />
              <Route path="new" element={<Navigate to="/model-setup/sets" replace />} />
              <Route path="presets/:profileId/edit" element={<RedirectPresetEdit />} />
            </Route>
          </Routes>
        </main>
      </div>
      <ModelSetupFab />
    </>
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
