import { useCallback } from 'react'
import { BrowserRouter, Routes, Route, useNavigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import MainPage from './pages/MainPage'
import UploadPage from './pages/UploadPage'
import VerifyProgressPage from './pages/VerifyProgressPage'
import ResultPage from './pages/ResultPage'
import StatsPage from './pages/StatsPage'
import ModelSetupLayout from './pages/model-setup/ModelSetupLayout'
import ModelSetupEntryPage from './pages/model-setup/ModelSetupEntryPage'
import ModelSetupPresetListPage from './pages/model-setup/ModelSetupPresetListPage'
import ModelSetupPresetEditorPage from './pages/model-setup/ModelSetupPresetEditorPage'
import ModelSetupFab from './components/model-setup/ModelSetupFab'

const queryClient = new QueryClient()

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
          <Route path="/stats" element={<StatsPage />} />
          <Route path="/verify/:lectureId" element={<VerifyProgressPage />} />
          <Route path="/result/:lectureId" element={<ResultPage />} />
          <Route path="/model-setup" element={<ModelSetupLayout />}>
            <Route index element={<ModelSetupEntryPage />} />
            <Route path="presets" element={<ModelSetupPresetListPage />} />
            <Route path="new" element={<ModelSetupPresetEditorPage mode="create" />} />
            <Route path="presets/:profileId/edit" element={<ModelSetupPresetEditorPage mode="edit" />} />
          </Route>
        </Routes>
      </main>
      <ModelSetupFab />
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
