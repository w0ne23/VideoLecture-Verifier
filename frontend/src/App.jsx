import { BrowserRouter, Routes, Route, Navigate, useLocation, useParams } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import SiteHeader from './components/layout/SiteHeader'
import SiteFooter from './components/layout/SiteFooter'
import MainPage from './pages/MainPage'
import VerifyEntryPage from './pages/VerifyEntryPage'
import UploadPage from './pages/UploadPage'
import LecturesPage from './pages/LecturesPage'
import VerifyProgressPage from './pages/VerifyProgressPage'
import ResultPage from './pages/ResultPage'
import VerifyStageReportPage from './pages/VerifyStageReportPage'
import VerifyDemoPage from './pages/VerifyDemoPage'
import VerifyDemoDiagramPage from './pages/VerifyDemoDiagramPage'
import VerifyDemoLivePage from './pages/VerifyDemoLivePage'
import StatsPage from './pages/StatsPage'
import ModelSetupLayout from './pages/model-setup/ModelSetupLayout'
import ModelSetupEntryPage from './pages/model-setup/ModelSetupEntryPage'
import ModelSetsPage from './pages/model-setup/ModelSetsPage'
import ModelRegistryPage from './pages/model-setup/ModelRegistryPage'

const queryClient = new QueryClient()

function RedirectPresetEdit() {
  const { profileId } = useParams()
  return <Navigate to={`/model-setup/sets?edit=${profileId}`} replace />
}

function AppShell() {
  const location = useLocation()
  const isHome = location.pathname === '/'

  return (
    <>
      <SiteHeader />
      <div className={isHome ? 'app app--home' : 'app'}>
        <main>
          <Routes>
            <Route path="/" element={<MainPage />} />
            <Route path="/verify" element={<VerifyEntryPage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/lectures" element={<LecturesPage />} />
            <Route path="/stats" element={<StatsPage />} />
            <Route path="/verify/:lectureId" element={<VerifyProgressPage />} />
            <Route path="/result/:lectureId" element={<ResultPage />} />
            <Route path="/result/:lectureId/stages" element={<VerifyStageReportPage />} />
            {/* 개발용: 다이어그램 파이프라인 + 실제 강의 결과물로 완성된 흐름을 보여주는 데모.
                실제 파이프라인이 아직 없어 결과는 하드코딩된 강의 하나의 실제 결과를 재사용한다.
                어디에도 링크하지 않고 개발자가 URL로 직접 들어가는 용도. */}
            <Route path="/dev/verify-demo" element={<VerifyDemoLivePage />} />
            {/* 개발용: 첨부된 아키텍처 다이어그램(전처리/검증, 병렬 두 갈래) 구조를 확인하는 데모. */}
            <Route path="/dev/verify-demo-diagram" element={<VerifyDemoDiagramPage />} />
            {/* 개발용: 파이프라인 진행 UI만 확인하는 데모(아이콘 애니메이션 포함). 실제 업로드/검증을 하지 않으며 어디에도 링크되지 않는다. */}
            <Route path="/dev/verify-demo-anime" element={<VerifyDemoPage />} />
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
      {isHome && <SiteFooter />}
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
