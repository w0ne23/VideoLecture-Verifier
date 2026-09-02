// 라우트 정의 + 전역 provider(react-query, 라우터) 마운트

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
import NotFoundPage from './pages/NotFoundPage'
import ModelSetupLayout from './pages/model-setup/ModelSetupLayout'
import ModelSetupEntryPage from './pages/model-setup/ModelSetupEntryPage'
import ModelSetsPage from './pages/model-setup/ModelSetsPage'
import ModelRegistryPage from './pages/model-setup/ModelRegistryPage'

// 앱 전체가 공유하는 react-query 캐시 (서버 상태 조회·무효화의 기준점)
const queryClient = new QueryClient()

// 구 preset 편집 경로(/model-setup/sets/:profileId/edit 등)를 통합 편집 화면으로 리다이렉트
function RedirectPresetEdit() {
  const { profileId } = useParams()
  return <Navigate to={`/model-setup/sets?edit=${profileId}`} replace />
}

// 공통 레이아웃(헤더/푸터) + 라우트 테이블
// 홈(/)만 전용 레이아웃 클래스와 푸터를 적용
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
            {/* 개발용 데모 — 어디에도 링크하지 않고 개발자가 URL 로 직접 진입 */}
            {/* 다이어그램 파이프라인 + 완성된 흐름. 실제 파이프라인이 없어 하드코딩된 강의 하나의 실제 결과를 재사용 */}
            <Route path="/dev/verify-demo" element={<VerifyDemoLivePage />} />
            {/* 아키텍처 다이어그램(전처리/검증, 병렬 두 갈래) 구조 확인용 */}
            <Route path="/dev/verify-demo-diagram" element={<VerifyDemoDiagramPage />} />
            {/* 파이프라인 진행 UI(아이콘 애니메이션)만 확인용. 실제 업로드/검증 없음 */}
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
            {/* 매칭되는 라우트 없음 — nginx 가 모든 경로에 index.html 을 내려주므로(SPA 새로고침 지원) 잘못된 주소도 여기로 도달 */}
            <Route path="*" element={<NotFoundPage />} />
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
