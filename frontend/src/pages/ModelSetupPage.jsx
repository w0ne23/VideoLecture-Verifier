import { useQuery } from '@tanstack/react-query'
import { getModelSettings } from '../api/modelSetup'

// 라우팅/API 배관만 갖춘 스켈레톤. 스테이지별 입력 폼(단일 모델 vs 앙상블 리스트)은
// 실제 화면 설계 시 여기를 채운다.
export default function ModelSetupPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['model-settings'],
    queryFn: getModelSettings,
  })

  return (
    <div className="model-setup">
      <h2 className="list-heading">Multi-LLM 설정</h2>
      <p className="list-note">준비 중입니다.</p>
      {isLoading && <p className="list-note">불러오는 중...</p>}
      {error && <p className="error-text">설정 조회 실패: {String(error?.message || error)}</p>}
      {data && (
        <pre className="claim-debug-json">
          <code>{JSON.stringify(data, null, 2)}</code>
        </pre>
      )}
    </div>
  )
}
