// API 호출 공통 유틸 — 베이스 경로 + 실패 응답 파싱

export const API_BASE = '/api'

// 실패 응답에서 사람이 읽을 메시지 추출 (JSON 의 detail/error → 본문 텍스트 → fallback 순)
export async function readError(res, fallback) {
  const data = await res.json().catch(() => null)
  return (data && (data.detail || data.error)) || (await res.text().catch(() => '')) || fallback
}
