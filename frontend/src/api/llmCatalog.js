// LiteLLM 게이트웨이에 등록된 모델 카탈로그 조회 API (/api/admin/llm-catalog)

import { API_BASE, readError } from './client'

// 선택 가능한 provider/model 목록 조회 — refresh=true 면 게이트웨이에서 다시 읽어옴
export async function getLlmCatalog({ refresh = false } = {}) {
  const query = refresh ? '?refresh=true' : ''
  const res = await fetch(`${API_BASE}/admin/llm-catalog${query}`)
  if (!res.ok) throw new Error(await readError(res, 'LiteLLM 모델 목록을 불러오지 못했습니다'))
  return res.json()
}
