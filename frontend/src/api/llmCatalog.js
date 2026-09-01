import { API_BASE, readError } from './client'

export async function getLlmCatalog({ refresh = false } = {}) {
  const query = refresh ? '?refresh=true' : ''
  const res = await fetch(`${API_BASE}/admin/llm-catalog${query}`)
  if (!res.ok) throw new Error(await readError(res, 'LiteLLM 모델 목록을 불러오지 못했습니다'))
  return res.json()
}
