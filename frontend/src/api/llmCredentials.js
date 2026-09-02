// LLM API 키 저장 API (/api/admin/llm-credentials)

import { API_BASE, readError } from './client'

// API 키를 서버에 저장하고 참조값 + 마스킹 라벨만 돌려받음 (원본 키는 클라이언트에 남기지 않음)
export async function saveLlmCredential({ provider, model, apiKey }) {
  const res = await fetch(`${API_BASE}/admin/llm-credentials`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider, model, api_key: apiKey }),
  })
  if (!res.ok) throw new Error(await readError(res, 'API 키를 안전하게 저장하지 못했습니다.'))
  return res.json()
}
