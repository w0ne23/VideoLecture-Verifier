import { API_BASE, readError } from './client'

/** Store an API key server-side and return only its reference and masked label. */
export async function saveLlmCredential({ provider, model, apiKey }) {
  const res = await fetch(`${API_BASE}/admin/llm-credentials`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider, model, api_key: apiKey }),
  })
  if (!res.ok) throw new Error(await readError(res, 'API 키를 안전하게 저장하지 못했습니다.'))
  return res.json()
}
