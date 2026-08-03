export const API_BASE = '/api'

export async function readError(res, fallback) {
  const data = await res.json().catch(() => null)
  return (data && (data.detail || data.error)) || (await res.text().catch(() => '')) || fallback
}
