import { API_BASE, readError } from './client'

/**
 * 통계 페이지 집계.
 * { lecture_count, by_tag[], by_domain[], by_duration[] }
 * - by_tag / by_domain 행: { key, typeDist, total, lectureCount }
 * - by_duration 행: { key, label, lectureMin, preprocessMin, verifyMin, total, lectureCount }
 */
export async function getStats() {
  const res = await fetch(`${API_BASE}/stats`)
  if (!res.ok) throw new Error(await readError(res, '통계를 불러오지 못했습니다.'))
  return res.json()
}
