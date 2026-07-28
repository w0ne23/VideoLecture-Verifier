const API_BASE = '/api'

async function readError(res, fallback) {
  const data = await res.json().catch(() => null)
  return (data && (data.detail || data.error)) || (await res.text().catch(() => '')) || fallback
}

export async function uploadLecture({ file, title, description = '' }) {
  const formData = new FormData()
  formData.append('video', file)
  formData.append('title', title)
  formData.append('description', description)
  formData.append('workflow_mode', 'verify')

  const res = await fetch(`${API_BASE}/lectures`, { method: 'POST', body: formData })
  if (!res.ok) throw new Error(await readError(res, 'Upload failed'))
  return res.json() // { id, job_id, job_type, status, title, ... }
}

export async function listLectures(status = '') {
  const query = status ? `?status=${encodeURIComponent(status)}` : ''
  const res = await fetch(`${API_BASE}/lectures${query}`)
  if (!res.ok) throw new Error(await readError(res, 'Failed to fetch lectures'))
  return res.json() // [{ id, title, status, job_id, pipeline_stages, ... }]
}

export async function getLectureDetail(lectureId) {
  const res = await fetch(`${API_BASE}/lectures/${lectureId}`)
  if (!res.ok) throw new Error(await readError(res, 'Detail fetch failed'))
  return res.json() // { id, title, video_url, job: {...} }
}

export async function getLectureResult(lectureId) {
  const res = await fetch(`${API_BASE}/lectures/${lectureId}/result`)
  if (!res.ok) throw new Error(await readError(res, 'Result fetch failed'))
  return res.json()
}

export async function retryLecture(lectureId) {
  const res = await fetch(`${API_BASE}/lectures/${lectureId}/jobs?mode=verify`, { method: 'POST' })
  if (!res.ok) throw new Error(await readError(res, 'Retry failed'))
  return res.json() // { status, job_id, job_type }
}

export async function deleteLecture(lectureId) {
  const res = await fetch(`${API_BASE}/lectures/${lectureId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(await readError(res, 'Delete failed'))
  return res.json()
}

export function jobStreamUrl(lectureId) {
  return `${API_BASE}/lectures/${lectureId}/stream`
}

export async function checkHealth(signal) {
  const res = await fetch(`${API_BASE}/health`, { cache: 'no-store', signal })
  if (!res.ok) throw new Error(`status ${res.status}`)
  return res.json()
}
