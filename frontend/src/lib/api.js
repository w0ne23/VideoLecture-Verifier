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

  const res = await fetch(`${API_BASE}/jobs`, { method: 'POST', body: formData })
  if (!res.ok) throw new Error(await readError(res, 'Upload failed'))
  return res.json() // { id, job_id, job_type, status, title, ... }
}

export async function listJobs(status = '') {
  const query = status ? `?status=${encodeURIComponent(status)}` : ''
  const res = await fetch(`${API_BASE}/jobs${query}`)
  if (!res.ok) throw new Error(await readError(res, 'Failed to fetch jobs'))
  return res.json() // [{ id, title, status, job_id, pipeline_stages, ... }]
}

export async function getLectureDetail(lectureId) {
  const res = await fetch(`${API_BASE}/results/${lectureId}`)
  if (!res.ok) throw new Error(await readError(res, 'Detail fetch failed'))
  return res.json() // { id, title, video_url, is_verified, job: {...} }
}

export async function getLectureVerifier(lectureId) {
  const res = await fetch(`${API_BASE}/results/${lectureId}/verifier`)
  if (!res.ok) {
    if (res.status === 404) return null // 아직 결과 파일 없음
    throw new Error(await readError(res, 'Verifier fetch failed'))
  }
  return res.json()
}

export async function retryLecture(lectureId) {
  const res = await fetch(`${API_BASE}/jobs/${lectureId}/retry?mode=verify`, { method: 'POST' })
  if (!res.ok) throw new Error(await readError(res, 'Retry failed'))
  return res.json() // { status, job_id, job_type }
}

export async function deleteLecture(lectureId) {
  const res = await fetch(`${API_BASE}/jobs/${lectureId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(await readError(res, 'Delete failed'))
  return res.json()
}

// 주의: 이전 프로젝트와 달리 VeriLec 백엔드에서는 confirm이 /lectures 라우터에 있다.
export async function confirmLectureVerification(lectureId) {
  const res = await fetch(`${API_BASE}/lectures/${lectureId}/verify/confirm`, { method: 'POST' })
  if (!res.ok) throw new Error(await readError(res, 'Verification confirm failed'))
  return res.json()
}

export function jobStreamUrl(lectureId, jobId = '') {
  const params = new URLSearchParams()
  if (jobId) params.set('job_id', jobId)
  params.set('mode', 'verify')
  return `${API_BASE}/jobs/${lectureId}/stream?${params.toString()}`
}

export async function checkHealth(signal) {
  const res = await fetch(`${API_BASE}/health`, { cache: 'no-store', signal })
  if (!res.ok) throw new Error(`status ${res.status}`)
  return res.json()
}
