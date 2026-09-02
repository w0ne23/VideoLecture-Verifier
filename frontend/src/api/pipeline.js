// 강의 업로드·조회·검증 API — 모두 /api/lectures 라우터

import { API_BASE, readError } from './client'

// 영상 업로드 → 검증 job 생성
// 반환: { id, job_id, job_type, status, title, source_tag, ... }
export async function uploadLecture({ file, title, description = '', sourceTag }) {
  const formData = new FormData()
  formData.append('video', file)
  formData.append('title', title)
  formData.append('description', description)
  formData.append('source_tag', sourceTag)
  formData.append('workflow_mode', 'verify')

  const res = await fetch(`${API_BASE}/lectures`, { method: 'POST', body: formData })
  if (!res.ok) throw new Error(await readError(res, 'Upload failed'))
  return res.json()
}

// 강의 목록 조회 (status 지정 시 해당 상태만)
// 반환: [{ id, title, status, job_id, pipeline_stages, ... }]
export async function listLectures(status = '') {
  const query = status ? `?status=${encodeURIComponent(status)}` : ''
  const res = await fetch(`${API_BASE}/lectures${query}`)
  if (!res.ok) throw new Error(await readError(res, 'Failed to fetch lectures'))
  return res.json()
}

// 강의 단건 메타 조회
// 반환: { id, title, video_url, job: {...} }
export async function getLectureDetail(lectureId) {
  const res = await fetch(`${API_BASE}/lectures/${lectureId}`)
  if (!res.ok) throw new Error(await readError(res, 'Detail fetch failed'))
  return res.json()
}

// 최종 검증 결과(feedback_items, summary 등) 조회
export async function getLectureResult(lectureId) {
  const res = await fetch(`${API_BASE}/lectures/${lectureId}/result`)
  if (!res.ok) throw new Error(await readError(res, 'Result fetch failed'))
  return res.json()
}

// 검증 단계별 중간 산출물(raw JSON) 조회 — "검증 과정 보기" 화면에서 사용
// stage: claim_extraction | issue_judge | issue_classification | web_grounding | final_verification | slide_review
export async function getLectureArtifact(lectureId, stage) {
  const res = await fetch(`${API_BASE}/lectures/${lectureId}/artifacts/${stage}`)
  if (!res.ok) throw new Error(await readError(res, 'Artifact fetch failed'))
  return res.json()
}

// 검증 재실행 job 생성
// 반환: { status, job_id, job_type }
export async function retryLecture(lectureId) {
  const res = await fetch(`${API_BASE}/lectures/${lectureId}/jobs?mode=verify`, { method: 'POST' })
  if (!res.ok) throw new Error(await readError(res, 'Retry failed'))
  return res.json()
}

// 강의 + 산출물 삭제
export async function deleteLecture(lectureId) {
  const res = await fetch(`${API_BASE}/lectures/${lectureId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(await readError(res, 'Delete failed'))
  return res.json()
}

// job 진행 상황 SSE 스트림 URL (EventSource 로 구독)
export function jobStreamUrl(lectureId) {
  return `${API_BASE}/lectures/${lectureId}/stream`
}

// 백엔드 헬스체크 — signal 로 취소 가능, 응답 캐시 안 함
export async function checkHealth(signal) {
  const res = await fetch(`${API_BASE}/health`, { cache: 'no-store', signal })
  if (!res.ok) throw new Error(`status ${res.status}`)
  return res.json()
}
