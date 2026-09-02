// Multi-LLM 조합(프로필) CRUD + 적용 API (/api/admin/model-settings/profiles)
// 프로필 하나 = 단계별 모델 조합 한 세트, 그중 하나만 "적용 중" 상태

import { API_BASE, readError } from './client'

// 프로필 목록 조회
export async function listModelSettingProfiles() {
  const res = await fetch(`${API_BASE}/admin/model-settings/profiles`)
  if (!res.ok) throw new Error(await readError(res, 'Failed to fetch model setting profiles'))
  return res.json()
}

// 프로필 생성
export async function createModelSettingProfile(payload) {
  const res = await fetch(`${API_BASE}/admin/model-settings/profiles`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(await readError(res, 'Failed to create model setting profile'))
  return res.json()
}

// 프로필 수정
export async function updateModelSettingProfile(profileId, payload) {
  const res = await fetch(`${API_BASE}/admin/model-settings/profiles/${profileId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(await readError(res, 'Failed to update model setting profile'))
  return res.json()
}

// 프로필 삭제
export async function deleteModelSettingProfile(profileId) {
  const res = await fetch(`${API_BASE}/admin/model-settings/profiles/${profileId}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error(await readError(res, 'Failed to delete model setting profile'))
  return res.json()
}

// 프로필을 "적용 중" 으로 전환 — 이후 검증 job 이 이 조합을 사용
export async function applyModelSettingProfile(profileId) {
  const res = await fetch(`${API_BASE}/admin/model-settings/profiles/${profileId}/apply`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error(await readError(res, 'Failed to apply model setting profile'))
  return res.json()
}
