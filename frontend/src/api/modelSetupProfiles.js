import { API_BASE, readError } from './client'

export async function listModelSettingProfiles() {
  const res = await fetch(`${API_BASE}/admin/model-settings/profiles`)
  if (!res.ok) throw new Error(await readError(res, 'Failed to fetch model setting profiles'))
  return res.json()
}

export async function getModelSettingProfile(profileId) {
  const res = await fetch(`${API_BASE}/admin/model-settings/profiles/${profileId}`)
  if (!res.ok) throw new Error(await readError(res, 'Failed to fetch model setting profile'))
  return res.json()
}

export async function createModelSettingProfile(payload) {
  const res = await fetch(`${API_BASE}/admin/model-settings/profiles`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(await readError(res, 'Failed to create model setting profile'))
  return res.json()
}

export async function updateModelSettingProfile(profileId, payload) {
  const res = await fetch(`${API_BASE}/admin/model-settings/profiles/${profileId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(await readError(res, 'Failed to update model setting profile'))
  return res.json()
}

export async function deleteModelSettingProfile(profileId) {
  const res = await fetch(`${API_BASE}/admin/model-settings/profiles/${profileId}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error(await readError(res, 'Failed to delete model setting profile'))
  return res.json()
}

export async function applyModelSettingProfile(profileId) {
  const res = await fetch(`${API_BASE}/admin/model-settings/profiles/${profileId}/apply`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error(await readError(res, 'Failed to apply model setting profile'))
  return res.json()
}
