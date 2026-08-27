import { API_BASE, readError } from './client'

export async function getModelSettings() {
  const res = await fetch(`${API_BASE}/admin/model-settings`)
  if (!res.ok) throw new Error(await readError(res, 'Failed to fetch model settings'))
  return res.json() // { stage_models: { [envKey]: string }, mode: 'fixed' | 'generic' }
}

export async function updateModelSettings(stageModels) {
  const res = await fetch(`${API_BASE}/admin/model-settings`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ stage_models: stageModels }),
  })
  if (!res.ok) throw new Error(await readError(res, 'Failed to update model settings'))
  return res.json()
}
