import { VERSION_TO_MODEL_ID, nextProviderId } from './stageModels'

export const CUSTOM_VERSION = '__custom__'

const REGISTRY_KEY = 'vlverifier_registered_llms'
const LEGACY_PROVIDERS_KEY = 'vlverifier_providers'

export function maskKey(key) {
  if (!key || key.length <= 8) return key || ''
  return `${key.slice(0, 8)}******${key.slice(-4)}`
}

/**
 * LiteLLM Catalog 응답을 화면·저장 모델에서 사용할 공통 메타데이터로 변환한다.
 * Provider와 모델 목록은 이 함수에 하드코딩하지 않고 Catalog 응답으로만 채운다.
 */
export function providerMeta(type, raw = {}) {
  const model = String(raw.modelId || raw.version || '').trim()
  return {
    name: String(raw.providerName || raw.name || type || 'Unknown provider'),
    protocol: String(raw.protocol || 'openai_chat_completions'),
    baseUrl: String(raw.baseUrl || ''),
    credentialRef: String(raw.credentialRef || ''),
    versions: model ? [model] : [],
    requiresKey: raw.requiresKey ?? raw.requires_key ?? true,
  }
}

export function providerRequiresKey(type, raw = {}) {
  return providerMeta(type, raw).requiresKey !== false
}

export function versionToModelId(version) {
  if (!version) return ''
  return VERSION_TO_MODEL_ID[version] || String(version).trim().toLowerCase().replace(/\s+/g, '-')
}

export function defaultEndpointConfig(type) {
  const meta = providerMeta(type)
  return {
    protocol: meta.protocol || 'openai_chat_completions',
    baseUrl: meta.baseUrl || '',
    credentialRef: meta.credentialRef || '',
    headers: {},
    timeout: { connectSec: 10, readSec: 180 },
    retry: { maxAttempts: 3, backoffSec: 2 },
    capabilities: {},
    providerOptions: {},
    enabled: true,
  }
}

function normalizeLlm(raw) {
  if (!raw || typeof raw !== 'object') return null
  const type = String(raw.type || '').trim()
  const version = String(raw.version || '').trim()
  if (!type || !version) return null
  const meta = providerMeta(type, raw)
  const endpoint = defaultEndpointConfig(type)
  return {
    id: String(raw.id || nextProviderId([])),
    type,
    version,
    modelId: String(raw.modelId || versionToModelId(version)),
    keyMasked: String(raw.keyMasked || ''),
    providerName: String(raw.providerName || meta.name),
    protocol: String(raw.protocol || endpoint.protocol),
    baseUrl: String(raw.baseUrl ?? endpoint.baseUrl),
    credentialRef: String(raw.credentialRef ?? endpoint.credentialRef),
    headers: raw.headers && typeof raw.headers === 'object' ? raw.headers : endpoint.headers,
    timeout: raw.timeout && typeof raw.timeout === 'object' ? raw.timeout : endpoint.timeout,
    retry: raw.retry && typeof raw.retry === 'object' ? raw.retry : endpoint.retry,
    capabilities: raw.capabilities && typeof raw.capabilities === 'object' ? raw.capabilities : endpoint.capabilities,
    providerOptions: raw.providerOptions && typeof raw.providerOptions === 'object' ? raw.providerOptions : endpoint.providerOptions,
    enabled: raw.enabled !== false,
  }
}

function migrateLegacyProviders(legacy) {
  if (!Array.isArray(legacy)) return []
  const migrated = []
  legacy.forEach(provider => {
    if (!provider || provider.isPresetPlaceholder) return
    const type = String(provider.type || '').trim()
    const version = String(provider.version || provider.modelId || '').trim()
    if (!type || !version) return
    const endpoint = defaultEndpointConfig(type)
    migrated.push({
      id: provider.id || nextProviderId(migrated),
      type,
      version,
      modelId: versionToModelId(version),
      keyMasked: provider.keyMasked || '',
      providerName: provider.providerName || type,
      ...endpoint,
    })
  })
  return migrated
}

export function loadRegisteredLlms() {
  try {
    const raw = JSON.parse(localStorage.getItem(REGISTRY_KEY) || 'null')
    if (Array.isArray(raw)) {
      return raw.map(normalizeLlm).filter(Boolean)
    }
  } catch {
    // fall through to legacy
  }

  try {
    const legacy = JSON.parse(localStorage.getItem(LEGACY_PROVIDERS_KEY) || 'null')
    const migrated = migrateLegacyProviders(legacy)
    if (migrated.length) {
      saveRegisteredLlms(migrated)
      return migrated
    }
  } catch {
    // ignore
  }
  return []
}

export function saveRegisteredLlms(llms) {
  const cleaned = (llms || []).map(normalizeLlm).filter(Boolean)
  localStorage.setItem(REGISTRY_KEY, JSON.stringify(cleaned))
  return cleaned
}

export function llmLabel(llm) {
  if (!llm) return ''
  const provider = providerMeta(llm.type, llm).name || llm.type
  return `${provider} · ${llm.version}`
}

export function nextLlmId(llms) {
  return nextProviderId(llms || [])
}

const STAGE_BINDING_KEYS = {
  claim: 'claim_extract',
  detect: 'issue_detect',
  classify: 'issue_classify',
  judge: 'verify',
  ground: 'grounding',
  slide: 'slide',
}

function endpointFromLlm(llm) {
  const defaults = defaultEndpointConfig(llm.type)
  return {
    id: llm.id,
    display_name: llm.displayName || llmLabel(llm),
    provider: llm.type,
    protocol: llm.protocol || defaults.protocol,
    base_url: llm.baseUrl || defaults.baseUrl,
    credential_ref: llm.credentialRef || defaults.credentialRef,
    headers: llm.headers || {},
    timeout: {
      connect_sec: Number(llm.timeout?.connectSec ?? 10),
      read_sec: Number(llm.timeout?.readSec ?? 180),
    },
    retry: {
      max_attempts: Number(llm.retry?.maxAttempts ?? 3),
      backoff_sec: Number(llm.retry?.backoffSec ?? 2),
    },
    capabilities: llm.capabilities || {},
    provider_options: llm.providerOptions || {},
    enabled: llm.enabled !== false,
  }
}

/**
 * 등록 모델과 단계 선택을 provider-neutral endpoint/stage binding 계약으로 변환한다.
 * stage_models는 기존 파이프라인 하위 호환을 위해 별도로 계속 생성한다.
 */
export function buildLlmConfig(providers, stages, stageOrder = Object.keys(STAGE_BINDING_KEYS)) {
  const endpoints = (providers || []).map(endpointFromLlm)
  const byId = new Map((providers || []).map(provider => [provider.id, provider]))
  const stageBindings = {}

  stageOrder.forEach(stageKey => {
    const stage = stages?.[stageKey]
    const bindingKey = STAGE_BINDING_KEYS[stageKey]
    if (!stage || !bindingKey) return

    const ids = stage.mode === 'multi' ? (stage.selected || []) : [stage.selected]
    const bindings = ids.map(providerId => {
      const provider = byId.get(providerId)
      if (!provider) return null
      const version = stage.mode === 'multi'
        ? stage.versions?.[providerId]
        : stage.version
      const model = versionToModelId(version || provider.modelId || provider.version)
      if (!model) return null
      return {
        endpoint_ref: providerId,
        model,
        weight: stage.mode === 'multi'
          ? Number(stage.weights?.[providerId] || 0)
          : 100,
      }
    }).filter(Boolean)

    if (bindings.length) stageBindings[bindingKey] = bindings
  })

  return {
    version: 1,
    endpoints,
    stage_bindings: stageBindings,
  }
}
