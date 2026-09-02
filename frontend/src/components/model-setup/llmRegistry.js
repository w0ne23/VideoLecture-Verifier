// 등록된 LLM(provider + model + 마스킹 키) 목록 관리 (localStorage) + 백엔드 계약 변환

import { VERSION_TO_MODEL_ID, nextProviderId } from './stageModels'

// 버전 선택에서 "직접 입력" 을 나타내는 sentinel
export const CUSTOM_VERSION = '__custom__'

const REGISTRY_KEY = 'vlverifier_registered_llms'
const LEGACY_PROVIDERS_KEY = 'vlverifier_providers'

// API 키를 앞 8자 + 뒤 4자만 남기고 마스킹 (원본은 서버에만 저장)
export function maskKey(key) {
  if (!key || key.length <= 8) return key || ''
  return `${key.slice(0, 8)}******${key.slice(-4)}`
}

// LiteLLM Catalog 응답 → 화면·저장 공통 메타데이터
// provider·모델 목록을 여기 하드코딩하지 않고 Catalog 응답으로만 채움
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

// 이 provider 가 API 키를 요구하는지 (로컬 ollama 등은 false)
export function providerRequiresKey(type, raw = {}) {
  return providerMeta(type, raw).requiresKey !== false
}

// 표시명 → 모델 ID (매핑에 없으면 소문자·하이픈으로 슬러그화)
export function versionToModelId(version) {
  if (!version) return ''
  return VERSION_TO_MODEL_ID[version] || String(version).trim().toLowerCase().replace(/\s+/g, '-')
}

// endpoint 설정 기본값 (프로토콜·타임아웃·재시도 등)
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

// 저장/로드 시 LLM 레코드를 표준 형태로 정리 (type·version 없으면 버림)
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

// 구 localStorage 키(vlverifier_providers)의 provider 배열 → 현재 LLM 레코드 형태
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

// 등록 LLM 목록 로드 — 현재 키 우선, 없으면 구 키에서 마이그레이션
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

// 등록 LLM 목록 저장 (정규화 후 기록)
export function saveRegisteredLlms(llms) {
  const cleaned = (llms || []).map(normalizeLlm).filter(Boolean)
  localStorage.setItem(REGISTRY_KEY, JSON.stringify(cleaned))
  return cleaned
}

// "provider · version" 표시 라벨
export function llmLabel(llm) {
  if (!llm) return ''
  const provider = providerMeta(llm.type, llm).name || llm.type
  return `${provider} · ${llm.version}`
}

export function nextLlmId(llms) {
  return nextProviderId(llms || [])
}

// UI 단계 키 → llm_config 의 stage_bindings 키
const STAGE_BINDING_KEYS = {
  claim: 'claim_extract',
  detect: 'issue_detect',
  classify: 'issue_classify',
  judge: 'verify',
  ground: 'grounding',
  slide: 'slide',
}

// LLM 레코드 → llm_config 의 endpoint 객체 (snake_case)
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

// 등록 모델 + 단계 선택 → provider 중립 endpoint/stage_binding 계약
// stage_models 는 기존 파이프라인 하위 호환용으로 별도 생성 (stagesToStageModels)
export function buildLlmConfig(providers, stages, stageOrder = Object.keys(STAGE_BINDING_KEYS)) {
  const endpoints = (providers || []).map(endpointFromLlm)
  const byId = new Map((providers || []).map(provider => [provider.id, provider]))
  const stageBindings = {}

  stageOrder.forEach(stageKey => {
    const stage = stages?.[stageKey]
    const bindingKey = STAGE_BINDING_KEYS[stageKey]
    if (!stage || !bindingKey) return

    // single 은 모델 1개(가중치 100), multi 는 선택된 모델별 가중치
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
