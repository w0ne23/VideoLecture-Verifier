import { VERSION_TO_MODEL_ID, nextProviderId } from './stageModels'

export const CUSTOM_VERSION = '__custom__'

export const PROVIDERS_META = {
  openai: {
    name: 'OpenAI',
    prefix: /^sk-(?!ant-)/,
    protocol: 'openai_chat_completions',
    baseUrl: 'https://api.openai.com/v1',
    credentialRef: 'OPENAI_API_KEY',
    versions: ['GPT-5.4', 'GPT-5.4 Mini', 'GPT-5.4 Nano'],
  },
  anthropic: {
    name: 'Anthropic',
    prefix: /^sk-ant-/,
    protocol: 'anthropic_messages',
    baseUrl: 'https://api.anthropic.com',
    credentialRef: 'ANTHROPIC_API_KEY',
    versions: ['Claude Opus 4.8', 'Claude Sonnet 5', 'Claude Haiku 4.5'],
  },
  xai: {
    name: 'xAI',
    prefix: /^xai-/,
    protocol: 'openai_chat_completions',
    baseUrl: 'https://api.x.ai/v1',
    credentialRef: 'XAI_API_KEY',
    versions: ['Grok-4', 'Grok-4 Mini'],
  },
  gemini: {
    name: 'Gemini',
    prefix: /^AIza/,
    protocol: 'gemini_generate_content',
    baseUrl: '',
    credentialRef: 'GOOGLE_API_KEY_1',
    versions: ['Gemini 3 Pro', 'Gemini 3 Flash', 'Gemini 3 Flash Lite'],
  },
  deepseek: {
    name: 'DeepSeek',
    // OpenAI 키와 접두사가 같아 자동 감지는 지원하지 않음(provider를 직접 선택해야 함)
    prefix: null,
    protocol: 'openai_chat_completions',
    baseUrl: 'https://api.deepseek.com/v1',
    credentialRef: 'DEEPSEEK_API_KEY',
    versions: ['DeepSeek V4', 'DeepSeek V4 Flash'],
  },
  groq: {
    name: 'Groq',
    prefix: /^gsk_/,
    protocol: 'openai_chat_completions',
    baseUrl: 'https://api.groq.com/openai/v1',
    credentialRef: 'GROQ_API_KEY',
    versions: [],
  },
  ollama: {
    name: 'Ollama',
    prefix: null,
    protocol: 'openai_chat_completions',
    baseUrl: 'http://localhost:11434/v1',
    credentialRef: '',
    versions: ['Gemma 3 27B', 'Gemma 3 9B', 'Qwen 3 32B', 'Qwen 3 14B'],
    requiresKey: false,
  },
  vllm: {
    name: 'vLLM',
    prefix: null,
    protocol: 'openai_chat_completions',
    baseUrl: 'http://localhost:8000/v1',
    credentialRef: 'VLLM_API_KEY',
    versions: [],
  },
  custom: {
    name: 'Custom OpenAI-compatible',
    prefix: null,
    protocol: 'openai_chat_completions',
    baseUrl: '',
    credentialRef: '',
    versions: [],
  },
}

const REGISTRY_KEY = 'vlverifier_registered_llms'
const LEGACY_PROVIDERS_KEY = 'vlverifier_providers'

export function maskKey(key) {
  if (!key || key.length <= 8) return key || ''
  return `${key.slice(0, 8)}******${key.slice(-4)}`
}

export function detectProvider(key) {
  if (PROVIDERS_META.anthropic.prefix.test(key)) return 'anthropic'
  if (PROVIDERS_META.xai.prefix.test(key)) return 'xai'
  if (PROVIDERS_META.gemini.prefix.test(key)) return 'gemini'
  if (PROVIDERS_META.openai.prefix.test(key)) return 'openai'
  return null
}

export function providerRequiresKey(type) {
  return PROVIDERS_META[type]?.requiresKey !== false
}

export function versionToModelId(version) {
  if (!version) return ''
  return VERSION_TO_MODEL_ID[version] || String(version).trim().toLowerCase().replace(/\s+/g, '-')
}

export function defaultEndpointConfig(type) {
  const meta = PROVIDERS_META[type] || PROVIDERS_META.custom
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
  if (!type || !version || !PROVIDERS_META[type]) return null
  const endpoint = defaultEndpointConfig(type)
  return {
    id: String(raw.id || nextProviderId([])),
    type,
    version,
    modelId: String(raw.modelId || versionToModelId(version)),
    keyMasked: String(raw.keyMasked || ''),
    providerName: PROVIDERS_META[type].name,
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
    const type = provider.type
    const meta = PROVIDERS_META[type]
    if (!meta) return
    const version = meta.versions[0]
    const endpoint = defaultEndpointConfig(type)
    migrated.push({
      id: provider.id || nextProviderId(migrated),
      type,
      version,
      modelId: versionToModelId(version),
      keyMasked: provider.keyMasked || '',
      providerName: meta.name,
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
  const provider = PROVIDERS_META[llm.type]?.name || llm.type
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
