import { VERSION_TO_MODEL_ID, nextProviderId } from './stageModels'

export const CUSTOM_VERSION = '__custom__'

export const PROVIDERS_META = {
  openai: {
    name: 'OpenAI',
    prefix: /^sk-(?!ant-)/,
    versions: ['GPT-5.4', 'GPT-5.4 Mini', 'GPT-5.4 Nano'],
  },
  anthropic: {
    name: 'Anthropic',
    prefix: /^sk-ant-/,
    versions: ['Claude Opus 4.8', 'Claude Sonnet 5', 'Claude Haiku 4.5'],
  },
  xai: {
    name: 'xAI',
    prefix: /^xai-/,
    versions: ['Grok-4', 'Grok-4 Mini'],
  },
  gemini: {
    name: 'Gemini',
    prefix: /^AIza/,
    versions: ['Gemini 3 Pro', 'Gemini 3 Flash', 'Gemini 3 Flash Lite'],
  },
  deepseek: {
    name: 'DeepSeek',
    // OpenAI 키와 접두사가 같아 자동 감지는 지원하지 않음(provider를 직접 선택해야 함)
    prefix: null,
    versions: ['DeepSeek V4', 'DeepSeek V4 Flash'],
  },
  ollama: {
    name: 'Ollama',
    prefix: null,
    versions: ['Gemma 3 27B', 'Gemma 3 9B', 'Qwen 3 32B', 'Qwen 3 14B'],
    requiresKey: false,
  },
}

const REGISTRY_KEY = 'verilec_registered_llms'
const LEGACY_PROVIDERS_KEY = 'verilec_providers'

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

function normalizeLlm(raw) {
  if (!raw || typeof raw !== 'object') return null
  const type = String(raw.type || '').trim()
  const version = String(raw.version || '').trim()
  if (!type || !version || !PROVIDERS_META[type]) return null
  return {
    id: String(raw.id || nextProviderId([])),
    type,
    version,
    modelId: String(raw.modelId || versionToModelId(version)),
    keyMasked: String(raw.keyMasked || ''),
    providerName: PROVIDERS_META[type].name,
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
    migrated.push({
      id: provider.id || nextProviderId(migrated),
      type,
      version,
      modelId: versionToModelId(version),
      keyMasked: provider.keyMasked || '',
      providerName: meta.name,
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
