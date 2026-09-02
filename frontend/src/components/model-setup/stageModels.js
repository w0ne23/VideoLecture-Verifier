// Multi-LLM 설정 화면의 단계 상태 ↔ 백엔드 형식(stage_models / editor_state) 변환 유틸

// UI 표시명 → 파이프라인이 읽는 모델 ID
export const VERSION_TO_MODEL_ID = {
  'GPT-5.4': 'gpt-5.4',
  'GPT-5.4 Mini': 'gpt-5.4-mini',
  'GPT-5.4 Nano': 'gpt-5.4-nano',
  'Claude Opus 4.8': 'claude-opus-4.8',
  'Claude Sonnet 5': 'claude-sonnet-4.5',
  'Claude Haiku 4.5': 'claude-haiku-4.5',
  'Grok-4': 'grok-4',
  'Grok-4 Mini': 'grok-4-mini',
  'Gemini 3 Pro': 'gemini-3-pro',
  'Gemini 3 Flash': 'gemini-3-flash',
  'Gemini 3 Flash Lite': 'gemini-3-flash-lite',
  'DeepSeek V4': 'deepseek-v4',
  'DeepSeek V4 Flash': 'deepseek-v4-flash',
  'Gemma 3 27B': 'gemma3:27b',
  'Gemma 3 9B': 'gemma3:9b',
  'Qwen 3 32B': 'qwen3:32b',
  'Qwen 3 14B': 'qwen3:14b',
}

// 모델 ID → UI 표시명 (역방향)
export const MODEL_ID_TO_VERSION = Object.fromEntries(
  Object.entries(VERSION_TO_MODEL_ID).map(([version, modelId]) => [modelId, version]),
)

// UI 단계 → 백엔드 stage_models 키 — slide 는 동일 값을 두 키에 넣음
export const STAGE_ENV_KEYS = {
  claim: ['VERIFIER_CLAIM_EXTRACT_MODEL'],
  detect: ['ISSUE_JUDGE_MODELS'],
  classify: ['ISSUE_TYPE_CLASSIFIER_MODELS'],
  judge: ['CLASSIFIED_ISSUE_VERIFIER_MODELS'],
  ground: ['CLASSIFIED_ISSUE_GROUNDING_MODELS'],
  slide: ['VERIFIER_SLIDE_ERROR_MODEL', 'VERIFIER_SLIDE_ERROR_TRANSCRIBE_MODEL'],
}

export const STAGE_LABELS = {
  claim: 'Claim 추출',
  detect: 'Issue 탐지',
  classify: '유형 분류',
  judge: 'Issue 판단',
  ground: '웹 그라운딩',
  slide: '슬라이드 오류',
}

// 모델 ID 접두사로 provider 종류 추론 (콜론 포함 = 로컬 ollama)
export function modelIdToProviderType(modelId) {
  const id = String(modelId || '').trim().toLowerCase()
  if (!id) return null
  if (id.startsWith('gpt') || id.startsWith('o1') || id.startsWith('o3')) return 'openai'
  if (
    id.startsWith('claude')
    || id.startsWith('sonnet')
    || id.startsWith('haiku')
    || id.startsWith('opus')
  ) {
    return 'anthropic'
  }
  if (id.startsWith('grok') || id.startsWith('xai')) return 'xai'
  if (id.startsWith('gemini')) return 'gemini'
  if (id.startsWith('deepseek')) return 'deepseek'
  if (id.includes(':')) return 'ollama'
  return null
}

// 표시명 → 모델 ID (매핑에 없으면 소문자·하이픈으로 슬러그화)
function versionToModelId(version) {
  if (!version) return ''
  return VERSION_TO_MODEL_ID[version] || String(version).trim().toLowerCase().replace(/\s+/g, '-')
}

// 모델 ID → 표시명 — 매핑에 없으면 provider 메타의 versions 목록에서 역추적
function modelIdToVersion(modelId, providerType, providersMeta) {
  if (MODEL_ID_TO_VERSION[modelId]) return MODEL_ID_TO_VERSION[modelId]
  const meta = providersMeta?.[providerType]
  if (!meta) return modelId
  const match = meta.versions.find(version => versionToModelId(version) === modelId)
  return match || meta.versions[0] || modelId
}

// 쉼표/공백으로 구분된 모델 ID 문자열 → 배열
function splitModels(raw) {
  return String(raw || '')
    .split(/[,\s]+/)
    .map(part => part.trim())
    .filter(Boolean)
}

// 기존 provider id(p1, p2, ...) 중 최대값 + 1 로 새 id 생성
export function nextProviderId(providers) {
  const maxId = providers.reduce((max, provider) => {
    const numericId = Number.parseInt(String(provider.id || '').replace('p', ''), 10)
    return Number.isFinite(numericId) ? Math.max(max, numericId) : max
  }, 0)
  return `p${maxId + 1}`
}

// React 단계 상태 → PUT /admin/model-settings 의 stage_models (envKey → "modelId,modelId" 문자열)
export function stagesToStageModels(stages, providers, stageOrder) {
  const stageModels = {}

  stageOrder.forEach(stageKey => {
    const stage = stages[stageKey]
    const envKeys = STAGE_ENV_KEYS[stageKey]
    if (!stage || !envKeys) return

    let modelIds = []
    if (stage.mode === 'single') {
      if (!stage.selected || !stage.version) return
      const provider = providers.find(item => item.id === stage.selected)
      if (!provider) return
      modelIds = [versionToModelId(stage.version)]
    } else {
      if (!stage.selected?.length) return
      modelIds = stage.selected
        .map(providerId => {
          const provider = providers.find(item => item.id === providerId)
          if (!provider) return null
          return versionToModelId(stage.versions?.[providerId])
        })
        .filter(Boolean)
    }

    if (!modelIds.length) return
    const value = modelIds.join(',')
    envKeys.forEach(envKey => {
      stageModels[envKey] = value
    })
  })

  return stageModels
}

// GET stage_models → React 단계 초기값에 반영 — 모델 ID 를 등록된 provider(type)에만 매칭
export function applyStageModelsToStages(baseStages, stageModels, providers, providersMeta) {
  if (!stageModels || !providers.length) return baseStages

  const next = { ...baseStages }

  Object.entries(STAGE_ENV_KEYS).forEach(([stageKey, envKeys]) => {
    const stage = next[stageKey]
    if (!stage) return

    // slide 는 두 키가 같은 값이므로 첫 키만 읽음
    const raw = stageModels[envKeys[0]]
    const modelIds = splitModels(raw)
    if (!modelIds.length) return

    // 같은 종류 provider 가 여러 개면 아직 안 쓴 것을 우선 배정, 없으면 첫 번째 재사용
    const usedInStage = new Set()
    const takeProvider = providerType => {
      const match = providers.find(
        provider => provider.type === providerType && !usedInStage.has(provider.id),
      )
      if (match) {
        usedInStage.add(match.id)
        return match
      }
      return providers.find(provider => provider.type === providerType) || null
    }

    if (stage.mode === 'single') {
      const modelId = modelIds[0]
      const providerType = modelIdToProviderType(modelId)
      const provider = providerType ? takeProvider(providerType) : null
      if (!provider) return
      next[stageKey] = {
        ...stage,
        selected: provider.id,
        version: modelIdToVersion(modelId, provider.type, providersMeta),
      }
      return
    }

    const selected = []
    const versions = {}
    modelIds.forEach(modelId => {
      const providerType = modelIdToProviderType(modelId)
      const provider = providerType ? takeProvider(providerType) : null
      if (!provider) return
      selected.push(provider.id)
      versions[provider.id] = modelIdToVersion(modelId, provider.type, providersMeta)
    })
    if (!selected.length) return

    // multi 단계는 가중치 정보가 없으므로 균등 분배로 복원 (나머지는 마지막에 몰아줌)
    const count = selected.length
    const base = Math.round(100 / count)
    const weights = selected.reduce((acc, providerId, index) => {
      acc[providerId] = index === count - 1 ? 100 - base * (count - 1) : base
      return acc
    }, {})

    next[stageKey] = {
      ...stage,
      selected,
      versions,
      weights,
      confirmed: true,
    }
  })

  return next
}

// 편집 화면 상태 → 저장용 editor_state (provider id 대신 providerType 으로 기록해 재현 가능)
export function serializeEditorState(stages, retryCounts, providers, stageOrder) {
  const serializedStages = {}

  stageOrder.forEach(stageKey => {
    const stage = stages[stageKey]
    if (!stage) return

    if (stage.mode === 'single') {
      if (!stage.selected || !stage.version) return
      const provider = providers.find(item => item.id === stage.selected)
      if (!provider) return
      serializedStages[stageKey] = {
        label: stage.label,
        mode: stage.mode,
        models: [{
          providerType: provider.type,
          version: stage.version,
          modelId: versionToModelId(stage.version),
          credentialRef: provider.credentialRef || '',
        }],
      }
      return
    }

    if (!stage.selected?.length) return
    serializedStages[stageKey] = {
      label: stage.label,
      mode: stage.mode,
      confirmed: Boolean(stage.confirmed),
      models: stage.selected
        .map(providerId => {
          const provider = providers.find(item => item.id === providerId)
          if (!provider) return null
          const version = stage.versions?.[providerId]
          return {
            providerType: provider.type,
            version,
            modelId: versionToModelId(version),
            credentialRef: provider.credentialRef || '',
            weight: Number(stage.weights?.[providerId] || 0),
          }
        })
        .filter(Boolean),
    }
  })

  return {
    stages: serializedStages,
    retryCounts: { ...retryCounts },
  }
}

// 저장된 editor_state 에 있으나 현재 등록 목록에 없는 provider 를 placeholder 로 채움
export function ensureProvidersForEditorState(existingProviders, editorState) {
  const nextProviders = [...existingProviders]
  const byType = new Map(nextProviders.map(provider => [provider.type, provider]))
  const stageEntries = Object.values(editorState?.stages || {})

  stageEntries.forEach(stage => {
    ;(stage.models || []).forEach(model => {
      if (!model.providerType || byType.has(model.providerType)) return
      const provider = {
        id: nextProviderId(nextProviders),
        type: model.providerType,
        keyMasked: '저장된 프리셋',
        credentialRef: model.credentialRef || '',
        isPresetPlaceholder: true,
      }
      nextProviders.push(provider)
      byType.set(provider.type, provider)
    })
  })

  return nextProviders
}

// 저장된 editor_state → React 단계 상태 (providerType 으로 현재 provider 를 재매칭)
export function applyEditorStateToStages(baseStages, editorState, providers) {
  if (!editorState?.stages || !providers.length) return baseStages

  const next = { ...baseStages }
  Object.entries(editorState.stages).forEach(([stageKey, savedStage]) => {
    const stage = next[stageKey]
    if (!stage) return
    const savedModels = Array.isArray(savedStage?.models) ? savedStage.models : []
    if (!savedModels.length) return

    if (stage.mode === 'single') {
      const savedModel = savedModels[0]
      const provider = providers.find(item => item.type === savedModel.providerType)
      if (!provider) return
      next[stageKey] = {
        ...stage,
        selected: provider.id,
        version: savedModel.version || MODEL_ID_TO_VERSION[savedModel.modelId] || stage.version,
      }
      return
    }

    const selected = []
    const versions = {}
    const weights = {}
    savedModels.forEach(model => {
      const provider = providers.find(item => item.type === model.providerType)
      if (!provider) return
      selected.push(provider.id)
      versions[provider.id] = model.version || MODEL_ID_TO_VERSION[model.modelId] || ''
      weights[provider.id] = Number(model.weight || 0)
    })
    if (!selected.length) return

    next[stageKey] = {
      ...stage,
      selected,
      versions,
      weights,
      confirmed: savedStage.confirmed ?? true,
    }
  })

  return next
}

// 검증 단계 기본 순서
export const DEFAULT_STAGE_ORDER = ['claim', 'detect', 'classify', 'judge', 'ground', 'slide']

// editor_state → 목록 표시용 요약 (단계별 라벨·모드·재시도·모델)
export function summarizeEditorState(editorState, stageOrder = DEFAULT_STAGE_ORDER) {
  const stages = editorState?.stages || {}
  const keys = [
    ...stageOrder.filter(key => stages[key]),
    ...Object.keys(stages).filter(key => !stageOrder.includes(key)),
  ]
  return keys.map(stageKey => {
    const stage = stages[stageKey] || {}
    return {
      stageKey,
      label: stage.label || STAGE_LABELS[stageKey] || stageKey,
      mode: stage.mode || 'single',
      retryCount: Number(editorState?.retryCounts?.[stageKey] || 0),
      models: Array.isArray(stage.models) ? stage.models : [],
    }
  })
}

// 저장 가능 여부 검증 — 문제가 있으면 안내 문구, 없으면 null
export function validateStagesForSave(stages, stageOrder) {
  for (const stageKey of stageOrder) {
    const stage = stages[stageKey]
    if (!stage) continue
    if (stage.mode === 'single') {
      if (!stage.selected || !stage.version) {
        return `${stage.label} 단계를 설정해주세요.`
      }
      continue
    }
    if (!stage.selected?.length) {
      return `${stage.label} 단계에서 모델을 하나 이상 선택해주세요.`
    }
    if (!stage.confirmed) {
      return `${stage.label} 단계 가중치를 확정해주세요.`
    }
  }
  return null
}
