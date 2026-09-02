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

// 표시명 → 모델 ID (매핑에 없으면 소문자·하이픈으로 슬러그화)
function versionToModelId(version) {
  if (!version) return ''
  return VERSION_TO_MODEL_ID[version] || String(version).trim().toLowerCase().replace(/\s+/g, '-')
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

