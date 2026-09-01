import {
  DEFAULT_STAGE_ORDER,
  STAGE_LABELS,
  stagesToStageModels,
  VERSION_TO_MODEL_ID,
} from './stageModels'
import { buildLlmConfig, versionToModelId } from './llmRegistry'

const MULTI_STAGES = ['detect', 'classify', 'judge']
const SINGLE_STAGES = ['claim', 'slide']

function equalWeights(ids) {
  const count = ids.length
  if (!count) return {}
  const base = Math.round(100 / count)
  return ids.reduce((weights, id, index) => {
    weights[id] = index === count - 1 ? 100 - base * (count - 1) : base
    return weights
  }, {})
}

function defaultRetryCounts(includeGrounding) {
  const counts = {
    claim: 1,
    detect: 1,
    classify: 1,
    judge: 1,
    slide: 1,
  }
  if (includeGrounding) counts.ground = 1
  return counts
}

/**
 * 등록 LLM + 메인 LLM + 그라운딩 여부 → 기존 stage_models / editor_state 형태로 변환.
 * 가중치는 균등 분할(기존 자동 배정과 동일)로 내부 저장만 하고 UI에서는 노출하지 않는다.
 */
export function buildSetPayload({ name, selectedLlms, mainLlmId, includeGrounding }) {
  const selected = (selectedLlms || []).filter(Boolean)
  if (!selected.length) {
    throw new Error('사용할 LLM을 하나 이상 선택해주세요.')
  }

  const main = selected.find(llm => llm.id === mainLlmId) || selected[0]
  // Keep the complete endpoint records here. In particular, a model set must
  // carry credentialRef into llm_config; reducing this to {id, type} silently
  // dropped the web-registered credential for preset-based runs.
  const providers = selected
  const selectedIds = selected.map(llm => llm.id)
  const weights = equalWeights(selectedIds)

  const stages = {
    claim: {
      mode: 'single',
      label: STAGE_LABELS.claim,
      selected: main.id,
      version: main.version,
    },
    detect: {
      mode: 'multi',
      label: STAGE_LABELS.detect,
      selected: selectedIds,
      versions: Object.fromEntries(selected.map(llm => [llm.id, llm.version])),
      weights,
      confirmed: true,
    },
    classify: {
      mode: 'multi',
      label: STAGE_LABELS.classify,
      selected: selectedIds,
      versions: Object.fromEntries(selected.map(llm => [llm.id, llm.version])),
      weights: { ...weights },
      confirmed: true,
    },
    judge: {
      mode: 'multi',
      label: STAGE_LABELS.judge,
      selected: selectedIds,
      versions: Object.fromEntries(selected.map(llm => [llm.id, llm.version])),
      weights: { ...weights },
      confirmed: true,
    },
    slide: {
      mode: 'single',
      label: STAGE_LABELS.slide,
      selected: main.id,
      version: main.version,
    },
  }

  const stageOrder = [...DEFAULT_STAGE_ORDER]
  if (includeGrounding) {
    stages.ground = {
      mode: 'single',
      label: STAGE_LABELS.ground,
      selected: main.id,
      version: main.version,
    }
  } else {
    const groundIndex = stageOrder.indexOf('ground')
    if (groundIndex >= 0) stageOrder.splice(groundIndex, 1)
  }

  const retryCounts = defaultRetryCounts(includeGrounding)
  const stageModels = stagesToStageModels(stages, providers, stageOrder)
  stageModels.CLASSIFIED_ISSUE_EVIDENCE_ENABLED = includeGrounding ? '1' : '0'

  const serializedStages = {}
  stageOrder.forEach(stageKey => {
    const stage = stages[stageKey]
    if (!stage) return
    if (stage.mode === 'single') {
      serializedStages[stageKey] = {
        label: stage.label,
        mode: 'single',
        models: [{
          providerType: main.type,
          version: main.version,
          modelId: main.modelId || versionToModelId(main.version),
          credentialRef: main.credentialRef || '',
        }],
      }
      return
    }
    serializedStages[stageKey] = {
      label: stage.label,
      mode: 'multi',
      confirmed: true,
      models: selected.map(llm => ({
        providerType: llm.type,
        version: llm.version,
        modelId: llm.modelId || versionToModelId(llm.version),
        credentialRef: llm.credentialRef || '',
        weight: Number(weights[llm.id] || 0),
      })),
    }
  })

  return {
    name: (name || '').trim(),
    stage_models: stageModels,
    llm_config: buildLlmConfig(selected, stages, stageOrder),
    editor_state: {
      kind: 'llm_set_v2',
      selectedLlmIds: selectedIds,
      mainLlmId: main.id,
      includeGrounding: Boolean(includeGrounding),
      models: selected.map(llm => ({
        id: llm.id,
        type: llm.type,
        version: llm.version,
        modelId: llm.modelId || versionToModelId(llm.version),
        credentialRef: llm.credentialRef || '',
        keyMasked: llm.keyMasked || '',
        providerName: llm.providerName || llm.type,
      })),
      stages: serializedStages,
      retryCounts,
    },
  }
}

export function parseSetEditorState(editorState, registeredLlms = []) {
  const state = editorState && typeof editorState === 'object' ? editorState : {}
  const registryById = new Map(registeredLlms.map(llm => [llm.id, llm]))

  if (state.kind === 'llm_set_v2') {
    const snapshot = Array.isArray(state.models) ? state.models : []
    const selectedIds = Array.isArray(state.selectedLlmIds)
      ? state.selectedLlmIds
      : snapshot.map(model => model.id)

    const selectedLlms = selectedIds
      .map(id => {
        if (registryById.has(id)) return registryById.get(id)
        const snap = snapshot.find(model => model.id === id)
        return snap || null
      })
      .filter(Boolean)

    return {
      selectedLlmIds: selectedLlms.map(llm => llm.id),
      selectedLlms,
      mainLlmId: state.mainLlmId || selectedLlms[0]?.id || '',
      includeGrounding: state.includeGrounding !== false,
    }
  }

  // 구버전 프리셋: multi 단계에서 쓰인 모델들을 셋 선택으로 복원
  const stages = state.stages || {}
  const multiModels = []
  const seen = new Set()
  MULTI_STAGES.forEach(stageKey => {
    const models = stages[stageKey]?.models || []
    models.forEach(model => {
      const modelId = model.modelId || VERSION_TO_MODEL_ID[model.version] || model.version
      const key = `${model.providerType}:${modelId}`
      if (seen.has(key)) return
      seen.add(key)
      multiModels.push({
        id: `legacy-${key}`,
        type: model.providerType,
        version: model.version,
        modelId,
        keyMasked: '',
        credentialRef: model.credentialRef || '',
        providerName: model.providerName || model.providerType,
      })
    })
  })

  const mainFromClaim = stages.claim?.models?.[0]
  let mainLlmId = multiModels[0]?.id || ''
  if (mainFromClaim) {
    const match = multiModels.find(
      llm => llm.type === mainFromClaim.providerType
        && (llm.version === mainFromClaim.version || llm.modelId === mainFromClaim.modelId),
    )
    if (match) mainLlmId = match.id
  }

  return {
    selectedLlmIds: multiModels.map(llm => llm.id),
    selectedLlms: multiModels,
    mainLlmId,
    includeGrounding: Boolean(stages.ground?.models?.length),
  }
}

export function summarizeSetConfig(editorState) {
  const state = editorState && typeof editorState === 'object' ? editorState : {}
  if (state.kind === 'llm_set_v2') {
    const models = Array.isArray(state.models) ? state.models : []
    const main = models.find(model => model.id === state.mainLlmId) || models[0]
    return {
      modelCount: models.length,
      modelNames: models.map(model => model.version).filter(Boolean),
      models: models.map(model => ({
        id: model.id,
        type: model.type,
        version: model.version,
        modelId: model.modelId,
        isMain: Boolean(main) && model.id === main.id,
      })),
      mainModelName: main?.version || '미지정',
      includeGrounding: state.includeGrounding !== false,
      stageCount: Object.keys(state.stages || {}).length || (state.includeGrounding === false ? 5 : 6),
    }
  }

  const stages = state.stages || {}
  const names = []
  const nameToModelId = new Map()
  const nameToType = new Map()
  const seen = new Set()
  Object.values(stages).forEach(stage => {
    ;(stage.models || []).forEach(model => {
      const name = model.version
      if (!name || seen.has(name)) return
      seen.add(name)
      names.push(name)
      nameToModelId.set(name, model.modelId)
      nameToType.set(name, model.providerType)
    })
  })
  const mainModelName = stages.claim?.models?.[0]?.version || names[0] || '미지정'
  return {
    modelCount: names.length,
    modelNames: names,
    models: names.map(name => ({
      id: name,
      type: nameToType.get(name),
      version: name,
      modelId: nameToModelId.get(name),
      isMain: name === mainModelName,
    })),
    mainModelName,
    includeGrounding: Boolean(stages.ground?.models?.length),
    stageCount: Object.keys(stages).length,
  }
}

export { MULTI_STAGES, SINGLE_STAGES }
