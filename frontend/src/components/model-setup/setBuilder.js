// "Multi-LLM 조합(셋)" 만들기 — 선택한 LLM 목록 → 저장 payload / 저장된 셋 → 편집 상태 복원

import {
  DEFAULT_STAGE_ORDER,
  STAGE_LABELS,
  stagesToStageModels,
  VERSION_TO_MODEL_ID,
} from './stageModels'
import { buildLlmConfig, versionToModelId } from './llmRegistry'

// 여러 모델이 교차 검증하는 단계 / 대표 모델 하나만 쓰는 단계
const MULTI_STAGES = ['detect', 'classify', 'judge']
const SINGLE_STAGES = ['claim', 'slide']

// id 배열에 100 을 균등 분배 (나머지는 마지막 id 에 몰아줌)
function equalWeights(ids) {
  const count = ids.length
  if (!count) return {}
  const base = Math.round(100 / count)
  return ids.reduce((weights, id, index) => {
    weights[id] = index === count - 1 ? 100 - base * (count - 1) : base
    return weights
  }, {})
}

// 단계별 재시도 횟수 기본값 (그라운딩 포함 시 ground 도 추가)
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

// 선택 LLM + 대표(main) LLM + 그라운딩 여부 → 저장 payload (stage_models / llm_config / editor_state)
// 단계 배치: claim·slide 는 대표 모델 1개, detect·classify·judge 는 선택 모델 전부(균등 가중치),
// ground 는 그라운딩 포함 시에만 대표 모델 1개
// 가중치는 균등 분배로 내부 저장만 하고 UI 에는 노출하지 않음
export function buildSetPayload({ name, selectedLlms, mainLlmId, includeGrounding }) {
  const selected = (selectedLlms || []).filter(Boolean)
  if (!selected.length) {
    throw new Error('사용할 LLM을 하나 이상 선택해주세요.')
  }

  const main = selected.find(llm => llm.id === mainLlmId) || selected[0]
  // endpoint 레코드를 그대로 유지 — credentialRef 가 llm_config 로 전달돼야 함
  // ({id, type} 로 줄이면 웹에서 등록한 키가 프리셋 실행에서 누락됨)
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

// 저장된 editor_state → 셋 편집 화면 상태 { selectedLlmIds, selectedLlms, mainLlmId, includeGrounding }
// 현재 등록 목록에 있으면 그 레코드, 없으면 저장 당시 스냅샷 사용
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

  // 구버전 프리셋(llm_set_v2 아님): multi 단계에서 쓰인 모델들을 셋 선택으로 복원
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

// editor_state → 셋 카드 표시용 요약 (모델 수·이름·대표 모델·그라운딩·단계 수)
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
