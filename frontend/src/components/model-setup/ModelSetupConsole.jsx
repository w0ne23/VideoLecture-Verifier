import { useEffect, useMemo, useRef, useState } from 'react'
import {
  applyEditorStateToStages,
  ensureProvidersForEditorState,
  nextProviderId,
  serializeEditorState,
  stagesToStageModels,
  validateStagesForSave,
} from './stageModels'
import {
  defaultEndpointConfig,
  buildLlmConfig,
  detectProvider,
  loadRegisteredLlms,
  maskKey,
  PROVIDERS_META,
  providerRequiresKey,
  saveRegisteredLlms,
} from './llmRegistry'

const STAGE_ORDER = ['claim', 'detect', 'classify', 'judge', 'ground', 'slide']

const createInitialStages = () => ({
  claim: {
    mode: 'single',
    label: 'Claim 추출',
    desc: '강의 발화에서 검증 가능한 사실 명제(Claim)를 추출해요.',
    selected: null,
    version: null,
  },
  detect: {
    mode: 'multi',
    label: 'Issue 탐지',
    desc: '추출된 Claim에서 잠재적 오류(Issue) 여부를 탐지해요.',
    selected: [],
    versions: {},
    weights: {},
    confirmed: false,
  },
  classify: {
    mode: 'multi',
    label: '유형 분류',
    desc: '탐지된 Issue를 사실 오류, 오래된 내용 등 유형으로 분류해요.',
    selected: [],
    versions: {},
    weights: {},
    confirmed: false,
  },
  judge: {
    mode: 'multi',
    label: 'Issue 판단',
    desc: '여러 모델의 판단이 갈리면 해당 Issue의 신뢰도를 낮춰요.',
    selected: [],
    versions: {},
    weights: {},
    confirmed: false,
  },
  ground: {
    mode: 'single',
    label: '웹 그라운딩',
    desc: '웹 검색으로 실제 사실과 대조해 오탐을 걸러내요. (모델 자체 웹서치 기능 사용)',
    selected: null,
    version: null,
  },
  slide: {
    mode: 'single',
    label: '슬라이드 오류',
    desc: '슬라이드 이미지/텍스트에서 오류를 검사해요.',
    selected: null,
    version: null,
  },
})

const SIDE_PANEL_TEXT = {
  key: '오류 검증에 쓰일 LLM 모델을 선택 후, API Key를 입력하세요. 여기서 등록한 모델로 이후 단계에 적용할 수 있어요.',
  auto: '등록한 모델을 검증 파이프라인의 각 단계와 슬라이드 오류 검사에 배정해요. 자동 배정 결과를 확인하거나, 상세 조절에서 단계별로 모델, 가중치, 재시도 횟수를 직접 조정할 수 있어요.',
  detail: '등록한 모델을 검증 파이프라인의 각 단계와 슬라이드 오류 검사에 배정해요. 자동 배정 결과를 확인하거나, 상세 조절에서 단계별로 모델, 가중치, 재시도 횟수를 직접 조정할 수 있어요.',
}

const createInitialRetryCounts = () => ({
  claim: 1,
  detect: 1,
  classify: 1,
  judge: 1,
  ground: 1,
  slide: 1,
})

function recomputeWeights(selected) {
  const count = selected.length
  if (!count) return {}
  const base = Math.round(100 / count)
  return selected.reduce((weights, providerId, index) => {
    weights[providerId] = index === count - 1 ? 100 - base * (count - 1) : base
    return weights
  }, {})
}

function providerName(providers, providerId) {
  const provider = providers.find(item => item.id === providerId)
  return provider ? (PROVIDERS_META[provider.type]?.name || provider.type) : ''
}

function getAutoStages(providers) {
  const nextStages = createInitialStages()
  const firstProvider = providers[0]

  nextStages.claim.selected = firstProvider.id
  nextStages.claim.version = firstProvider.version || PROVIDERS_META[firstProvider.type]?.versions?.[0] || ''
  nextStages.ground.selected = firstProvider.id
  nextStages.ground.version = firstProvider.version || PROVIDERS_META[firstProvider.type]?.versions?.[0] || ''
  nextStages.slide.selected = firstProvider.id
  nextStages.slide.version = firstProvider.version || PROVIDERS_META[firstProvider.type]?.versions?.[0] || ''

  ;['detect', 'classify', 'judge'].forEach(stageKey => {
    const selected = providers.map(provider => provider.id)
    nextStages[stageKey].selected = selected
    nextStages[stageKey].versions = providers.reduce((versions, provider) => {
      versions[provider.id] = provider.version || PROVIDERS_META[provider.type]?.versions?.[0] || ''
      return versions
    }, {})
    nextStages[stageKey].weights = recomputeWeights(selected)
    nextStages[stageKey].confirmed = true
  })

  return nextStages
}

export default function ModelSetupConsole({
  headerTitle = '새 프리셋 만들기',
  initialName = '',
  initialEditorState = null,
  onBack,
  onRequestSubmit,
  confirmMessage = '저장하시겠습니까?',
  confirmActionLabel = '확인',
  cancelActionLabel = '취소',
  saveLabel = '저장하기',
  showPostSaveDialog = false,
  postSaveMessage = '저장 완료했습니다. 프리셋을 확인해 보시겠습니까?',
  onGoHome,
  onGoList,
  isSubmitting = false,
  errorMessage = '',
}) {
  const [view, setView] = useState('key')
  const [presetName, setPresetName] = useState(initialName)
  const [providers, setProviders] = useState(() => loadRegisteredLlms())
  const [providerSelect, setProviderSelect] = useState('auto')
  const [keyValue, setKeyValue] = useState('')
  const [rememberKey, setRememberKey] = useState(true)
  const [showHelp, setShowHelp] = useState(false)
  const [stages, setStages] = useState(() => createInitialStages())
  const [activeStage, setActiveStage] = useState('claim')
  const [openDropdown, setOpenDropdown] = useState(null)
  const [retryCounts, setRetryCounts] = useState(() => createInitialRetryCounts())
  const hydratedRef = useRef(false)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [doneOpen, setDoneOpen] = useState(false)

  useEffect(() => {
    if (hydratedRef.current) return
    if (!initialEditorState) {
      hydratedRef.current = true
      return
    }

    const nextProviders = ensureProvidersForEditorState(providers, initialEditorState)
    setProviders(nextProviders)
    setStages(current => applyEditorStateToStages(current, initialEditorState, nextProviders))
    setRetryCounts(current => ({
      ...current,
      ...(initialEditorState.retryCounts || {}),
    }))
    hydratedRef.current = true
  }, [initialEditorState, providers])

  useEffect(() => {
    setPresetName(initialName)
  }, [initialName])

  const validateBeforeSave = () => {
    if (!(presetName || '').trim()) {
      window.alert('프리셋 이름을 입력해주세요.')
      return null
    }
    if (!providers.length) {
      window.alert('먼저 모델을 하나 이상 등록해주세요.')
      return null
    }
    const validationError = validateStagesForSave(stages, STAGE_ORDER)
    if (validationError) {
      window.alert(validationError)
      return null
    }
    const stageModels = stagesToStageModels(stages, providers, STAGE_ORDER)
    if (!Object.keys(stageModels).length) {
      window.alert('저장할 모델 설정이 없어요.')
      return null
    }
    return {
      name: presetName.trim(),
      stage_models: stageModels,
      llm_config: buildLlmConfig(providers, stages, STAGE_ORDER),
      editor_state: serializeEditorState(stages, retryCounts, providers, STAGE_ORDER),
    }
  }

  const handleSave = () => {
    if (isSubmitting) return
    if (!validateBeforeSave()) return
    setConfirmOpen(true)
  }

  const handleConfirmSave = async () => {
    const payload = validateBeforeSave()
    if (!payload) return
    setConfirmOpen(false)
    try {
      await onRequestSubmit?.(payload)
      if (showPostSaveDialog) setDoneOpen(true)
    } catch {
      // 에러는 상위 mutation errorMessage로 표시
    }
  }

  const stageUsage = useMemo(() => {
    const usage = {}
    providers.forEach(provider => {
      usage[provider.id] = STAGE_ORDER.filter(stageKey => {
        const stage = stages[stageKey]
        return stage.mode === 'single'
          ? stage.selected === provider.id
          : stage.selected.includes(provider.id)
      }).map(stageKey => stages[stageKey].label)
    })
    return usage
  }, [providers, stages])

  useEffect(() => {
    if (rememberKey) {
      saveRegisteredLlms(providers.filter(provider => !provider.isPresetPlaceholder))
      return
    }
    localStorage.removeItem('verilec_registered_llms')
    localStorage.removeItem('verilec_providers')
  }, [providers, rememberKey])

  const addProvider = () => {
    const key = keyValue.trim()
    if (!key && providerSelect === 'auto') {
      window.alert('API 키를 입력해주세요.')
      return
    }

    const detected = detectProvider(key)
    let finalType = providerSelect

    if (providerSelect === 'auto') {
      if (!detected) {
        window.alert('키 형식을 인식할 수 없어요. 위 드롭다운에서 provider를 직접 선택해주세요.')
        return
      }
      finalType = detected
    } else if (providerRequiresKey(finalType) && !key) {
      window.alert('API 키를 입력해주세요.')
      return
    } else if (detected && detected !== providerSelect) {
      const shouldUseDetected = window.confirm(
        `입력하신 키는 ${PROVIDERS_META[detected].name} 형식으로 보여요.\n선택하신 ${PROVIDERS_META[providerSelect].name} 대신 ${PROVIDERS_META[detected].name}로 등록할까요?`,
      )
      if (!shouldUseDetected) return
      finalType = detected
    }

    setProviders(current => {
      const placeholder = current.find(
        provider => provider.type === finalType && provider.isPresetPlaceholder,
      )
      if (placeholder) {
        return current.map(provider => (
          provider.id === placeholder.id
            ? { ...provider, keyMasked: maskKey(key), isPresetPlaceholder: false }
            : provider
        ))
      }
      return [
        ...current,
        {
          id: nextProviderId(current),
          type: finalType,
          version: PROVIDERS_META[finalType]?.versions?.[0] || '',
          modelId: PROVIDERS_META[finalType]?.versions?.[0] || '',
          keyMasked: providerRequiresKey(finalType) ? maskKey(key) : '키 불필요',
          providerName: PROVIDERS_META[finalType]?.name || finalType,
          ...defaultEndpointConfig(finalType),
          isPresetPlaceholder: false,
        },
      ]
    })
    setKeyValue('')
    setProviderSelect('auto')
  }

  const removeProvider = providerId => {
    const usedStages = stageUsage[providerId] || []
    if (usedStages.length) {
      const shouldRemove = window.confirm(
        `이 모델은 ${usedStages.join(', ')} 단계에서 사용 중이에요.\n삭제하면 해당 단계 설정이 초기화돼요. 계속할까요?`,
      )
      if (!shouldRemove) return
    }

    setProviders(current => current.filter(provider => provider.id !== providerId))
    setStages(current => {
      const next = { ...current }
      STAGE_ORDER.forEach(stageKey => {
        const stage = next[stageKey]
        if (stage.mode === 'single' && stage.selected === providerId) {
          next[stageKey] = { ...stage, selected: null, version: null }
        }
        if (stage.mode === 'multi' && stage.selected.includes(providerId)) {
          const selected = stage.selected.filter(id => id !== providerId)
          const { [providerId]: _removedVersion, ...versions } = stage.versions
          next[stageKey] = {
            ...stage,
            selected,
            versions,
            weights: recomputeWeights(selected),
            confirmed: false,
          }
        }
      })
      return next
    })
  }

  const goKey = () => {
    setView('key')
    setOpenDropdown(null)
  }

  const goAuto = () => {
    if (!providers.length) {
      window.alert('먼저 모델을 하나 이상 등록해주세요.')
      return
    }
    setStages(getAutoStages(providers))
    setView('auto')
    setOpenDropdown(null)
  }

  const goDetail = () => {
    if (!providers.length) {
      window.alert('먼저 모델을 하나 이상 등록해주세요.')
      return
    }
    setStages(current => (current.claim.selected ? current : getAutoStages(providers)))
    setActiveStage('claim')
    setOpenDropdown(null)
    setView('detail')
  }

  const toggleChip = providerId => {
    const provider = providers.find(item => item.id === providerId)
    if (!provider) return

    setStages(current => {
      const stage = current[activeStage]
      const meta = PROVIDERS_META[provider.type] || PROVIDERS_META.custom
      if (stage.mode === 'single') {
        return {
          ...current,
          [activeStage]: { ...stage, selected: providerId, version: meta.versions[0] },
        }
      }

      const alreadySelected = stage.selected.includes(providerId)
      const selected = alreadySelected
        ? stage.selected.filter(id => id !== providerId)
        : [...stage.selected, providerId]
      const versions = { ...stage.versions }
      if (alreadySelected) {
        delete versions[providerId]
      } else {
        versions[providerId] = meta.versions[0]
      }

      return {
        ...current,
        [activeStage]: {
          ...stage,
          selected,
          versions,
          weights: recomputeWeights(selected),
          confirmed: false,
        },
      }
    })
    setOpenDropdown(null)
  }

  const pickVersion = (providerId, version) => {
    setStages(current => {
      const stage = current[activeStage]
      if (stage.mode === 'single') {
        if (stage.selected !== providerId) return current
        return { ...current, [activeStage]: { ...stage, version } }
      }
      return {
        ...current,
        [activeStage]: {
          ...stage,
          versions: { ...stage.versions, [providerId]: version },
        },
      }
    })
    setOpenDropdown(null)
  }

  const setWeight = (providerId, rawValue) => {
    const value = Math.max(0, Math.min(100, Number.parseInt(rawValue, 10) || 0))
    setStages(current => {
      const stage = current[activeStage]
      if (stage.mode !== 'multi') return current
      return {
        ...current,
        [activeStage]: {
          ...stage,
          weights: { ...stage.weights, [providerId]: value },
        },
      }
    })
  }

  const toggleConfirm = () => {
    setStages(current => {
      const stage = current[activeStage]
      if (stage.mode !== 'multi') return current
      const total = stage.selected.reduce((sum, providerId) => sum + (stage.weights[providerId] || 0), 0)
      if (!stage.confirmed && total !== 100) return current
      return {
        ...current,
        [activeStage]: { ...stage, confirmed: !stage.confirmed },
      }
    })
  }

  const setRetryCount = (stageKey, rawValue) => {
    const value = Math.max(0, Math.min(5, Number.parseInt(rawValue, 10) || 0))
    setRetryCounts(current => ({ ...current, [stageKey]: value }))
  }

  const renderPipeline = editable => (
    <div className="ms-pipeline-scroll">
      <div className="ms-pipeline">
        {STAGE_ORDER.map((stageKey, index) => {
          const stage = stages[stageKey]
          const confirmed = stage.mode === 'multi' ? stage.confirmed : Boolean(stage.selected)
          const sub = stage.mode === 'multi'
            ? stage.confirmed
              ? `${stage.selected.length}개 모델`
              : stage.selected.length
                ? `${stage.selected.length}개 선택 중`
                : '미설정'
            : stage.selected
              ? providerName(providers, stage.selected)
              : '미설정'

          const className = [
            'ms-stage-box',
            confirmed ? 'ms-stage-box--configured' : '',
            editable && stageKey === activeStage ? 'ms-stage-box--active' : '',
          ].filter(Boolean).join(' ')

          const stageContent = (
            <>
              <span className="ms-stage-num">{index + 1}</span>
              <span className="ms-stage-label">{stage.label}</span>
              <span className="ms-stage-sub">{sub}</span>
            </>
          )

          return (
            <div className="ms-pipeline-part" key={stageKey}>
              {stageKey === 'slide' && <span className="ms-stage-divider" aria-hidden="true" />}
              {editable ? (
                <button
                  className={className}
                  type="button"
                  onClick={() => {
                    setActiveStage(stageKey)
                    setOpenDropdown(null)
                  }}
                >
                  {stageContent}
                </button>
              ) : (
                <div className={className}>{stageContent}</div>
              )}
              {index < STAGE_ORDER.length - 1 && stageKey !== 'ground' && (
                <span className="ms-arrow">-&gt;</span>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )

  const activeStageConfig = stages[activeStage]
  const activeWeightTotal = activeStageConfig.mode === 'multi'
    ? activeStageConfig.selected.reduce((sum, providerId) => sum + (activeStageConfig.weights[providerId] || 0), 0)
    : 0

  return (
    <section className="model-setup">
      <div className="ms-header-row">
        <h2 className="ms-app-title">{headerTitle}</h2>
        <button className="ms-link-btn ms-link-btn--compact" type="button" onClick={onBack}>
          뒤로 가기 -&gt;
        </button>
      </div>

      <div className="ms-layout">
        <aside className="ms-side-panel">
          <p>{SIDE_PANEL_TEXT[view]}</p>
        </aside>

        <div className="ms-main-panel">
          <div className="ms-card">
            <p className="ms-label">프리셋 이름</p>
            <input
              className="ms-name-input"
              type="text"
              value={presetName}
              placeholder="예: 빠른 검증, 정확도 우선"
              onChange={event => setPresetName(event.target.value)}
            />
          </div>

          {view === 'key' && (
            <>
              <div className="ms-card">
                <p className="ms-label">등록된 모델</p>
                {providers.length ? (
                  <div className="ms-provider-list">
                    {providers.map(provider => (
                      <div className="ms-provider-row" key={provider.id}>
                        <span className="ms-provider-name">{PROVIDERS_META[provider.type]?.name || provider.type}</span>
                        <span className="ms-provider-key">{provider.keyMasked}</span>
                        <span className="ms-provider-check">
                          {provider.isPresetPlaceholder ? '복원' : 'OK'}
                        </span>
                        <button
                          className="ms-icon-btn"
                          type="button"
                          aria-label="모델 삭제"
                          onClick={() => removeProvider(provider.id)}
                        >
                          x
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="ms-empty">아직 등록된 모델이 없어요.</p>
                )}

                <div className="ms-add-row">
                  <select value={providerSelect} onChange={event => setProviderSelect(event.target.value)}>
                    <option value="auto">자동 인식</option>
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                    <option value="xai">xAI</option>
                  </select>
                  <input
                    type="text"
                    value={keyValue}
                    placeholder="API 키를 붙여넣으세요"
                    onChange={event => setKeyValue(event.target.value)}
                  />
                  <button className="ms-btn-secondary ms-add-btn" type="button" onClick={addProvider}>
                    추가
                  </button>
                </div>

                <button className="ms-link-btn" type="button" onClick={() => setShowHelp(current => !current)}>
                  키 발급 방법 보기
                </button>
                {showHelp && (
                  <div className="ms-help-panel">
                    <div className="ms-help-row">
                      <span>OpenAI</span>
                      <a href="https://platform.openai.com/api-keys" target="_blank" rel="noreferrer">
                        API Keys 페이지 열기 -&gt;
                      </a>
                    </div>
                    <div className="ms-help-row">
                      <span>Anthropic</span>
                      <a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noreferrer">
                        API Keys 페이지 열기 -&gt;
                      </a>
                    </div>
                    <div className="ms-help-row">
                      <span>xAI</span>
                      <a href="https://console.x.ai" target="_blank" rel="noreferrer">
                        콘솔 열기 -&gt;
                      </a>
                    </div>
                  </div>
                )}

                <label className="ms-checkbox-row">
                  <input
                    type="checkbox"
                    checked={rememberKey}
                    onChange={event => setRememberKey(event.target.checked)}
                  />
                  다음에도 이 키 기억하기
                </label>
              </div>

              <div className="ms-actions">
                <button className="ms-btn-secondary" type="button" onClick={goDetail}>
                  상세 조절
                </button>
                <button className="ms-btn-primary" type="button" onClick={goAuto}>
                  자동 배정
                </button>
              </div>
            </>
          )}

          {view === 'auto' && (
            <>
              {renderPipeline(false)}
              <div className="ms-card">
                {STAGE_ORDER.map((stageKey, index) => {
                  const stage = stages[stageKey]
                  const detailText = stage.mode === 'single'
                    ? `${providerName(providers, stage.selected)} · ${stage.version}`
                    : `${stage.selected
                        .map(providerId => `${providerName(providers, providerId)} ${stage.versions[providerId]}`)
                        .join(', ')} · 균등 가중치`
                  return (
                    <div className="ms-summary-row" key={stageKey}>
                      <strong>{index + 1}. {stage.label}</strong>
                      <span>{detailText}</span>
                    </div>
                  )
                })}
              </div>
              <div className="ms-actions">
                <button className="ms-btn-secondary" type="button" onClick={goDetail}>
                  상세 조절로 전환
                </button>
                <button
                  className="ms-btn-primary"
                  type="button"
                  disabled={isSubmitting}
                  onClick={handleSave}
                >
                  {isSubmitting ? '저장 중…' : saveLabel}
                </button>
              </div>
              {errorMessage && (
                <p className="ms-save-error">
                  {errorMessage}
                </p>
              )}
            </>
          )}

          {view === 'detail' && (
            <>
              {renderPipeline(true)}
              <div className="ms-card">
                <p className="ms-desc">{activeStageConfig.desc}</p>
                <div className="ms-chip-row">
                  {providers.map(provider => {
                    const meta = PROVIDERS_META[provider.type] || PROVIDERS_META.custom
                    const isSelected = activeStageConfig.mode === 'single'
                      ? activeStageConfig.selected === provider.id
                      : activeStageConfig.selected.includes(provider.id)
                    const currentVersion = activeStageConfig.mode === 'single'
                      ? activeStageConfig.selected === provider.id
                        ? activeStageConfig.version
                        : meta.versions[0]
                      : activeStageConfig.versions[provider.id] || meta.versions[0]

                    return (
                      <div className="ms-chip-wrap" key={provider.id}>
                        <button
                          className={`ms-chip-main${isSelected ? ' ms-chip-main--selected' : ''}`}
                          type="button"
                          onClick={() => toggleChip(provider.id)}
                        >
                          {meta.name} · {currentVersion}
                        </button>
                        <button
                          className={`ms-chip-caret${isSelected ? ' ms-chip-caret--selected' : ''}`}
                          type="button"
                          aria-label={`${meta.name} 모델 버전 선택`}
                          onClick={() => setOpenDropdown(current => (current === provider.id ? null : provider.id))}
                        >
                          v
                        </button>
                        {openDropdown === provider.id && (
                          <div className="ms-dropdown">
                            {meta.versions.map(version => (
                              <button
                                className={`ms-dd-item${version === currentVersion ? ' ms-dd-item--active' : ''}`}
                                type="button"
                                key={version}
                                onClick={() => pickVersion(provider.id, version)}
                              >
                                {version}
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>

                {activeStageConfig.mode === 'multi' && activeStageConfig.selected.length > 0 && (
                  <div className="ms-weight-block">
                    {activeStageConfig.selected.map(providerId => {
                      const weight = activeStageConfig.weights[providerId] || 0
                      return (
                        <div className="ms-weight-row" key={providerId}>
                          <span className="ms-weight-label">{providerName(providers, providerId)}</span>
                          <input
                            type="range"
                            min="0"
                            max="100"
                            value={weight}
                            disabled={activeStageConfig.confirmed}
                            onChange={event => setWeight(providerId, event.target.value)}
                          />
                          <input
                            type="number"
                            min="0"
                            max="100"
                            value={weight}
                            disabled={activeStageConfig.confirmed}
                            onChange={event => setWeight(providerId, event.target.value)}
                          />
                          <span className="ms-weight-percent">%</span>
                        </div>
                      )
                    })}
                    <div className="ms-weight-footer">
                      <span className={activeWeightTotal === 100 ? 'ms-ok' : 'ms-warn'}>
                        합계 {activeWeightTotal}%{activeWeightTotal === 100 ? '' : ' · 100%을 맞춰주세요'}
                      </span>
                      <button
                        className={`ms-btn-toggle${activeStageConfig.confirmed ? '' : ' ms-btn-toggle--primary'}`}
                        type="button"
                        disabled={!activeStageConfig.confirmed && activeWeightTotal !== 100}
                        onClick={toggleConfirm}
                      >
                        {activeStageConfig.confirmed ? '수정' : '확정'}
                      </button>
                    </div>
                  </div>
                )}

                <div className="ms-retry-row">
                  <div>
                    <p className="ms-label ms-label--tight">재시도 횟수</p>
                    <p className="ms-hint">API 오류 시 이 단계에서 자동 재시도할 횟수예요</p>
                  </div>
                  <input
                    type="number"
                    min="0"
                    max="5"
                    value={retryCounts[activeStage]}
                    onChange={event => setRetryCount(activeStage, event.target.value)}
                  />
                </div>
              </div>
              <div className="ms-actions">
                <button className="ms-btn-secondary" type="button" onClick={goAuto}>
                  자동 배정
                </button>
                <button
                  className="ms-btn-primary"
                  type="button"
                  disabled={isSubmitting}
                  onClick={handleSave}
                >
                  {isSubmitting ? '저장 중…' : saveLabel}
                </button>
              </div>
              {errorMessage && <p className="ms-save-error">{errorMessage}</p>}
            </>
          )}
        </div>
      </div>

      {confirmOpen && (
        <div className="ms-modal-backdrop" role="presentation" onClick={() => setConfirmOpen(false)}>
          <div className="ms-modal" role="dialog" aria-modal="true" onClick={event => event.stopPropagation()}>
            <h3 className="ms-modal-title">{confirmMessage}</h3>
            <div className="ms-modal-actions">
              <button className="ms-btn-primary" type="button" disabled={isSubmitting} onClick={handleConfirmSave}>
                {confirmActionLabel}
              </button>
              <button className="ms-btn-secondary" type="button" onClick={() => setConfirmOpen(false)}>
                {cancelActionLabel}
              </button>
            </div>
          </div>
        </div>
      )}

      {doneOpen && (
        <div className="ms-modal-backdrop" role="presentation">
          <div className="ms-modal" role="dialog" aria-modal="true">
            <h3 className="ms-modal-title">{postSaveMessage}</h3>
            <div className="ms-modal-actions">
              <button className="ms-btn-primary" type="button" onClick={() => onGoList?.()}>
                목록으로
              </button>
              <button className="ms-btn-secondary" type="button" onClick={() => onGoHome?.()}>
                메인으로
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
