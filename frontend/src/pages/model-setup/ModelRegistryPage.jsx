import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  CUSTOM_VERSION,
  PROVIDERS_META,
  detectProvider,
  llmLabel,
  loadRegisteredLlms,
  maskKey,
  nextLlmId,
  providerRequiresKey,
  saveRegisteredLlms,
  versionToModelId,
} from '../../components/model-setup/llmRegistry'

const NO_KEY_LABEL = '키 불필요'

export default function ModelRegistryPage() {
  const navigate = useNavigate()
  const [llms, setLlms] = useState(() => loadRegisteredLlms())
  const [providerSelect, setProviderSelect] = useState('auto')
  const [keyValue, setKeyValue] = useState('')
  const [version, setVersion] = useState('')
  const [customVersion, setCustomVersion] = useState('')
  // const [showHelp, setShowHelp] = useState(false)
  const [editingId, setEditingId] = useState(null)
  const [editProviderSelect, setEditProviderSelect] = useState('auto')
  const [editKeyValue, setEditKeyValue] = useState('')
  const [editVersion, setEditVersion] = useState('')
  const [editCustomVersion, setEditCustomVersion] = useState('')

  const resolvedType = useMemo(() => {
    if (providerSelect !== 'auto') return providerSelect
    return detectProvider(keyValue.trim()) || ''
  }, [providerSelect, keyValue])

  const keyRequired = resolvedType ? providerRequiresKey(resolvedType) : true
  const versionOptions = resolvedType ? PROVIDERS_META[resolvedType].versions : []

  useEffect(() => {
    if (!resolvedType) {
      setVersion(current => (current ? '' : current))
      return
    }
    const options = PROVIDERS_META[resolvedType].versions
    setVersion(current => (current === CUSTOM_VERSION || options.includes(current) ? current : options[0]))
  }, [resolvedType])

  const editingLlm = editingId ? llms.find(llm => llm.id === editingId) : null

  const editResolvedType = useMemo(() => {
    if (editProviderSelect !== 'auto') return editProviderSelect
    return detectProvider(editKeyValue.trim()) || (editingLlm ? editingLlm.type : '')
  }, [editProviderSelect, editKeyValue, editingLlm])

  const editKeyRequired = editResolvedType ? providerRequiresKey(editResolvedType) : true
  const editVersionOptions = editResolvedType ? PROVIDERS_META[editResolvedType].versions : []

  useEffect(() => {
    if (!editingId) return
    const options = editResolvedType ? PROVIDERS_META[editResolvedType].versions : []
    setEditVersion(current => (
      current === CUSTOM_VERSION || options.includes(current) ? current : (options[0] || '')
    ))
  }, [editingId, editResolvedType])

  const persist = next => {
    setLlms(next)
    saveRegisteredLlms(next)
  }

  const handleAdd = () => {
    const key = keyValue.trim()
    let finalType = providerSelect

    if (providerSelect === 'auto') {
      if (!key) {
        window.alert('API 키를 입력해주세요.')
        return
      }
      const detected = detectProvider(key)
      if (!detected) {
        window.alert('키 형식을 인식할 수 없어요. provider를 직접 선택해주세요.')
        return
      }
      finalType = detected
    } else if (keyRequired) {
      if (!key) {
        window.alert('API 키를 입력해주세요.')
        return
      }
      const detected = detectProvider(key)
      if (detected && detected !== providerSelect) {
        const useDetected = window.confirm(
          `입력하신 키는 ${PROVIDERS_META[detected].name} 형식으로 보여요.\n`
          + `선택하신 ${PROVIDERS_META[providerSelect].name} 대신 ${PROVIDERS_META[detected].name}로 등록할까요?`,
        )
        if (!useDetected) return
        finalType = detected
      }
    }

    const meta = PROVIDERS_META[finalType]
    const selectedVersion = version === CUSTOM_VERSION
      ? customVersion.trim()
      : (versionOptions.includes(version) ? version : meta.versions[0])
    if (!selectedVersion) {
      window.alert('모델명을 입력해주세요.')
      return
    }

    const modelId = versionToModelId(selectedVersion)
    const keyMasked = providerRequiresKey(finalType) ? maskKey(key) : NO_KEY_LABEL
    const duplicate = llms.some(
      llm => llm.type === finalType && llm.modelId === modelId
        && (!providerRequiresKey(finalType) || llm.keyMasked === keyMasked),
    )
    if (duplicate) {
      window.alert(providerRequiresKey(finalType) ? '이미 같은 키·모델로 등록되어 있어요.' : '이미 등록된 로컬 모델이에요.')
      return
    }

    const next = [
      ...llms,
      {
        id: nextLlmId(llms),
        type: finalType,
        version: selectedVersion,
        modelId,
        keyMasked,
        providerName: meta.name,
      },
    ]
    persist(next)
    setKeyValue('')
    setVersion('')
    setCustomVersion('')
    setProviderSelect('auto')
  }

  const handleRemove = id => {
    if (!window.confirm('이 LLM 등록을 삭제할까요?')) return
    persist(llms.filter(llm => llm.id !== id))
  }

  const handleEditStart = llm => {
    setEditingId(llm.id)
    setEditProviderSelect(llm.type)
    setEditKeyValue(llm.keyMasked)
    const meta = PROVIDERS_META[llm.type]
    const isPresetVersion = meta?.versions.includes(llm.version)
    setEditVersion(isPresetVersion ? llm.version : CUSTOM_VERSION)
    setEditCustomVersion(isPresetVersion ? '' : llm.version)
  }

  const handleEditCancel = () => {
    setEditingId(null)
    setEditProviderSelect('auto')
    setEditKeyValue('')
    setEditVersion('')
    setEditCustomVersion('')
  }

  const handleEditSave = () => {
    const current = llms.find(llm => llm.id === editingId)
    if (!current) return

    const key = editKeyValue.trim()
    const keyChanged = key !== current.keyMasked
    let finalType = editProviderSelect === 'auto' ? current.type : editProviderSelect
    let keyMasked = current.keyMasked

    if (editKeyRequired && keyChanged && key) {
      const detected = detectProvider(key)
      if (editProviderSelect === 'auto') {
        if (!detected) {
          window.alert('키 형식을 인식할 수 없어요. provider를 직접 선택해주세요.')
          return
        }
        finalType = detected
      } else if (detected && detected !== editProviderSelect) {
        const useDetected = window.confirm(
          `입력하신 키는 ${PROVIDERS_META[detected].name} 형식으로 보여요.\n`
          + `선택하신 ${PROVIDERS_META[editProviderSelect].name} 대신 ${PROVIDERS_META[detected].name}로 변경할까요?`,
        )
        finalType = useDetected ? detected : editProviderSelect
      }
      keyMasked = maskKey(key)
    } else if (!providerRequiresKey(finalType)) {
      keyMasked = NO_KEY_LABEL
    }

    const meta = PROVIDERS_META[finalType]
    const selectedVersion = editVersion === CUSTOM_VERSION
      ? editCustomVersion.trim()
      : (meta.versions.includes(editVersion) ? editVersion : meta.versions[0])
    if (!selectedVersion) {
      window.alert('모델명을 입력해주세요.')
      return
    }
    const modelId = versionToModelId(selectedVersion)

    const next = llms.map(llm => (llm.id === editingId
      ? { ...llm, type: finalType, version: selectedVersion, modelId, keyMasked, providerName: meta.name }
      : llm))
    persist(next)
    handleEditCancel()
  }

  return (
    <section className="model-setup">
      <div className="ms-header-row">
        <h2 className="ms-app-title">LLM 모델 등록</h2>
        <button className="ms-back-btn" type="button" onClick={() => navigate('/model-setup')} aria-label="선택 화면으로">
          ←
        </button>
      </div>

      <div className="ms-stack">
        <div className="ms-stack-section">
          <h3 className="ms-split-title">새 모델 등록</h3>
          <div className="ms-card">
            <p className="ms-hint" style={{ marginBottom: 12 }}>
              API 키(또는 로컬 모델명)를 등록하면 여기서 등록한 모델만 LLM 셋 만들기에서 선택할 수 있어요.
            </p>
            <div className="ms-add-row ms-add-row--wrap">
              <select value={providerSelect} onChange={event => {
                setProviderSelect(event.target.value)
                setVersion('')
                setCustomVersion('')
              }}
              >
                <option value="auto">자동 감지</option>
                {Object.entries(PROVIDERS_META).map(([type, meta]) => (
                  <option value={type} key={type}>{meta.name}</option>
                ))}
              </select>
              <input
                type="password"
                value={keyValue}
                placeholder={keyRequired ? 'API Key 입력' : '로컬 모델은 키가 필요 없어요'}
                disabled={!keyRequired}
                onChange={event => setKeyValue(event.target.value)}
                onKeyDown={event => {
                  if (event.key === 'Enter') handleAdd()
                }}
              />
              <select
                value={version || ''}
                onChange={event => setVersion(event.target.value)}
                disabled={!versionOptions.length}
              >
                <option value="">{versionOptions.length ? '모델 선택' : '키/provider 먼저'}</option>
                {versionOptions.map(option => (
                  <option value={option} key={option}>{option}</option>
                ))}
                {resolvedType && <option value={CUSTOM_VERSION}>직접 입력...</option>}
              </select>
              {version === CUSTOM_VERSION && (
                <input
                  type="text"
                  value={customVersion}
                  placeholder="모델명 직접 입력 (예: gemma3:12b)"
                  onChange={event => setCustomVersion(event.target.value)}
                  onKeyDown={event => {
                    if (event.key === 'Enter') handleAdd()
                  }}
                />
              )}
              <button className="ms-btn-primary ms-add-btn" type="button" onClick={handleAdd}>
                등록
              </button>
            </div>

            {/* API 키 발급 안내: 나중에 필요하면 복원
            <button className="ms-link-btn" type="button" onClick={() => setShowHelp(current => !current)}>
              {showHelp ? '도움말 접기' : 'API 키 발급 안내'}
            </button>
            {showHelp && (
              <div className="ms-help-panel">
                <div className="ms-help-row">
                  <span>OpenAI</span>
                  <a href="https://platform.openai.com/api-keys" target="_blank" rel="noreferrer">키 발급</a>
                </div>
                <div className="ms-help-row">
                  <span>Anthropic</span>
                  <a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noreferrer">키 발급</a>
                </div>
                <div className="ms-help-row">
                  <span>xAI</span>
                  <a href="https://console.x.ai/" target="_blank" rel="noreferrer">키 발급</a>
                </div>
              </div>
            )}
            */}
          </div>
        </div>

        <div className="ms-stack-section">
          <h3 className="ms-split-title">등록된 모델 <span className="ms-split-title-count">{llms.length}</span></h3>
          <div className="ms-card">
            {llms.length ? (
              <div className="ms-provider-list">
                {llms.map((llm, index) => (
                  editingId === llm.id ? (
                    <div className="ms-provider-row ms-provider-row--edit" key={llm.id}>
                      <span className="ms-provider-index">{index + 1}</span>
                      <div className="ms-provider-edit-fields">
                        <select
                          value={editProviderSelect}
                          onChange={event => setEditProviderSelect(event.target.value)}
                        >
                          <option value="auto">자동 감지</option>
                          {Object.entries(PROVIDERS_META).map(([type, meta]) => (
                            <option value={type} key={type}>{meta.name}</option>
                          ))}
                        </select>
                        <input
                          type="text"
                          value={editKeyValue}
                          placeholder={editKeyRequired ? 'API 키' : '로컬 모델은 키가 필요 없어요'}
                          disabled={!editKeyRequired}
                          onChange={event => setEditKeyValue(event.target.value)}
                          onKeyDown={event => {
                            if (event.key === 'Enter') handleEditSave()
                            if (event.key === 'Escape') handleEditCancel()
                          }}
                        />
                        <select
                          value={editVersion || ''}
                          onChange={event => setEditVersion(event.target.value)}
                          disabled={!editVersionOptions.length}
                        >
                          <option value="">{editVersionOptions.length ? '모델 선택' : '키/provider 먼저'}</option>
                          {editVersionOptions.map(option => (
                            <option value={option} key={option}>{option}</option>
                          ))}
                          {editResolvedType && <option value={CUSTOM_VERSION}>직접 입력...</option>}
                        </select>
                        {editVersion === CUSTOM_VERSION && (
                          <input
                            type="text"
                            value={editCustomVersion}
                            placeholder="모델명 직접 입력 (예: gemma3:12b)"
                            onChange={event => setEditCustomVersion(event.target.value)}
                            onKeyDown={event => {
                              if (event.key === 'Enter') handleEditSave()
                              if (event.key === 'Escape') handleEditCancel()
                            }}
                          />
                        )}
                      </div>
                      <div className="ms-provider-actions">
                        <button className="ms-btn-primary ms-add-btn" type="button" onClick={handleEditSave}>
                          저장
                        </button>
                        <button className="ms-link-btn ms-link-btn--compact" type="button" onClick={handleEditCancel}>
                          취소
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="ms-provider-row" key={llm.id}>
                      <span className="ms-provider-index">{index + 1}</span>
                      <div className="ms-provider-info">
                        <span className="ms-provider-name">{PROVIDERS_META[llm.type]?.name || llm.type}</span>
                        <span className="ms-provider-key-masked">{llm.keyMasked}</span>
                        <span className={`ms-llm-version-chip ms-llm-version-chip--${llm.type}`}>{llm.version}</span>
                      </div>
                      <div className="ms-provider-actions">
                        <button className="ms-icon-btn" type="button" onClick={() => handleEditStart(llm)} aria-label={`${llmLabel(llm)} 수정`}>
                          ✎
                        </button>
                        <button className="ms-icon-btn" type="button" onClick={() => handleRemove(llm.id)} aria-label={`${llmLabel(llm)} 삭제`}>
                          ✕
                        </button>
                      </div>
                    </div>
                  )
                ))}
              </div>
            ) : (
              <p className="ms-empty">아직 등록된 LLM이 없어요. 위에서 새 모델을 등록해보세요.</p>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}
