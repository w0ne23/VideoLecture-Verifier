// LLM 모델 등록 화면 — LiteLLM 카탈로그에서 provider/model 선택 + API 키 등록/수정/삭제
// 등록 목록은 localStorage(llmRegistry), API 키 원본은 서버에만 저장하고 마스킹 값만 보관

import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getLlmCatalog } from '../../api/llmCatalog'
import { saveLlmCredential } from '../../api/llmCredentials'
import SearchableSelect from '../../components/model-setup/SearchableSelect'
import {
  CUSTOM_VERSION,
  llmLabel,
  loadRegisteredLlms,
  maskKey,
  nextLlmId,
  providerMeta,
  providerRequiresKey,
  saveRegisteredLlms,
  versionToModelId,
} from '../../components/model-setup/llmRegistry'

const NO_KEY_LABEL = '키 불필요'

export default function ModelRegistryPage() {
  const navigate = useNavigate()
  const [llms, setLlms] = useState(() => loadRegisteredLlms())
  const [providerSelect, setProviderSelect] = useState('')
  const [keyValue, setKeyValue] = useState('')
  const [version, setVersion] = useState('')
  const [customVersion, setCustomVersion] = useState('')
  const [editingId, setEditingId] = useState(null)
  const [editProviderSelect, setEditProviderSelect] = useState('')
  const [editKeyValue, setEditKeyValue] = useState('')
  const [editVersion, setEditVersion] = useState('')
  const [editCustomVersion, setEditCustomVersion] = useState('')
  const [catalog, setCatalog] = useState(null)
  const [catalogStatus, setCatalogStatus] = useState('loading')
  const [catalogError, setCatalogError] = useState('')
  const [credentialSaving, setCredentialSaving] = useState(false)

  // 마운트 시 LiteLLM 카탈로그(선택 가능한 provider/model 목록) 로드
  useEffect(() => {
    let cancelled = false
    getLlmCatalog()
      .then(data => {
        if (cancelled) return
        setCatalog(data)
        setCatalogStatus('ready')
      })
      .catch(error => {
        if (cancelled) return
        setCatalogStatus('error')
        setCatalogError(error.message || 'LiteLLM 모델 목록을 불러오지 못했습니다.')
      })
    return () => { cancelled = true }
  }, [])

  const catalogProviderMap = useMemo(
    () => new Map((catalog?.providers || []).map(provider => [provider.id, provider])),
    [catalog],
  )

  const providerEntries = useMemo(() => {
    if (catalogStatus !== 'ready') return []
    return (catalog?.providers || []).map(provider => [
      provider.id,
      providerMeta(provider.id, provider),
    ])
  }, [catalog, catalogProviderMap, catalogStatus])

  const providerOptions = useMemo(
    () => providerEntries.map(([value, meta]) => ({ value, label: meta.name })),
    [providerEntries],
  )

  const catalogModelsFor = type => (catalog?.models || [])
    .filter(model => model.provider === type)
    .map(model => model.id)

  const modelOptionsFor = type => {
    return catalogStatus === 'ready' ? catalogModelsFor(type) : []
  }

  const metaFor = type => providerMeta(type, {
    providerName: catalogProviderMap.get(type)?.name,
  })

  const resolvedType = useMemo(() => {
    return providerSelect
  }, [providerSelect])

  const keyRequired = resolvedType
    ? providerRequiresKey(resolvedType, catalogProviderMap.get(resolvedType))
    : true
  const versionOptions = resolvedType ? modelOptionsFor(resolvedType) : []

  // provider 변경 시 모델 선택값을 그 provider 의 첫 모델로 재설정 (직접 입력·유효 선택은 유지)
  useEffect(() => {
    if (!resolvedType) {
      setVersion(current => (current ? '' : current))
      return
    }
    const options = modelOptionsFor(resolvedType)
    setVersion(current => (current === CUSTOM_VERSION || options.includes(current) ? current : options[0]))
  }, [resolvedType, catalogStatus, catalog])

  const editingLlm = editingId ? llms.find(llm => llm.id === editingId) : null

  const editResolvedType = useMemo(() => {
    return editProviderSelect || (editingLlm ? editingLlm.type : '')
  }, [editProviderSelect, editingLlm])

  const editKeyRequired = editResolvedType
    ? providerRequiresKey(editResolvedType, catalogProviderMap.get(editResolvedType))
    : true
  const editVersionOptions = editResolvedType ? modelOptionsFor(editResolvedType) : []

  useEffect(() => {
    if (!editingId) return
    const options = editResolvedType ? modelOptionsFor(editResolvedType) : []
    setEditVersion(current => (
      current === CUSTOM_VERSION || options.includes(current) ? current : (options[0] || '')
    ))
  }, [editingId, editResolvedType, catalogStatus, catalog])

  // 등록 목록 갱신 + localStorage 저장
  const persist = next => {
    setLlms(next)
    saveRegisteredLlms(next)
  }

  // 새 모델 등록 — 키 필요 시 서버에 저장(credentialRef 받음), 중복(같은 provider·model·키) 방지
  const handleAdd = async () => {
    if (credentialSaving) return
    const key = keyValue.trim()
    const finalType = providerSelect
    if (!finalType) {
      window.alert('LiteLLM Provider를 선택해주세요.')
      return
    }
    if (keyRequired && !key) {
      window.alert('API 키를 입력해주세요.')
      return
    }

    const meta = metaFor(finalType)
    const selectedVersion = version === CUSTOM_VERSION
      ? customVersion.trim()
      : (versionOptions.includes(version) ? version : versionOptions[0])
    if (!selectedVersion) {
      window.alert('모델명을 입력해주세요.')
      return
    }

    const catalogModel = (catalog?.models || []).find(
      model => model.provider === finalType && model.id === selectedVersion,
    )
    const modelId = catalogModel?.id || versionToModelId(selectedVersion)
    const finalProvider = catalogProviderMap.get(finalType)
    const requiresKey = providerRequiresKey(finalType, finalProvider)
    const keyMasked = requiresKey ? maskKey(key) : NO_KEY_LABEL
    const duplicate = llms.some(
      llm => llm.type === finalType && llm.modelId === modelId
        && (!requiresKey || llm.keyMasked === keyMasked),
    )
    if (duplicate) {
      window.alert(requiresKey ? '이미 같은 키·모델로 등록되어 있어요.' : '이미 등록된 로컬 모델이에요.')
      return
    }

    setCredentialSaving(true)
    try {
      const credential = requiresKey
        ? await saveLlmCredential({ provider: finalType, model: modelId, apiKey: key })
        : null
      const next = [
        ...llms,
        {
          id: nextLlmId(llms),
          type: finalType,
          version: selectedVersion,
          modelId,
          credentialRef: credential?.credential_ref || '',
          keyMasked: credential?.key_masked || keyMasked,
          providerName: meta.name,
          capabilities: catalogModel ? {
            reasoning: catalogModel.supports_reasoning === true,
            vision: catalogModel.supports_vision === true,
            json_schema: catalogModel.supports_response_schema === true,
          } : {},
        },
      ]
      persist(next)
      setKeyValue('')
      setVersion('')
      setCustomVersion('')
      setProviderSelect('')
    } catch (error) {
      window.alert(String(error.message || error))
    } finally {
      setCredentialSaving(false)
    }
  }

  const handleRemove = id => {
    if (!window.confirm('이 LLM 등록을 삭제할까요?')) return
    persist(llms.filter(llm => llm.id !== id))
  }

  // 수정 시작 — 마스킹된 키는 입력 필드에 넣지 않음 (빈 값 = 기존 자격증명 유지)
  const handleEditStart = llm => {
    setEditingId(llm.id)
    setEditProviderSelect(llm.type)
    setEditKeyValue('')
    const meta = metaFor(llm.type)
    const isPresetVersion = meta?.versions.includes(llm.version)
    setEditVersion(isPresetVersion ? llm.version : CUSTOM_VERSION)
    setEditCustomVersion(isPresetVersion ? '' : llm.version)
  }

  const handleEditCancel = () => {
    setEditingId(null)
    setEditProviderSelect('')
    setEditKeyValue('')
    setEditVersion('')
    setEditCustomVersion('')
  }

  // 수정 저장 — provider 변경 시 새 키 필수, 키 변경 시에만 서버 재저장
  const handleEditSave = async () => {
    if (credentialSaving) return
    const current = llms.find(llm => llm.id === editingId)
    if (!current) return

    const key = editKeyValue.trim()
    const finalType = editProviderSelect || current.type
    const providerChanged = finalType !== current.type
    const keyChanged = Boolean(key)
    const requiresKey = providerRequiresKey(finalType, catalogProviderMap.get(finalType))
    let keyMasked = current.keyMasked

    if (requiresKey && providerChanged && !key) {
      window.alert('Provider를 바꾸려면 새 API 키를 입력해주세요.')
      return
    }
    if (requiresKey && keyChanged) {
      keyMasked = maskKey(key)
    } else if (!requiresKey) {
      keyMasked = NO_KEY_LABEL
    }

    const meta = metaFor(finalType)
    const selectedVersion = editVersion === CUSTOM_VERSION
      ? editCustomVersion.trim()
      : (editVersionOptions.includes(editVersion) ? editVersion : editVersionOptions[0])
    if (!selectedVersion) {
      window.alert('모델명을 입력해주세요.')
      return
    }
    const modelId = versionToModelId(selectedVersion)

    setCredentialSaving(true)
    try {
      const credential = requiresKey && keyChanged
        ? await saveLlmCredential({ provider: finalType, model: modelId, apiKey: key })
        : null
      const next = llms.map(llm => (llm.id === editingId
        ? {
          ...llm,
          type: finalType,
          version: selectedVersion,
          modelId,
          credentialRef: credential?.credential_ref || (providerChanged ? '' : llm.credentialRef || ''),
          keyMasked: credential?.key_masked || (providerChanged ? NO_KEY_LABEL : keyMasked),
          providerName: meta.name,
        }
        : llm))
      persist(next)
      handleEditCancel()
    } catch (error) {
      window.alert(String(error.message || error))
    } finally {
      setCredentialSaving(false)
    }
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
              API 키(또는 로컬 모델명)를 등록하면 여기서 등록한 모델만 LLM 조합 만들기에서 선택할 수 있어요.
            </p>
            <div className="ms-add-row ms-add-row--wrap">
              <SearchableSelect
                className="ms-provider-select"
                value={providerSelect}
                options={providerOptions}
                placeholder={catalogStatus === 'loading' ? 'Provider 로딩 중…' : 'Provider 선택'}
                searchPlaceholder="Provider 검색…"
                disabled={catalogStatus !== 'ready'}
                onChange={nextValue => {
                setProviderSelect(nextValue)
                setVersion('')
                setCustomVersion('')
              }}
              />
              <input
                type="password"
                value={keyValue}
                placeholder={keyRequired ? 'API Key 입력' : '로컬 모델은 키가 필요 없어요'}
                disabled={!keyRequired}
                onChange={event => setKeyValue(event.target.value)}
                onKeyDown={event => {
                  if (event.key === 'Enter') void handleAdd()
                }}
              />
              <SearchableSelect
                className="ms-model-select"
                value={version || ''}
                onChange={setVersion}
                options={[
                  ...versionOptions.map(option => ({ value: option, label: option })),
                  ...(resolvedType ? [{ value: CUSTOM_VERSION, label: '직접 입력...' }] : []),
                ]}
                placeholder={versionOptions.length ? '모델 선택' : 'Provider 먼저 선택'}
                searchPlaceholder="모델 검색…"
                disabled={!versionOptions.length}
              />
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
              <button className="ms-btn-primary ms-add-btn" type="button" onClick={() => void handleAdd()} disabled={credentialSaving}>
                {credentialSaving ? '저장 중…' : '등록'}
              </button>
            </div>
            <p className="ms-hint" style={{ marginTop: 10, marginBottom: 0 }}>
              {catalogStatus === 'loading' && 'LiteLLM 지원 목록을 불러오는 중이에요…'}
              {catalogStatus === 'ready' && `LiteLLM 카탈로그에서 ${catalog.providers.length}개 Provider를 불러왔어요.`}
              {catalogStatus === 'error' && `LiteLLM 카탈로그를 불러오지 못했습니다. (${catalogError})`}
            </p>

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
                        <SearchableSelect
                          className="ms-provider-select"
                          value={editProviderSelect}
                          options={providerOptions}
                          placeholder="Provider 선택"
                          searchPlaceholder="Provider 검색…"
                          disabled={catalogStatus !== 'ready'}
                          onChange={setEditProviderSelect}
                        />
                        <input
                          type="password"
                          value={editKeyValue}
                          placeholder={editKeyRequired ? `바꾸려면 새 API 키 입력 (${llm.keyMasked || '미등록'})` : '로컬 모델은 키가 필요 없어요'}
                          disabled={!editKeyRequired}
                          onChange={event => setEditKeyValue(event.target.value)}
                          onKeyDown={event => {
                            if (event.key === 'Enter') void handleEditSave()
                            if (event.key === 'Escape') handleEditCancel()
                          }}
                        />
                        <SearchableSelect
                          className="ms-model-select"
                          value={editVersion || ''}
                          onChange={setEditVersion}
                          options={[
                            ...editVersionOptions.map(option => ({ value: option, label: option })),
                            ...(editResolvedType ? [{ value: CUSTOM_VERSION, label: '직접 입력...' }] : []),
                          ]}
                          placeholder={editVersionOptions.length ? '모델 선택' : 'Provider 먼저 선택'}
                          searchPlaceholder="모델 검색…"
                          disabled={!editVersionOptions.length}
                        />
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
                        <button className="ms-btn-primary ms-add-btn" type="button" onClick={() => void handleEditSave()} disabled={credentialSaving}>
                          {credentialSaving ? '저장 중…' : '저장'}
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
                        <span className="ms-provider-name">{providerMeta(llm.type, llm).name || llm.type}</span>
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
