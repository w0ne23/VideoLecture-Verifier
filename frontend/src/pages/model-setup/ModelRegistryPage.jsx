import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  PROVIDERS_META,
  detectProvider,
  llmLabel,
  loadRegisteredLlms,
  maskKey,
  nextLlmId,
  saveRegisteredLlms,
  versionToModelId,
} from '../../components/model-setup/llmRegistry'

export default function ModelRegistryPage() {
  const navigate = useNavigate()
  const [llms, setLlms] = useState(() => loadRegisteredLlms())
  const [providerSelect, setProviderSelect] = useState('auto')
  const [keyValue, setKeyValue] = useState('')
  const [version, setVersion] = useState('')
  const [showHelp, setShowHelp] = useState(false)
  const [remember, setRemember] = useState(true)

  const resolvedType = useMemo(() => {
    if (providerSelect !== 'auto') return providerSelect
    return detectProvider(keyValue.trim()) || ''
  }, [providerSelect, keyValue])

  const versionOptions = resolvedType ? PROVIDERS_META[resolvedType].versions : []

  useEffect(() => {
    if (!resolvedType) {
      setVersion(current => (current ? '' : current))
      return
    }
    const options = PROVIDERS_META[resolvedType].versions
    setVersion(current => (options.includes(current) ? current : options[0]))
  }, [resolvedType])

  const persist = next => {
    setLlms(next)
    if (remember) saveRegisteredLlms(next)
    else localStorage.removeItem('verilec_registered_llms')
  }

  const handleAdd = () => {
    const key = keyValue.trim()
    if (!key) {
      window.alert('API 키를 입력해주세요.')
      return
    }

    const detected = detectProvider(key)
    let finalType = providerSelect
    if (providerSelect === 'auto') {
      if (!detected) {
        window.alert('키 형식을 인식할 수 없어요. provider를 직접 선택해주세요.')
        return
      }
      finalType = detected
    } else if (detected && detected !== providerSelect) {
      const useDetected = window.confirm(
        `입력하신 키는 ${PROVIDERS_META[detected].name} 형식으로 보여요.\n`
        + `선택하신 ${PROVIDERS_META[providerSelect].name} 대신 ${PROVIDERS_META[detected].name}로 등록할까요?`,
      )
      if (!useDetected) return
      finalType = detected
    }

    const meta = PROVIDERS_META[finalType]
    const selectedVersion = versionOptions.includes(version) ? version : meta.versions[0]
    const modelId = versionToModelId(selectedVersion)
    const duplicate = llms.some(
      llm => llm.type === finalType && llm.modelId === modelId && llm.keyMasked === maskKey(key),
    )
    if (duplicate) {
      window.alert('이미 같은 키·모델로 등록되어 있어요.')
      return
    }

    const next = [
      ...llms,
      {
        id: nextLlmId(llms),
        type: finalType,
        version: selectedVersion,
        modelId,
        keyMasked: maskKey(key),
        providerName: meta.name,
      },
    ]
    persist(next)
    setKeyValue('')
    setVersion('')
    setProviderSelect('auto')
  }

  const handleRemove = id => {
    if (!window.confirm('이 LLM 등록을 삭제할까요?')) return
    persist(llms.filter(llm => llm.id !== id))
  }

  return (
    <section className="model-setup">
      <div className="ms-header-row">
        <h2 className="ms-app-title">LLM 모델 등록</h2>
        <button className="ms-link-btn ms-link-btn--compact" type="button" onClick={() => navigate('/model-setup')}>
          선택 화면으로
        </button>
      </div>

      <div className="ms-layout">
        <aside className="ms-side-panel">
          <p>
            API 키로 LLM을 등록해요. 여기서 등록한 모델만 LLM 셋 만들기에서 선택할 수 있어요.
          </p>
        </aside>

        <div className="ms-main-panel">
          <div className="ms-card">
            <p className="ms-label">등록된 LLM</p>
            {llms.length ? (
              <div className="ms-provider-list">
                {llms.map(llm => (
                  <div className="ms-provider-row" key={llm.id}>
                    <span className="ms-provider-name">{PROVIDERS_META[llm.type]?.name || llm.type}</span>
                    <span className="ms-llm-version-chip">{llm.version}</span>
                    <span className="ms-provider-key" title={llm.keyMasked}>{llm.keyMasked}</span>
                    <span className="ms-provider-check">등록됨</span>
                    <button className="ms-icon-btn" type="button" onClick={() => handleRemove(llm.id)} aria-label={`${llmLabel(llm)} 삭제`}>
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <p className="ms-empty">아직 등록된 LLM이 없어요.</p>
            )}

            <div className="ms-add-row ms-add-row--wrap">
              <select value={providerSelect} onChange={event => {
                setProviderSelect(event.target.value)
                setVersion('')
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
                placeholder="API Key 입력"
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
              </select>
              <button className="ms-btn-primary ms-add-btn" type="button" onClick={handleAdd}>
                등록
              </button>
            </div>

            <label className="ms-checkbox-row">
              <input
                type="checkbox"
                checked={remember}
                onChange={event => {
                  const next = event.target.checked
                  setRemember(next)
                  if (next) saveRegisteredLlms(llms)
                  else localStorage.removeItem('verilec_registered_llms')
                }}
              />
              이 브라우저에 등록 정보 저장
            </label>

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
          </div>

          <div className="ms-actions">
            <button className="ms-btn-secondary" type="button" onClick={() => navigate('/model-setup')}>
              돌아가기
            </button>
            <button className="ms-btn-primary" type="button" onClick={() => navigate('/model-setup/sets/new')}>
              셋 만들기로
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}
