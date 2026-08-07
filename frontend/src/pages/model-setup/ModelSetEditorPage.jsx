import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import {
  createModelSettingProfile,
  getModelSettingProfile,
  updateModelSettingProfile,
} from '../../api/modelSetupProfiles'
import { llmLabel, loadRegisteredLlms } from '../../components/model-setup/llmRegistry'
import { buildSetPayload, parseSetEditorState } from '../../components/model-setup/setBuilder'

export default function ModelSetEditorPage({ mode }) {
  const navigate = useNavigate()
  const { profileId } = useParams()
  const queryClient = useQueryClient()
  const isEdit = mode === 'edit'

  const [registeredLlms] = useState(() => loadRegisteredLlms())
  const [name, setName] = useState('')
  const [selectedIds, setSelectedIds] = useState([])
  const [mainLlmId, setMainLlmId] = useState('')
  const [includeGrounding, setIncludeGrounding] = useState(true)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [doneOpen, setDoneOpen] = useState(false)
  const [hydrated, setHydrated] = useState(!isEdit)

  const profileQuery = useQuery({
    queryKey: ['model-setting-profile', profileId],
    queryFn: () => getModelSettingProfile(profileId),
    enabled: isEdit && Boolean(profileId),
  })

  const mutation = useMutation({
    mutationFn: payload => (
      isEdit
        ? updateModelSettingProfile(profileId, payload)
        : createModelSettingProfile(payload)
    ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['model-setting-profiles'] })
    },
  })

  useEffect(() => {
    if (!isEdit || !profileQuery.data || hydrated) return
    const parsed = parseSetEditorState(profileQuery.data.editor_state, registeredLlms)
    setName(profileQuery.data.name || '')
    setSelectedIds(parsed.selectedLlmIds)
    setMainLlmId(parsed.mainLlmId)
    setIncludeGrounding(parsed.includeGrounding)
    setHydrated(true)
  }, [isEdit, profileQuery.data, registeredLlms, hydrated])

  const availableLlms = useMemo(() => {
    if (!isEdit || !profileQuery.data) return registeredLlms
    const parsed = parseSetEditorState(profileQuery.data.editor_state, registeredLlms)
    const byId = new Map(registeredLlms.map(llm => [llm.id, llm]))
    parsed.selectedLlms.forEach(llm => {
      if (!byId.has(llm.id)) byId.set(llm.id, llm)
    })
    return [...byId.values()]
  }, [isEdit, profileQuery.data, registeredLlms])

  const selectedLlms = useMemo(
    () => availableLlms.filter(llm => selectedIds.includes(llm.id)),
    [availableLlms, selectedIds],
  )

  useEffect(() => {
    if (!selectedIds.length) {
      if (mainLlmId) setMainLlmId('')
      return
    }
    if (!selectedIds.includes(mainLlmId)) {
      setMainLlmId(selectedIds[0])
    }
  }, [selectedIds, mainLlmId])

  const toggleLlm = id => {
    setSelectedIds(current => (
      current.includes(id)
        ? current.filter(item => item !== id)
        : [...current, id]
    ))
  }

  const buildPayloadOrAlert = () => {
    if (!(name || '').trim()) {
      window.alert('셋 이름을 입력해주세요.')
      return null
    }
    if (!selectedLlms.length) {
      window.alert('사용할 LLM을 하나 이상 선택해주세요.')
      return null
    }
    if (!mainLlmId) {
      window.alert('메인 LLM을 지정해주세요.')
      return null
    }
    try {
      return buildSetPayload({
        name,
        selectedLlms,
        mainLlmId,
        includeGrounding,
      })
    } catch (error) {
      window.alert(String(error.message || error))
      return null
    }
  }

  const handleSave = () => {
    if (mutation.isPending) return
    if (!buildPayloadOrAlert()) return
    setConfirmOpen(true)
  }

  const handleConfirmSave = async () => {
    const payload = buildPayloadOrAlert()
    if (!payload) return
    setConfirmOpen(false)
    try {
      await mutation.mutateAsync(payload)
      if (isEdit) navigate('/model-setup/sets')
      else setDoneOpen(true)
    } catch {
      // mutation.error로 표시
    }
  }

  if (isEdit && profileQuery.isLoading) {
    return (
      <section className="model-setup">
        <div className="ms-card"><p className="ms-empty">셋을 불러오는 중이에요…</p></div>
      </section>
    )
  }

  if (isEdit && profileQuery.error) {
    return (
      <section className="model-setup">
        <div className="ms-card">
          <p className="ms-save-error">{String(profileQuery.error.message || profileQuery.error)}</p>
        </div>
      </section>
    )
  }

  return (
    <section className="model-setup">
      <div className="ms-header-row">
        <h2 className="ms-app-title">{isEdit ? 'LLM 셋 수정' : 'LLM 셋 만들기'}</h2>
        <button
          className="ms-link-btn ms-link-btn--compact"
          type="button"
          onClick={() => navigate(isEdit ? '/model-setup/sets' : '/model-setup')}
        >
          돌아가기
        </button>
      </div>

      <div className="ms-card">
        <p className="ms-label">셋 이름</p>
        <input
          className="ms-name-input"
          type="text"
          value={name}
          placeholder="예: 기본 검증 셋"
          onChange={event => setName(event.target.value)}
        />
      </div>

      <div className="ms-card">
        <div className="ms-set-section-head">
          <div>
            <p className="ms-label ms-label--tight">사용할 LLM 선택</p>
            <p className="ms-hint">등록한 모델 중 이 셋에서 쓸 LLM을 고르세요. Multi 단계에는 선택한 모델이 모두 쓰입니다.</p>
          </div>
          <button className="ms-link-btn ms-link-btn--compact" type="button" onClick={() => navigate('/model-setup/models')}>
            모델 등록
          </button>
        </div>

        {availableLlms.length ? (
          <div className="ms-llm-pick-grid">
            {availableLlms.map(llm => {
              const checked = selectedIds.includes(llm.id)
              return (
                <label
                  className={`ms-llm-pick-card${checked ? ' ms-llm-pick-card--on' : ''}`}
                  key={llm.id}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleLlm(llm.id)}
                  />
                  <span className="ms-llm-pick-provider">{llm.providerName || llm.type}</span>
                  <strong>{llm.version}</strong>
                  {llm.keyMasked ? <span className="ms-llm-pick-key">{llm.keyMasked}</span> : null}
                </label>
              )
            })}
          </div>
        ) : (
          <p className="ms-empty">등록된 LLM이 없어요. 먼저 LLM 모델 등록으로 이동해 주세요.</p>
        )}
      </div>

      <div className="ms-card">
        <p className="ms-label">메인 LLM</p>
        <p className="ms-hint">Claim 추출·슬라이드 오류 등 단일 LLM 단계 전체에 공통으로 사용됩니다.</p>
        <select
          className="ms-name-input"
          value={mainLlmId}
          disabled={!selectedLlms.length}
          onChange={event => setMainLlmId(event.target.value)}
        >
          {!selectedLlms.length && <option value="">먼저 LLM을 선택하세요</option>}
          {selectedLlms.map(llm => (
            <option value={llm.id} key={llm.id}>{llmLabel(llm)}</option>
          ))}
        </select>
      </div>

      <div className="ms-card">
        <p className="ms-label">웹 그라운딩</p>
        <p className="ms-hint">웹 검색으로 이슈를 재검증하는 단계를 포함할지 선택합니다.</p>
        <div className="ms-toggle-row">
          <button
            type="button"
            className={`ms-btn-toggle${includeGrounding ? ' ms-btn-toggle--on' : ''}`}
            onClick={() => setIncludeGrounding(true)}
          >
            포함
          </button>
          <button
            type="button"
            className={`ms-btn-toggle${!includeGrounding ? ' ms-btn-toggle--on' : ''}`}
            onClick={() => setIncludeGrounding(false)}
          >
            미포함
          </button>
        </div>
      </div>

      {mutation.error && (
        <p className="ms-save-error">{String(mutation.error.message || mutation.error)}</p>
      )}

      <div className="ms-actions">
        <button
          className="ms-btn-secondary"
          type="button"
          onClick={() => navigate(isEdit ? '/model-setup/sets' : '/model-setup')}
        >
          취소
        </button>
        <button
          className="ms-btn-primary"
          type="button"
          disabled={mutation.isPending}
          onClick={handleSave}
        >
          {mutation.isPending ? '저장 중…' : '저장하기'}
        </button>
      </div>

      {confirmOpen && (
        <div className="ms-modal-backdrop" role="presentation" onClick={() => setConfirmOpen(false)}>
          <div className="ms-modal" role="dialog" aria-modal="true" onClick={event => event.stopPropagation()}>
            <p>저장하시겠습니까?</p>
            <div className="ms-modal-actions">
              <button className="ms-btn-secondary" type="button" onClick={() => setConfirmOpen(false)}>취소</button>
              <button className="ms-btn-primary" type="button" onClick={handleConfirmSave}>확인</button>
            </div>
          </div>
        </div>
      )}

      {doneOpen && (
        <div className="ms-modal-backdrop" role="presentation">
          <div className="ms-modal" role="dialog" aria-modal="true">
            <p>저장 완료했습니다. 설정 목록에서 확인해 보시겠습니까?</p>
            <div className="ms-modal-actions">
              <button className="ms-btn-secondary" type="button" onClick={() => navigate('/')}>메인으로</button>
              <button className="ms-btn-primary" type="button" onClick={() => navigate('/model-setup/sets')}>목록 보기</button>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
