import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
  applyModelSettingProfile,
  createModelSettingProfile,
  deleteModelSettingProfile,
  listModelSettingProfiles,
  updateModelSettingProfile,
} from '../../api/modelSetupProfiles'
import { summarizeEditorState } from '../../components/model-setup/stageModels'
import { loadRegisteredLlms } from '../../components/model-setup/llmRegistry'
import { buildSetPayload, parseSetEditorState, summarizeSetConfig } from '../../components/model-setup/setBuilder'

function sortProfilesActiveFirst(list) {
  const active = list.filter(profile => profile.is_active)
  const inactive = list.filter(profile => !profile.is_active)
  return [...active, ...inactive]
}

function formatModelLabel(model) {
  return model.version || model.modelId || '모델'
}

function IconEdit() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
      <path d="M4 20h4l10.5-10.5a2.1 2.1 0 0 0-3-3L5 17v3z" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
      <path d="M13.5 6.5l3 3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
}

function IconSearch() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
      <circle cx="11" cy="11" r="6.5" fill="none" stroke="currentColor" strokeWidth="2" />
      <path d="M16 16l5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
}

function IconVideoInput() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
      <rect x="3" y="6" width="12" height="12" rx="2" fill="none" stroke="currentColor" strokeWidth="2" />
      <path d="M15 10.2l6-3.2v10l-6-3.2z" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
    </svg>
  )
}

function IconTextOutput() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
      <rect x="4" y="3" width="16" height="18" rx="2" fill="none" stroke="currentColor" strokeWidth="2" />
      <path d="M8 8h8M8 12h8M8 16h5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
}

function IconFeedback() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
      <path d="M4 5h16v11H8l-4 4V5z" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
      <path d="M8 10h8M8 13h5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
}

const PREPROCESS_INPUT_LABEL = '강의 영상'
const PREPROCESS_OUTPUT_LABEL = '통합 텍스트'
const VERIFY_OUTPUT_LABEL = '피드백'

const PREPROCESS_VIDEO_LABEL = '비디오 분석'
const PREPROCESS_AUDIO_LABEL = '오디오 분석'
const PREPROCESS_VIDEO_STEPS = ['전처리', '슬라이드\n추출', '슬라이드\n중복 처리', '슬라이드\n분석']
const PREPROCESS_AUDIO_STEPS = ['오디오 분리 및\n품질 분석', '음성 전사 및\n텍스트 교정', '발화 구간\n재구성', '발화 내\n강조 신호 분석']

function buildDetailFlow(llmRows = []) {
  const toNode = row => ({
    stageKey: row.stageKey,
    label: row.stageKey === 'slide' ? '슬라이드 텍스트 오류 점검' : row.label,
    kind: 'llm',
    models: Array.isArray(row.models) ? row.models : [],
  })

  const slideRow = llmRows.find(row => row.stageKey === 'slide')
  const chainRows = llmRows.filter(row => row.stageKey !== 'slide')

  return {
    chain: chainRows.map(toNode),
    parallel: slideRow ? [toNode(slideRow)] : [],
  }
}

function PipeCard({ node, badge }) {
  return (
    <article className={`ms-pipe-card ms-pipe-card--${node.kind}`}>
      <div className="ms-pipe-card-top">
        <span className="ms-pipe-idx">{badge}</span>
        <h4 className="ms-pipe-title">{node.label}</h4>
      </div>
      {node.kind === 'llm' && (
        <div className="ms-pipe-models">
          {node.models.length ? node.models.map(model => {
            const label = formatModelLabel(model)
            return (
              <span
                key={`${node.stageKey}-${model.version}-${model.providerType}`}
                className={`ms-pipe-model-chip ms-preset-tag--${model.providerType}`}
                title={label}
                aria-label={label}
              >
                {label.charAt(0).toUpperCase()}
              </span>
            )
          }) : (
            <span className="ms-pipe-models-empty">미설정</span>
          )}
        </div>
      )}
    </article>
  )
}

function PipeArrow() {
  return (
    <div className="ms-pipe-arrow" aria-hidden="true">
      <svg viewBox="0 0 32 32" width="14" height="14">
        <path d="M4 16h20" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
        <path d="M18 8l10 8-10 8" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  )
}

function PipeArrowDown() {
  return (
    <div className="ms-pipe-arrow-divider">
      <div className="ms-pipe-arrow ms-pipe-arrow--down" aria-hidden="true">
        <svg viewBox="0 0 32 32" width="52" height="52">
          <path d="M16 2v22" fill="none" stroke="currentColor" strokeWidth="5" strokeLinecap="round" />
          <path d="M6 18l10 10 10-10" fill="none" stroke="currentColor" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
    </div>
  )
}

function PipeGroupHead({ index, title }) {
  return (
    <div className="ms-pipe-group-head">
      <h4 className="ms-pipe-group-title">{index}. {title}</h4>
    </div>
  )
}

function PipeIO({ icon, label }) {
  return (
    <div className="ms-pipe-io">
      <div className="ms-pipe-io-icon">{icon}</div>
      <span className="ms-pipe-io-label">{label}</span>
    </div>
  )
}

function PipeLanesTable() {
  return (
    <div className="ms-pipe-lanes-table">
      <div className="ms-pipe-lanes-row ms-pipe-lanes-row--head">
        <span className="ms-pipe-lane-legend-item" aria-hidden="true" />
        {PREPROCESS_VIDEO_STEPS.map((_, index) => (
          <span className="ms-pipe-idx ms-pipe-stage-idx" key={index}>{index + 1}</span>
        ))}
      </div>
      <div className="ms-pipe-lanes-row">
        <span className="ms-pipe-lane-legend-item">{PREPROCESS_VIDEO_LABEL}</span>
        {PREPROCESS_VIDEO_STEPS.map(label => (
          <div className="ms-pipe-step" key={label}>{label}</div>
        ))}
      </div>
      <div className="ms-pipe-lanes-row">
        <span className="ms-pipe-lane-legend-item">{PREPROCESS_AUDIO_LABEL}</span>
        {PREPROCESS_AUDIO_STEPS.map(label => (
          <div className="ms-pipe-step" key={label}>{label}</div>
        ))}
      </div>
    </div>
  )
}

function PresetFlow({ flow }) {
  const scrollRef = useRef(null)

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollLeft = 0
  }, [flow])

  return (
    <div className="ms-pipe">
      <div className="ms-pipe-scroll" ref={scrollRef}>
        <div className="ms-pipe-stack">
          <section className="ms-pipe-group">
            <PipeGroupHead index="I" title="전처리" />
            <div className="ms-pipe-group-box ms-pipe-group-box--preprocess">
              <PipeIO icon={<IconVideoInput />} label={PREPROCESS_INPUT_LABEL} />

              <PipeArrow />

              <div className="ms-pipe-parallel ms-pipe-parallel--lanes">
                <PipeLanesTable />
              </div>

              <PipeArrow />

              <PipeIO icon={<IconTextOutput />} label={PREPROCESS_OUTPUT_LABEL} />
            </div>
          </section>

          <PipeArrowDown />

          <section className="ms-pipe-group">
            <PipeGroupHead index="II" title="검증" />
            <div className="ms-pipe-group-box ms-pipe-group-box--preprocess">
              <PipeIO icon={<IconTextOutput />} label={PREPROCESS_OUTPUT_LABEL} />

              <PipeArrow />

              <div className="ms-pipe-parallel ms-pipe-parallel--lanes">
                <div className="ms-pipe-verify-lanes">
                  <div className="ms-pipe-verify-row">
                    <span className="ms-pipe-lane-legend-item">발화 분석</span>
                    <div className="ms-pipe-chain">
                      {flow.chain.map((node, index) => (
                        <div className="ms-pipe-follow" key={node.stageKey}>
                          {index > 0 && <PipeArrow />}
                          <PipeCard node={node} badge={String(index + 1)} />
                        </div>
                      ))}
                    </div>
                  </div>
                  {flow.parallel.length > 0 && (
                    <div className="ms-pipe-verify-row ms-pipe-verify-row--slide">
                      <span className="ms-pipe-lane-legend-item">슬라이드 분석</span>
                      <div className="ms-pipe-chain">
                        {flow.parallel.map(node => (
                          <div className="ms-pipe-follow" key={node.stageKey}>
                            <PipeCard node={node} badge="1" />
                          </div>
                        ))}
                        <div className="ms-pipe-follow">
                          <PipeArrow />
                          <PipeCard
                            node={{
                              stageKey: 'slide-formula',
                              label: '코드/수식 오류 점검',
                              kind: 'llm',
                              models: flow.parallel[0]?.models || [],
                            }}
                            badge="2"
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <PipeArrow />

              <PipeIO icon={<IconFeedback />} label={VERIFY_OUTPUT_LABEL} />
            </div>
          </section>
        </div>
      </div>

      <p className="ms-pipe-note">
        강의 영상은 비디오·오디오로 병렬 분석된 뒤 통합 텍스트를 도출합니다. 이를 바탕으로 발화를 분석하며 슬라이드 분석이 병렬 처리됩니다.
      </p>
    </div>
  )
}

export default function ModelSetsPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [searchParams, setSearchParams] = useSearchParams()

  const [selectedId, setSelectedId] = useState(null)
  const [detailId, setDetailId] = useState(null)
  const builderRef = useRef(null)

  const [editingProfileId, setEditingProfileId] = useState(() => searchParams.get('edit'))
  const [registeredLlms] = useState(() => loadRegisteredLlms())
  const [name, setName] = useState('')
  const [selectedIds, setSelectedIds] = useState([])
  const [mainLlmId, setMainLlmId] = useState('')
  const [includeGrounding, setIncludeGrounding] = useState(true)
  const [confirmOpen, setConfirmOpen] = useState(false)

  const { data, isLoading, error } = useQuery({
    queryKey: ['model-setting-profiles'],
    queryFn: listModelSettingProfiles,
  })

  const applyMutation = useMutation({
    mutationFn: applyModelSettingProfile,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['model-setting-profiles'] })
      setSelectedId(null)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deleteModelSettingProfile,
    onSuccess: (_data, profileId) => {
      setDetailId(current => (current === profileId ? null : current))
      setSelectedId(current => (current === profileId ? null : current))
      if (String(editingProfileId) === String(profileId)) startCreate()
      queryClient.invalidateQueries({ queryKey: ['model-setting-profiles'] })
    },
  })

  const saveMutation = useMutation({
    mutationFn: payload => (
      editingProfileId
        ? updateModelSettingProfile(editingProfileId, payload)
        : createModelSettingProfile(payload)
    ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['model-setting-profiles'] })
    },
  })

  const allProfiles = useMemo(() => data?.profiles || [], [data])

  useEffect(() => {
    if (!detailId) return undefined
    const onKeyDown = event => {
      if (event.key === 'Escape') setDetailId(null)
    }
    document.body.style.overflow = 'hidden'
    window.addEventListener('keydown', onKeyDown)
    return () => {
      document.body.style.overflow = ''
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [detailId])

  const profiles = useMemo(
    () => sortProfilesActiveFirst(allProfiles),
    [allProfiles],
  )

  const detailProfile = useMemo(
    () => profiles.find(profile => profile.id === detailId) || null,
    [profiles, detailId],
  )
  const detailFlow = useMemo(
    () => (detailProfile ? buildDetailFlow(summarizeEditorState(detailProfile.editor_state || {})) : null),
    [detailProfile],
  )
  const detailSummary = useMemo(
    () => (detailProfile ? summarizeSetConfig(detailProfile.editor_state || {}) : null),
    [detailProfile],
  )
  const editingProfile = useMemo(
    () => allProfiles.find(profile => String(profile.id) === String(editingProfileId)) || null,
    [allProfiles, editingProfileId],
  )

  const availableLlms = useMemo(() => {
    if (!editingProfile) return registeredLlms
    const parsed = parseSetEditorState(editingProfile.editor_state, registeredLlms)
    const byId = new Map(registeredLlms.map(llm => [llm.id, llm]))
    parsed.selectedLlms.forEach(llm => {
      if (!byId.has(llm.id)) byId.set(llm.id, llm)
    })
    return [...byId.values()]
  }, [editingProfile, registeredLlms])

  const selectedLlms = useMemo(
    () => availableLlms.filter(llm => selectedIds.includes(llm.id)),
    [availableLlms, selectedIds],
  )

  useEffect(() => {
    if (!editingProfileId) return
    if (!editingProfile) return
    const parsed = parseSetEditorState(editingProfile.editor_state, registeredLlms)
    setName(editingProfile.name || '')
    setSelectedIds(parsed.selectedLlmIds)
    setMainLlmId(parsed.mainLlmId)
    setIncludeGrounding(parsed.includeGrounding)
  }, [editingProfileId, editingProfile, registeredLlms])


  useEffect(() => {
    if (!selectedIds.length) {
      if (mainLlmId) setMainLlmId('')
      return
    }
    if (!selectedIds.includes(mainLlmId)) {
      setMainLlmId(selectedIds[0])
    }
  }, [selectedIds, mainLlmId])

  const selectProfile = profileId => {
    setSelectedId(current => (current === profileId ? null : profileId))
  }

  const openDetail = profileId => {
    setDetailId(profileId)
  }

  const handleConfirmApply = () => {
    if (!selectedId || applyMutation.isPending) return
    applyMutation.mutate(selectedId)
  }

  const focusBuilder = () => {
    builderRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  function startCreate() {
    setEditingProfileId(null)
    setName('')
    setSelectedIds([])
    setMainLlmId('')
    setIncludeGrounding(true)
    setSearchParams(prev => {
      const next = new URLSearchParams(prev)
      next.delete('edit')
      return next
    }, { replace: true })
  }

  const startEdit = profileId => {
    setEditingProfileId(String(profileId))
    setSearchParams(prev => {
      const next = new URLSearchParams(prev)
      next.set('edit', String(profileId))
      return next
    }, { replace: true })
    focusBuilder()
  }

  const toggleLlm = id => {
    setSelectedIds(current => (
      current.includes(id)
        ? current.filter(item => item !== id)
        : [...current, id]
    ))
  }

  const buildPayloadOrAlert = () => {
    if (!(name || '').trim()) {
      window.alert('LLM 조합 이름을 입력해주세요.')
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
    if (saveMutation.isPending) return
    if (!buildPayloadOrAlert()) return
    setConfirmOpen(true)
  }

  const handleConfirmSave = async () => {
    const payload = buildPayloadOrAlert()
    if (!payload) return
    setConfirmOpen(false)
    try {
      await saveMutation.mutateAsync(payload)
      startCreate()
    } catch {
      // saveMutation.error로 표시
    }
  }

  return (
    <section className="model-setup">
      <div className="ms-header-row">
        <h2 className="ms-app-title">Multi-LLM 구성하기</h2>
        <button className="ms-back-btn" type="button" onClick={() => navigate('/model-setup')} aria-label="선택 화면으로">
          ←
        </button>
      </div>

      <div className="ms-stack">
        <div className="ms-stack-section" ref={builderRef}>
          <h3 className="ms-split-title">
            {editingProfileId ? `"${editingProfile?.name || ''}" 수정` : '새 조합 만들기'}
          </h3>

          <div className="ms-card">
            <div className="ms-set-field">
              <p className="ms-label">LLM 조합 이름</p>
              <input
                className="ms-name-input"
                type="text"
                value={name}
                placeholder="예: 오픈소스 개발자대회 시연용"
                onChange={event => setName(event.target.value)}
              />
            </div>

            <div className="ms-set-field">
              <div className="ms-set-section-head">
                <div className="ms-set-label-row">
                  <p className="ms-label ms-label--tight">사용할 LLM 선택</p>
                  <span className="ms-set-label-sep" aria-hidden="true">|</span>
                  <p className="ms-hint ms-hint--inline">
                    등록된 모델 중 검증에 쓰일 모델을 선택하고, 대표 모델에 별 표시를 남겨주세요.
                  </p>
                </div>
                <button className="ms-link-btn ms-link-btn--compact" type="button" onClick={() => navigate('/model-setup/models')}>
                  모델 추가 등록
                </button>
              </div>

              {availableLlms.length ? (
                <div className="ms-llm-pick-wrap">
                  {selectedIds.length > 0 && (
                    <p className="ms-main-bubble" role="note">
                      대표 모델은 Claim 추출·슬라이드 오류 검사·웹 그라운딩처럼 모델 하나만 쓰는 단계에서 사용돼요.
                    </p>
                  )}
                  <div className="ms-llm-pick-grid">
                    {availableLlms.map(llm => {
                      const checked = selectedIds.includes(llm.id)
                      const isMain = mainLlmId === llm.id
                      return (
                        <label
                          className={`ms-llm-pick-card${checked ? ' ms-llm-pick-card--on' : ''}`}
                          key={llm.id}
                        >
                          <input
                            type="checkbox"
                            className="ms-visually-hidden"
                            checked={checked}
                            onChange={() => toggleLlm(llm.id)}
                          />
                          <strong>{llm.version}</strong>
                          {checked && (
                            <button
                              type="button"
                              className={`ms-llm-pick-main-star${isMain ? ' ms-llm-pick-main-star--on' : ''}`}
                              disabled={isMain}
                              aria-label={isMain ? '메인으로 지정됨' : '메인으로 지정'}
                              onClick={event => {
                                event.preventDefault()
                                event.stopPropagation()
                                setMainLlmId(llm.id)
                              }}
                            >
                              {isMain ? '★' : '☆'}
                            </button>
                          )}
                        </label>
                      )
                    })}
                  </div>
                </div>
              ) : (
                <p className="ms-empty">등록된 LLM이 없어요. 먼저 LLM 모델 화면에서 등록해 주세요.</p>
              )}
            </div>

            <div className="ms-set-field">
              <div className="ms-set-section-head">
                <div className="ms-set-label-row">
                  <p className="ms-label ms-label--tight">웹 그라운딩</p>
                  <span className="ms-set-label-sep" aria-hidden="true">|</span>
                  <p className="ms-hint ms-hint--inline">웹 검색으로 이슈를 재검증하는 단계를 포함할지 선택합니다.</p>
                </div>
                <div className="ms-radio-row">
                  <label className="ms-radio-option">
                    <input
                      type="radio"
                      name="includeGrounding"
                      checked={includeGrounding}
                      onChange={() => setIncludeGrounding(true)}
                    />
                    포함
                  </label>
                  <label className="ms-radio-option">
                    <input
                      type="radio"
                      name="includeGrounding"
                      checked={!includeGrounding}
                      onChange={() => setIncludeGrounding(false)}
                    />
                    미포함
                  </label>
                </div>
              </div>
            </div>

            {saveMutation.error && (
              <p className="ms-save-error">{String(saveMutation.error.message || saveMutation.error)}</p>
            )}

            <div className="ms-actions">
              <button
                className="ms-btn-secondary"
                type="button"
                onClick={startCreate}
                disabled={!editingProfileId && !name && !selectedIds.length}
              >
                {editingProfileId ? '편집 취소' : '입력 지우기'}
              </button>
              <button
                className="ms-btn-primary"
                type="button"
                disabled={saveMutation.isPending}
                onClick={handleSave}
              >
                {saveMutation.isPending ? '저장 중…' : editingProfileId ? '수정 저장' : '만들기'}
              </button>
            </div>
          </div>
        </div>

        <div className="ms-stack-section">
          <h3 className="ms-split-title">생성된 셋 <span className="ms-split-title-count">{profiles.length}</span></h3>
          {isLoading && <div className="ms-card"><p className="ms-empty">셋을 불러오는 중이에요…</p></div>}
          {error && <div className="ms-card"><p className="ms-save-error">{String(error.message || error)}</p></div>}

          {!isLoading && !error && (
            <div className="ms-preset-list">
              {profiles.length ? profiles.map((profile, profileIndex) => {
                const summary = summarizeSetConfig(profile.editor_state || {})
                const isSelected = selectedId === profile.id
                const isEditingThis = String(editingProfileId) === String(profile.id)

                return (
                  <article
                    className={[
                      'ms-preset-card',
                      'ms-preset-card--row',
                      profile.is_active ? 'ms-preset-card--active' : '',
                      isSelected ? 'ms-preset-card--selected' : '',
                      isEditingThis ? 'ms-preset-card--selected' : '',
                    ].filter(Boolean).join(' ')}
                    key={profile.id}
                    onClick={event => {
                      if (event.target.closest('.ms-preset-actions, .ms-preset-row-delete, .ms-preset-icon-btn')) return
                      selectProfile(profile.id)
                    }}
                  >
                    <span className="ms-provider-index">{profileIndex + 1}</span>
                    <div className="ms-preset-summary ms-preset-summary--row">
                      <div className="ms-preset-title-row">
                        <h3>{profile.name}</h3>
                        <span className="ms-preset-count">{summary.modelCount}개 LLM</span>
                        {profile.is_active && <span className="ms-preset-badge">적용 중</span>}
                        {isEditingThis && <span className="ms-preset-badge ms-preset-badge--apply">편집 중</span>}
                      </div>
                      <div className="ms-preset-tags">
                        {summary.models.length ? summary.models.map(model => (
                          <span className={`ms-preset-tag ms-preset-tag--${model.type}`} key={model.id}>
                            {model.isMain && <span aria-hidden="true">★</span>}
                            {model.version}
                          </span>
                        )) : (
                          <span className="ms-preset-tag ms-preset-tag--empty">모델 미설정</span>
                        )}
                        <span className={`ms-preset-tag ms-preset-tag--meta${summary.includeGrounding ? '' : ' ms-preset-tag--off'}`}>
                          <span aria-hidden="true">{summary.includeGrounding ? '✓' : '✕'}</span>
                          웹그라운딩 {summary.includeGrounding ? '포함' : '미포함'}
                        </span>
                      </div>
                    </div>

                    <div className="ms-preset-actions ms-preset-actions--row">
                      {isSelected && !profile.is_active && (
                        <button
                          className="ms-btn-primary ms-preset-confirm-btn"
                          type="button"
                          disabled={applyMutation.isPending}
                          onClick={handleConfirmApply}
                        >
                          {applyMutation.isPending ? '적용 중…' : '적용'}
                        </button>
                      )}
                      <button
                        className="ms-preset-icon-btn"
                        type="button"
                        aria-label={`${profile.name} 상세 보기`}
                        onClick={() => openDetail(profile.id)}
                      >
                        <IconSearch />
                      </button>
                      <button
                        className="ms-preset-icon-btn"
                        type="button"
                        aria-label={`${profile.name} 수정`}
                        onClick={() => startEdit(profile.id)}
                      >
                        <IconEdit />
                      </button>
                      <button
                        className="ms-preset-row-delete"
                        type="button"
                        disabled={deleteMutation.isPending}
                        aria-label={`${profile.name} 삭제`}
                        onClick={() => {
                          if (window.confirm(`"${profile.name}" 셋을 삭제할까요?`)) {
                            deleteMutation.mutate(profile.id)
                          }
                        }}
                      >
                        ×
                      </button>
                    </div>
                  </article>
                )
              }) : (
                <div className="ms-card ms-preset-empty"><p className="ms-empty">아직 생성된 셋이 없어요.</p></div>
              )}
            </div>
          )}
        </div>
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

      {detailProfile && createPortal(
        <div
          className="ms-preset-detail-backdrop"
          role="presentation"
          onClick={() => setDetailId(null)}
        >
          <div
            className={`ms-preset-detail-modal${detailProfile.is_active ? ' ms-preset-detail-modal--active' : ''}`}
            role="dialog"
            aria-modal="true"
            aria-labelledby="ms-preset-detail-title"
            onClick={event => event.stopPropagation()}
          >
            <div className="ms-preset-detail-head">
              <div>
                <div className="ms-preset-title-row">
                  <h3 id="ms-preset-detail-title">{detailProfile.name}</h3>
                  <span className="ms-preset-count">{detailSummary?.modelCount || 0}개 LLM</span>
                  {detailProfile.is_active && <span className="ms-preset-badge">적용 중</span>}
                </div>
              </div>
              <button className="ms-preset-detail-close" type="button" onClick={() => setDetailId(null)} aria-label="닫기">
                ×
              </button>
            </div>

            <div className="ms-preset-tags ms-preset-tags--detail">
              {detailSummary?.models?.length ? detailSummary.models.map(model => (
                <span className={`ms-preset-tag ms-preset-tag--${model.type}`} key={model.id}>
                  {model.isMain && <span aria-hidden="true">★</span>}
                  {model.version}
                </span>
              )) : (
                <span className="ms-preset-tag ms-preset-tag--empty">모델 미설정</span>
              )}
              <span className={`ms-preset-tag ms-preset-tag--meta${detailSummary?.includeGrounding ? '' : ' ms-preset-tag--off'}`}>
                <span aria-hidden="true">{detailSummary?.includeGrounding ? '✓' : '✕'}</span>
                웹그라운딩 {detailSummary?.includeGrounding ? '포함' : '미포함'}
              </span>
            </div>

            {detailFlow && <PresetFlow flow={detailFlow} />}
          </div>
        </div>,
        document.body,
      )}
    </section>
  )
}
