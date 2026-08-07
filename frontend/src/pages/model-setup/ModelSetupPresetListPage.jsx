import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  applyModelSettingProfile,
  deleteModelSettingProfile,
  listModelSettingProfiles,
} from '../../api/modelSetupProfiles'
import { summarizeEditorState } from '../../components/model-setup/stageModels'
import { summarizeSetConfig } from '../../components/model-setup/setBuilder'

const ORDER_STORAGE_KEY = 'verilec-model-setup-preset-order'

function formatUpdatedAt(value) {
  if (!value) return '시간 정보 없음'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '시간 정보 없음' : date.toLocaleString('ko-KR')
}

function loadStoredOrder() {
  try {
    const raw = localStorage.getItem(ORDER_STORAGE_KEY)
    const parsed = JSON.parse(raw || '[]')
    return Array.isArray(parsed) ? parsed.map(String) : []
  } catch {
    return []
  }
}

function saveStoredOrder(ids) {
  localStorage.setItem(ORDER_STORAGE_KEY, JSON.stringify(ids))
}

function sortProfilesByOrder(list, orderIds) {
  const map = new Map(list.map(profile => [String(profile.id), profile]))
  const ordered = []
  orderIds.forEach(id => {
    const profile = map.get(String(id))
    if (!profile) return
    ordered.push(profile)
    map.delete(String(id))
  })
  const rest = [...map.values(), ...ordered]
  const active = rest.filter(profile => profile.is_active)
  const inactive = rest.filter(profile => !profile.is_active)
  return [...active, ...inactive]
}

function moveIdBefore(orderIds, fromId, toId) {
  const next = orderIds.filter(id => id !== fromId)
  const toIndex = next.indexOf(toId)
  if (toIndex < 0) return orderIds
  next.splice(toIndex, 0, fromId)
  return next
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

const PREPROCESS_PARALLEL = [
  { key: 'video', label: '비디오 처리', desc: '슬라이드·장면 추출' },
  { key: 'audio', label: '오디오 처리', desc: '음성 품질·전사' },
]

const PREPROCESS_MERGE = {
  key: 'merge',
  label: '통합 텍스트',
  desc: '강의 맥락 구성',
}

function buildDetailFlow(llmRows = []) {
  return {
    parallel: PREPROCESS_PARALLEL.map(step => ({
      stageKey: step.key,
      label: step.label,
      desc: step.desc,
      kind: 'preprocess',
      models: [],
    })),
    merge: {
      stageKey: PREPROCESS_MERGE.key,
      label: PREPROCESS_MERGE.label,
      desc: PREPROCESS_MERGE.desc,
      kind: 'preprocess',
      models: [],
    },
    llm: llmRows.map(row => ({
      stageKey: row.stageKey,
      label: row.label,
      desc: row.mode === 'multi' ? '멀티 LLM' : '메인 LLM',
      kind: 'llm',
      models: Array.isArray(row.models) ? row.models : [],
    })),
  }
}

function PipeCard({ node, badge, tone = 0 }) {
  return (
    <article className={`ms-pipe-card ms-pipe-card--${node.kind} ms-pipe-tone-${tone % 6}`}>
      <div className="ms-pipe-card-top">
        <span className="ms-pipe-idx">{badge}</span>
        <h4 className="ms-pipe-title">{node.label}</h4>
      </div>
      <p className="ms-pipe-desc">{node.desc}</p>
      {node.kind === 'llm' ? (
        <ul className="ms-pipe-models">
          {node.models.length ? node.models.map(model => (
            <li key={`${node.stageKey}-${model.version}-${model.providerType}`}>
              {formatModelLabel(model)}
            </li>
          )) : (
            <li className="ms-pipe-models-empty">미설정</li>
          )}
        </ul>
      ) : (
        <span className="ms-pipe-tag">고정 단계</span>
      )}
    </article>
  )
}

function PipeArrow() {
  return (
    <div className="ms-pipe-arrow" aria-hidden="true">
      <svg viewBox="0 0 32 32" width="28" height="28">
        <path d="M4 16h20" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
        <path d="M18 8l10 8-10 8" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  )
}

function PresetFlow({ flow }) {
  const scrollRef = useRef(null)
  const stageCount = flow.parallel.length + 1 + flow.llm.length

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollLeft = 0
  }, [flow])

  return (
    <div className="ms-pipe">
      <div className="ms-pipe-scroll" ref={scrollRef}>
        <div className="ms-pipe-row">
          <div className="ms-pipe-parallel">
            <span className="ms-pipe-parallel-label">병렬 처리</span>
            {flow.parallel.map((node, index) => (
              <PipeCard
                key={node.stageKey}
                node={node}
                badge={`1${String.fromCharCode(97 + index)}`}
                tone={index}
              />
            ))}
          </div>

          <PipeArrow />

          <PipeCard node={flow.merge} badge="2" tone={2} />

          {flow.llm.map((node, index) => (
            <div className="ms-pipe-follow" key={node.stageKey}>
              <PipeArrow />
              <PipeCard node={node} badge={String(index + 3)} tone={index + 3} />
            </div>
          ))}
        </div>
      </div>

      <p className="ms-pipe-note">
        비디오·오디오는 동시에 처리된 뒤 통합 텍스트로 합쳐집니다. · 총 {stageCount}단계
      </p>
    </div>
  )
}

export default function ModelSetupPresetListPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState(null)
  const [detailId, setDetailId] = useState(null)
  const [orderIds, setOrderIds] = useState(loadStoredOrder)
  const [dragId, setDragId] = useState(null)
  const [overId, setOverId] = useState(null)
  const suppressClickRef = useRef(false)

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
      setOrderIds(current => {
        const next = current.filter(id => id !== String(profileId))
        saveStoredOrder(next)
        return next
      })
      setDetailId(current => (current === profileId ? null : current))
      setSelectedId(current => (current === profileId ? null : current))
      queryClient.invalidateQueries({ queryKey: ['model-setting-profiles'] })
    },
  })

  const allProfiles = data?.profiles || []

  useEffect(() => {
    if (!allProfiles.length) return
    setOrderIds(current => {
      const existing = new Set(allProfiles.map(profile => String(profile.id)))
      const kept = current.filter(id => existing.has(id))
      const missing = allProfiles
        .map(profile => String(profile.id))
        .filter(id => !kept.includes(id))
      const next = [...missing, ...kept]
      if (next.join(',') === current.join(',')) return current
      saveStoredOrder(next)
      return next
    })
  }, [allProfiles])

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

  const profiles = useMemo(() => {
    const sorted = sortProfilesByOrder(allProfiles, orderIds)
    const keyword = search.trim().toLowerCase()
    if (!keyword) return sorted
    return sorted.filter(profile => profile.name.toLowerCase().includes(keyword))
  }, [allProfiles, orderIds, search])

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

  const selectedProfile = useMemo(
    () => profiles.find(profile => profile.id === selectedId) || null,
    [profiles, selectedId],
  )

  const reorderProfiles = (fromId, toId) => {
    if (!fromId || !toId || fromId === toId) return
    setOrderIds(current => {
      const base = current.length
        ? current
        : allProfiles.map(profile => String(profile.id))
      const next = moveIdBefore(base, String(fromId), String(toId))
      saveStoredOrder(next)
      return next
    })
  }

  const selectProfile = profileId => {
    if (suppressClickRef.current) return
    setSelectedId(profileId)
  }

  const openDetail = profileId => {
    setDetailId(profileId)
  }

  const handleConfirmApply = () => {
    if (!selectedId || applyMutation.isPending) return
    applyMutation.mutate(selectedId)
  }

  return (
    <section className="model-setup">
      <div className="ms-header-row">
        <h2 className="ms-app-title">LLM 설정 목록</h2>
        <button className="ms-link-btn ms-link-btn--compact" type="button" onClick={() => navigate('/model-setup')}>
          선택 화면으로
        </button>
      </div>

      <div className="ms-card">
        <div className="ms-preset-toolbar">
          <input
            className="ms-name-input"
            type="text"
            value={search}
            placeholder="셋 이름 검색"
            onChange={event => setSearch(event.target.value)}
          />
          <button className="ms-btn-primary ms-preset-toolbar-btn" type="button" onClick={() => navigate('/model-setup/sets/new')}>
            셋 만들기
          </button>
        </div>
        <p className="ms-preset-hint">드래그로 순서를 바꿀 수 있어요. 항목을 누르면 선택되고, 상단에서 확정하면 적용돼요.</p>
      </div>

      {selectedProfile && (
        <div className="ms-preset-selection-bar">
          <p>
            <strong>{selectedProfile.name}</strong> 선택되었습니다.
            {selectedProfile.is_active && <span className="ms-preset-selection-note"> (현재 적용 중)</span>}
          </p>
          <button
            className="ms-btn-primary ms-preset-confirm-btn"
            type="button"
            disabled={applyMutation.isPending || selectedProfile.is_active}
            onClick={handleConfirmApply}
          >
            {selectedProfile.is_active ? '적용됨' : applyMutation.isPending ? '적용 중…' : '확정'}
          </button>
        </div>
      )}

      {isLoading && <div className="ms-card"><p className="ms-empty">셋을 불러오는 중이에요…</p></div>}
      {error && <div className="ms-card"><p className="ms-save-error">{String(error.message || error)}</p></div>}

      {!isLoading && !error && (
        <div className="ms-preset-list">
          {profiles.length ? profiles.map(profile => {
            const summary = summarizeSetConfig(profile.editor_state || {})
            const isSelected = selectedId === profile.id
            const isDragging = dragId === String(profile.id)
            const isOver = overId === String(profile.id) && dragId && dragId !== String(profile.id)

            return (
              <article
                className={[
                  'ms-preset-card',
                  'ms-preset-card--row',
                  profile.is_active ? 'ms-preset-card--active' : '',
                  isSelected ? 'ms-preset-card--selected' : '',
                  isDragging ? 'ms-preset-card--dragging' : '',
                  isOver ? 'ms-preset-card--drag-over' : '',
                ].filter(Boolean).join(' ')}
                key={profile.id}
                draggable
                onDragStart={event => {
                  if (event.target.closest('.ms-preset-actions, .ms-preset-row-delete, .ms-preset-icon-btn')) {
                    event.preventDefault()
                    return
                  }
                  const id = String(profile.id)
                  suppressClickRef.current = false
                  setDragId(id)
                  event.dataTransfer.effectAllowed = 'move'
                  event.dataTransfer.setData('text/plain', id)
                }}
                onDrag={() => {
                  suppressClickRef.current = true
                }}
                onDragEnd={() => {
                  setDragId(null)
                  setOverId(null)
                  window.setTimeout(() => {
                    suppressClickRef.current = false
                  }, 120)
                }}
                onDragOver={event => {
                  event.preventDefault()
                  event.dataTransfer.dropEffect = 'move'
                  setOverId(String(profile.id))
                }}
                onDragLeave={() => {
                  setOverId(current => (current === String(profile.id) ? null : current))
                }}
                onDrop={event => {
                  event.preventDefault()
                  const fromId = event.dataTransfer.getData('text/plain') || dragId
                  reorderProfiles(fromId, profile.id)
                  setDragId(null)
                  setOverId(null)
                }}
                onClick={event => {
                  if (event.target.closest('.ms-preset-actions, .ms-preset-row-delete, .ms-preset-icon-btn')) return
                  selectProfile(profile.id)
                }}
              >
                <div className="ms-preset-summary ms-preset-summary--row">
                  <div className="ms-preset-title-row">
                    <h3>{profile.name}</h3>
                    {profile.is_active && <span className="ms-preset-badge">적용 중</span>}
                    {isSelected && !profile.is_active && (
                      <span className="ms-preset-badge ms-preset-badge--apply">적용</span>
                    )}
                  </div>
                  <p className="ms-preset-row-meta">
                    {summary.modelCount}개 LLM
                    {' · '}
                    {formatUpdatedAt(profile.updated_at)}
                  </p>
                  <div className="ms-preset-tags">
                    <span className="ms-preset-tag ms-preset-tag--meta">메인 {summary.mainModelName}</span>
                    <span className="ms-preset-tag ms-preset-tag--meta">
                      그라운딩 {summary.includeGrounding ? '포함' : '미포함'}
                    </span>
                    {summary.modelNames.length ? summary.modelNames.map((modelName, index) => (
                      <span className={`ms-preset-tag ms-preset-tag--tone-${index % 5}`} key={modelName}>{modelName}</span>
                    )) : (
                      <span className="ms-preset-tag ms-preset-tag--empty">모델 미설정</span>
                    )}
                  </div>
                </div>

                <div className="ms-preset-actions ms-preset-actions--row">
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
                    onClick={() => navigate(`/model-setup/sets/${profile.id}/edit`)}
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
            <div className="ms-card ms-preset-empty"><p className="ms-empty">검색 결과가 없어요.</p></div>
          )}
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
                  {detailProfile.is_active && <span className="ms-preset-badge">적용 중</span>}
                </div>
                <p>
                  {formatUpdatedAt(detailProfile.updated_at)} 업데이트
                  {' · '}
                  {detailSummary?.modelCount || 0}개 LLM
                </p>
              </div>
              <button className="ms-preset-detail-close" type="button" onClick={() => setDetailId(null)} aria-label="닫기">
                ×
              </button>
            </div>

            <div className="ms-preset-tags ms-preset-tags--detail">
              <span className="ms-preset-tag ms-preset-tag--meta">메인 {detailSummary?.mainModelName}</span>
              <span className="ms-preset-tag ms-preset-tag--meta">
                그라운딩 {detailSummary?.includeGrounding ? '포함' : '미포함'}
              </span>
            </div>

            <div className="ms-preset-flow-head">
              <strong>전체 파이프라인</strong>
              <span>
                {(detailFlow?.parallel.length || 0) + 1 + (detailFlow?.llm.length || 0)}단계
              </span>
            </div>
            {detailFlow && <PresetFlow flow={detailFlow} />}

            <div className="ms-preset-detail-actions">
              <button
                className="ms-btn-secondary"
                type="button"
                onClick={() => navigate(`/model-setup/sets/${detailProfile.id}/edit`)}
              >
                수정
              </button>
              <button className="ms-btn-secondary" type="button" onClick={() => setDetailId(null)}>
                닫기
              </button>
            </div>
          </div>
        </div>,
        document.body,
      )}
    </section>
  )
}
