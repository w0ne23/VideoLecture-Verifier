import { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  applyModelSettingProfile,
  deleteModelSettingProfile,
  listModelSettingProfiles,
} from '../../api/modelSetupProfiles'
import { summarizeEditorState } from '../../components/model-setup/stageModels'

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
  return [...map.values(), ...ordered]
}

function moveIdBefore(orderIds, fromId, toId) {
  const next = orderIds.filter(id => id !== fromId)
  const toIndex = next.indexOf(toId)
  if (toIndex < 0) return orderIds
  next.splice(toIndex, 0, fromId)
  return next
}

function formatModelLabel(row, model) {
  if (row.mode === 'multi') return `${model.version} (${model.weight || 0}%)`
  return model.version
}

function collectUsedModels(rows) {
  const seen = new Set()
  const names = []
  rows.forEach(row => {
    row.models.forEach(model => {
      const name = String(model.version || '').trim()
      if (!name || seen.has(name)) return
      seen.add(name)
      names.push(name)
    })
  })
  return names
}

function PresetFlow({ rows }) {
  const rowChunks = []
  for (let index = 0; index < rows.length; index += 3) {
    rowChunks.push(rows.slice(index, index + 3))
  }

  return (
    <div className="ms-preset-flow">
      {rowChunks.map((chunk, rowIndex) => (
        <div className="ms-preset-flow-row" key={`flow-row-${rowIndex}`}>
          {chunk.map((row, chunkIndex) => {
            const stageIndex = rowIndex * 3 + chunkIndex
            return (
              <div
                className="ms-preset-flow-part"
                key={row.stageKey}
                style={{ '--ms-stage-delay': `${80 + stageIndex * 90}ms` }}
              >
                <div className={`ms-preset-flow-stage${row.models.length ? ' ms-preset-flow-stage--set' : ''} ms-preset-flow-stage--tone-${stageIndex % 6}`}>
                  <div className="ms-preset-flow-stage-top">
                    <span className="ms-preset-flow-num">{stageIndex + 1}단계</span>
                    <div>
                      <strong>{row.label}</strong>
                      <p>{row.mode === 'multi' ? '멀티' : '싱글'} · 재시도 {row.retryCount}회</p>
                    </div>
                  </div>
                  <div className="ms-preset-flow-models">
                    {row.models.length ? row.models.map(model => (
                      <span className="ms-preset-flow-chip" key={`${row.stageKey}-${model.version}-${model.providerType}`}>
                        {formatModelLabel(row, model)}
                      </span>
                    )) : (
                      <span className="ms-preset-flow-empty">미설정</span>
                    )}
                  </div>
                </div>
                {chunkIndex < chunk.length - 1 && (
                  <div
                    className="ms-preset-flow-arrow"
                    aria-hidden="true"
                    style={{ '--ms-stage-delay': `${140 + stageIndex * 90}ms` }}
                  >
                    <svg viewBox="0 0 24 24" width="16" height="16">
                      <path d="M5 12h12M13 6l6 6-6 6" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      ))}
    </div>
  )
}

export default function ModelSetupPresetListPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [search, setSearch] = useState('')
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
  const detailRows = useMemo(
    () => (detailProfile ? summarizeEditorState(detailProfile.editor_state || {}) : []),
    [detailProfile],
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

  const openDetail = profileId => {
    if (suppressClickRef.current) return
    setDetailId(profileId)
  }

  return (
    <section className="model-setup">
      <div className="ms-header-row">
        <h2 className="ms-app-title">프리셋 목록</h2>
        <button className="ms-link-btn ms-link-btn--compact" type="button" onClick={() => navigate('/model-setup')}>
          선택 화면으로 -&gt;
        </button>
      </div>

      <div className="ms-card">
        <div className="ms-preset-toolbar">
          <input
            className="ms-name-input"
            type="text"
            value={search}
            placeholder="프리셋 이름 검색"
            onChange={event => setSearch(event.target.value)}
          />
          <button className="ms-btn-primary ms-preset-toolbar-btn" type="button" onClick={() => navigate('/model-setup/new')}>
            새로 만들기
          </button>
        </div>
        <p className="ms-preset-hint">카드를 드래그해 순서를 바꾸고, 클릭하면 상세를 볼 수 있어요.</p>
      </div>

      {isLoading && <div className="ms-card"><p className="ms-empty">프리셋을 불러오는 중이에요…</p></div>}
      {error && <div className="ms-card"><p className="ms-save-error">{String(error.message || error)}</p></div>}

      {!isLoading && !error && (
        <div className="ms-preset-list">
          {profiles.length ? profiles.map(profile => {
            const rows = summarizeEditorState(profile.editor_state || {})
            const usedModels = collectUsedModels(rows)
            const isDragging = dragId === String(profile.id)
            const isOver = overId === String(profile.id) && dragId && dragId !== String(profile.id)

            return (
              <article
                className={[
                  'ms-preset-card',
                  profile.is_active ? 'ms-preset-card--active' : '',
                  isDragging ? 'ms-preset-card--dragging' : '',
                  isOver ? 'ms-preset-card--drag-over' : '',
                ].filter(Boolean).join(' ')}
                key={profile.id}
                draggable
                onDragStart={event => {
                  if (event.target.closest('.ms-preset-actions')) {
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
                  if (event.target.closest('.ms-preset-actions')) return
                  openDetail(profile.id)
                }}
              >
                <div className="ms-preset-summary">
                  <div className="ms-preset-summary-head">
                    <div>
                      <div className="ms-preset-title-row">
                        <h3>{profile.name}</h3>
                        {profile.is_active && <span className="ms-preset-badge">적용 중</span>}
                      </div>
                      <p>{profile.is_active ? '지금 검증에 쓰이는 설정이에요' : '저장된 프리셋'}</p>
                    </div>
                    <span className="ms-preset-date">{formatUpdatedAt(profile.updated_at)}</span>
                  </div>
                  <div className="ms-preset-tags">
                    <span className="ms-preset-tag ms-preset-tag--stage">{rows.length}단계</span>
                    {usedModels.length ? usedModels.map((name, index) => (
                      <span className={`ms-preset-tag ms-preset-tag--tone-${index % 5}`} key={name}>{name}</span>
                    )) : (
                      <span className="ms-preset-tag ms-preset-tag--empty">모델 미설정</span>
                    )}
                  </div>
                  <p className="ms-preset-more">클릭하여 자세히 보기</p>
                </div>

                <div className="ms-preset-actions">
                  <button
                    className="ms-btn-primary"
                    type="button"
                    disabled={applyMutation.isPending || profile.is_active}
                    onClick={() => applyMutation.mutate(profile.id)}
                  >
                    {profile.is_active ? '적용됨' : '적용'}
                  </button>
                  <button
                    className="ms-btn-secondary"
                    type="button"
                    onClick={() => navigate(`/model-setup/presets/${profile.id}/edit`)}
                  >
                    수정
                  </button>
                  <button
                    className="ms-btn-secondary"
                    type="button"
                    disabled={deleteMutation.isPending}
                    onClick={() => {
                      if (window.confirm(`"${profile.name}" 프리셋을 삭제할까요?`)) {
                        deleteMutation.mutate(profile.id)
                      }
                    }}
                  >
                    삭제
                  </button>
                </div>
              </article>
            )
          }) : (
            <div className="ms-card ms-preset-empty"><p className="ms-empty">검색 결과가 없어요.</p></div>
          )}
        </div>
      )}

      {detailProfile && (
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
                <p>{formatUpdatedAt(detailProfile.updated_at)} 업데이트 · {detailRows.length}단계 파이프라인</p>
              </div>
              <button className="ms-preset-detail-close" type="button" onClick={() => setDetailId(null)} aria-label="닫기">
                ×
              </button>
            </div>

            <div className="ms-preset-flow-head">
              <strong>파이프라인 구성</strong>
              <span>{detailRows.length}단계</span>
            </div>
            <PresetFlow rows={detailRows} />

            <div className="ms-preset-detail-actions">
              <button
                className="ms-btn-primary"
                type="button"
                disabled={applyMutation.isPending || detailProfile.is_active}
                onClick={() => applyMutation.mutate(detailProfile.id)}
              >
                {detailProfile.is_active ? '적용됨' : '적용'}
              </button>
              <button
                className="ms-btn-secondary"
                type="button"
                onClick={() => navigate(`/model-setup/presets/${detailProfile.id}/edit`)}
              >
                수정
              </button>
              <button className="ms-btn-secondary" type="button" onClick={() => setDetailId(null)}>
                닫기
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
