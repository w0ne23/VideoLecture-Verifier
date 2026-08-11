import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { listModelSettingProfiles } from '../../api/modelSetupProfiles'
import { summarizeEditorState } from './stageModels'
import { summarizeSetConfig } from './setBuilder'

function formatModelText(row) {
  if (!row.models.length) return '미설정'
  return row.models.map(model => model.version || model.modelId || '모델').join(', ')
}

export default function ActiveLlmSetBanner() {
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const { data, isLoading } = useQuery({
    queryKey: ['model-setting-profiles'],
    queryFn: listModelSettingProfiles,
  })

  const activeProfile = useMemo(() => {
    const profiles = data?.profiles || []
    return profiles.find(profile => profile.is_active) || null
  }, [data])

  const rows = useMemo(
    () => summarizeEditorState(activeProfile?.editor_state || {}),
    [activeProfile],
  )
  const summary = useMemo(
    () => summarizeSetConfig(activeProfile?.editor_state || {}),
    [activeProfile],
  )

  const label = activeProfile?.name || (isLoading ? '불러오는 중…' : '적용된 셋 없음')

  return (
    <>
      <button
        type="button"
        className="main-llm-set-banner"
        onClick={() => setOpen(true)}
        aria-haspopup="dialog"
        aria-label={`현재 적용 LLM 셋: ${label}`}
      >
        <span className="main-llm-set-banner-text">
          <strong>{label}</strong>
        </span>
        {activeProfile && summary.models.length > 0 && (
          <span className="main-llm-set-banner-models">
            {summary.models.map(model => (
              <span key={model.id} className={`ms-preset-tag ms-preset-tag--${model.type}`}>
                {model.isMain && <span aria-hidden="true">★</span>}
                {model.version}
              </span>
            ))}
          </span>
        )}
      </button>

      {open && (
        <div
          className="ms-preset-detail-backdrop"
          role="presentation"
          onClick={() => setOpen(false)}
        >
          <div
            className={`ms-preset-detail-modal${activeProfile ? ' ms-preset-detail-modal--active' : ''} main-llm-set-modal`}
            role="dialog"
            aria-modal="true"
            aria-labelledby="main-llm-set-title"
            onClick={event => event.stopPropagation()}
          >
            <div className="ms-preset-detail-head">
              <div>
                <div className="ms-preset-title-row">
                  <h3 id="main-llm-set-title">{activeProfile?.name || '적용된 LLM 셋 없음'}</h3>
                  {activeProfile && <span className="ms-preset-badge">적용 중</span>}
                </div>
                <p>
                  {activeProfile
                    ? `메인 ${summary.mainModelName} · 그라운딩 ${summary.includeGrounding ? '포함' : '미포함'} · ${rows.length}단계`
                    : 'Multi-LLM 설정에서 셋을 적용해 주세요.'}
                </p>
              </div>
              <button
                className="ms-preset-detail-close"
                type="button"
                onClick={() => setOpen(false)}
                aria-label="닫기"
              >
                ×
              </button>
            </div>

            {activeProfile ? (
              <div className="ms-fab-stage-list main-llm-set-stage-list">
                {rows.map((row, index) => (
                  <div
                    className={`ms-fab-stage-row ms-fab-stage-row--tone-${index % 6}`}
                    key={row.stageKey}
                  >
                    <span className="ms-fab-stage-num">{index + 1}단계</span>
                    <strong className="ms-fab-stage-label">{row.label}</strong>
                    <span className="ms-fab-stage-models">{formatModelText(row)}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="ms-empty">적용된 셋이 아직 없어요.</p>
            )}

            <div className="ms-preset-detail-actions">
              <button
                className="ms-btn-primary"
                type="button"
                onClick={() => {
                  setOpen(false)
                  navigate('/model-setup/sets')
                }}
              >
                설정 변경
              </button>
              <button className="ms-btn-secondary" type="button" onClick={() => setOpen(false)}>
                닫기
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
