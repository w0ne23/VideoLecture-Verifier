import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { listModelSettingProfiles } from '../../api/modelSetupProfiles'
import { summarizeEditorState } from './stageModels'

function formatModelText(row) {
  if (!row.models.length) return '미설정'
  return row.models.map(model => model.version || model.modelId || '모델').join(', ')
}

export default function ModelSetupFab() {
  const [open, setOpen] = useState(false)
  const [showTip, setShowTip] = useState(true)
  const { data } = useQuery({
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

  const handleFabClick = () => {
    setShowTip(false)
    setOpen(current => !current)
  }

  return (
    <div className={`ms-fab-wrap${open ? ' ms-fab-wrap--open' : ''}`}>
      <div className="ms-fab-panel" aria-hidden={!open}>
        <div className="ms-fab-panel-inner">
          <div className="ms-fab-header">
            <div>
              <p className="ms-fab-kicker">현재 적용 LLM 셋</p>
              <h3>{activeProfile?.name || '아직 적용된 셋 없음'}</h3>
            </div>
            {activeProfile && (
              <span className="ms-fab-stage-count">{rows.length}단계</span>
            )}
          </div>
          {activeProfile ? (
            <div className="ms-fab-stage-list">
              {rows.map((row, index) => (
                <div
                  className={`ms-fab-stage-row ms-fab-stage-row--tone-${index % 6}`}
                  key={row.stageKey}
                >
                  <span className="ms-fab-stage-num">{index + 1}단계</span>
                  <strong className="ms-fab-stage-label">{row.label}</strong>
                  <span className="ms-fab-stage-models">
                    {formatModelText(row)}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="ms-empty">적용된 셋이 아직 없어요.</p>
          )}
        </div>
      </div>

      <div className="ms-fab-dock">
        {showTip && (
          <div className="ms-fab-tip" role="status">
            <p>현재 설정을 보고싶다면 클릭하세요!</p>
          </div>
        )}
        <button
          className="ms-fab-button"
          type="button"
          aria-expanded={open}
          onClick={handleFabClick}
        >
          <span className="ms-fab-icon-grid" aria-hidden="true">
            <span />
            <span />
            <span />
            <span />
          </span>
        </button>
      </div>
    </div>
  )
}
