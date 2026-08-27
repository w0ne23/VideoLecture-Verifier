import { Fragment, useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { PHASES } from '../components/verifier/verifierConstants'
import DiagramPipeline from '../components/verifier/DiagramPipeline'
import {
  NODE_BY_ID,
  NODE_IDS,
  bigPhaseFor,
  formatDuration,
  laneToneFor,
  statusMapFromPipelineStages,
} from '../components/verifier/diagramPipelineConstants'
import { useJobStream } from '../hooks/useJobStream'
import { useElapsedStopwatch } from '../hooks/useElapsedStopwatch'

export default function VerifyProgressPage() {
  const { lectureId } = useParams()
  const navigate = useNavigate()
  const goToList = () => navigate('/lectures')

  const {
    phase,
    lecture,
    pipelineStages,
    currentStage,
    errorMessage,
    isLoading,
    isMutating,
    actions,
  } = useJobStream(lectureId, { onExit: goToList })

  const elapsedMs = useElapsedStopwatch(!isLoading && phase === PHASES.PIPELINE)

  // 파이프라인이 완료되면 결과 화면으로 이동한다. 이미 완료된 강의로 바로 진입한 경우도
  // 여기서 즉시 리다이렉트되므로, 목록에서는 상태와 무관하게 항상 이 라우트로 보내면 된다.
  useEffect(() => {
    if (phase === PHASES.VERIFY_READY) {
      navigate(`/result/${lectureId}`, { replace: true })
    }
  }, [phase, lectureId, navigate])

  if (isLoading) return <p className="list-note">불러오는 중...</p>

  return (
    <div className="detail">
      <div className="detail-head">
        <button type="button" className="btn" onClick={goToList}>← 목록으로</button>
        <h2>{lecture.title || lectureId}</h2>
      </div>

      {phase === PHASES.PIPELINE && (() => {
        const diagramStatus = statusMapFromPipelineStages(pipelineStages)
        const activeIds = NODE_IDS.filter(id => diagramStatus[id] === 'run')
        const bigPhase = bigPhaseFor(activeIds[0] || 'error_output')
        const stageMessage = String(currentStage || '').trim() || '분석 준비 중...'
        return (
          <div className="vf-pipe">
            <div className="vf-progress-row">
              <div className="vf-progress-message"><strong>{bigPhase}</strong>: {stageMessage}</div>
              <span className="vf-stopwatch">{formatDuration(elapsedMs)}</span>
            </div>
            {activeIds.length > 0 && (
              <div className="vf-stage-bars">
                {activeIds.map(id => (
                  <Fragment key={id}>
                    <span className="vf-stage-bar-label">{NODE_BY_ID[id]?.label}</span>
                    <div className="vf-stage-bar-track">
                      <div className={`vf-stage-bar-fill vf-stage-bar-fill--indeterminate vf-stage-bar-fill--${laneToneFor(id)}`} />
                    </div>
                  </Fragment>
                ))}
              </div>
            )}
            <DiagramPipeline status={diagramStatus} diffLine diffNode />
          </div>
        )
      })()}

      {phase === PHASES.ERROR && (
        <div className="error-box">
          <h3>분석 실패</h3>
          <p>{errorMessage || currentStage || '알 수 없는 오류가 발생했습니다.'}</p>
        </div>
      )}

      {errorMessage && phase !== PHASES.ERROR && <p className="error-text">{errorMessage}</p>}

      <div className="button-row detail-actions">
        {phase === PHASES.ERROR && (
          <button type="button" className="btn" disabled={isMutating} onClick={actions.restart}>
            다시 검증
          </button>
        )}
        <button type="button" className="btn btn--danger" disabled={isMutating} onClick={actions.remove}>
          삭제
        </button>
      </div>
    </div>
  )
}
