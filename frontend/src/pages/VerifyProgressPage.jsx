import { useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { PHASES } from '../components/verifier/verifierConstants'
import PipelineProgress from '../components/verifier/PipelineProgress'
import { useJobStream } from '../hooks/useJobStream'

export default function VerifyProgressPage() {
  const { lectureId } = useParams()
  const navigate = useNavigate()
  const goToList = () => navigate('/upload')

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

      {lecture.video_url && (
        <video className="detail-video" src={lecture.video_url} controls preload="metadata" />
      )}

      {phase === PHASES.PIPELINE && (
        <PipelineProgress stages={pipelineStages} statusMessage={currentStage} />
      )}

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
