import { useRef } from 'react'
import { PHASES } from './verifier/verifierConstants'
import PipelineProgress from './verifier/PipelineProgress'
import VerifierResults from './verifier/VerifierResults'
import { useJobStream } from '../hooks/useJobStream'

export default function LectureDetail({ lectureId, onExit }) {
  const videoRef = useRef(null)
  const {
    phase,
    lecture,
    verifier,
    pipelineStages,
    currentStage,
    errorMessage,
    isLoading,
    isMutating,
    isVerified,
    actions,
  } = useJobStream(lectureId, { onExit })

  function seekTo(seconds) {
    const video = videoRef.current
    if (!video) return
    video.currentTime = seconds
    video.play().catch(() => {})
    video.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  if (isLoading) return <p className="list-note">불러오는 중...</p>

  return (
    <div className="detail">
      <div className="detail-head">
        <button type="button" className="btn" onClick={onExit}>← 목록으로</button>
        <h2>{lecture.title || lectureId}</h2>
        {isVerified && <span className="lecture-verified">검토 완료</span>}
      </div>

      {lecture.video_url && (
        <video ref={videoRef} className="detail-video" src={lecture.video_url} controls preload="metadata" />
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

      {phase === PHASES.VERIFY_READY && (
        <VerifierResults verifier={verifier} onSeek={seekTo} />
      )}

      {phase === PHASES.VERIFY_READY && !verifier && (
        <p className="list-note">검증 결과 파일을 불러올 수 없습니다. 파이프라인 로그를 확인하세요.</p>
      )}

      {errorMessage && phase !== PHASES.ERROR && <p className="error-text">{errorMessage}</p>}

      <div className="button-row detail-actions">
        {phase === PHASES.VERIFY_READY && !isVerified && (
          <button type="button" className="btn btn--primary" disabled={isMutating} onClick={actions.confirmReview}>
            검토 완료 처리
          </button>
        )}
        {(phase === PHASES.VERIFY_READY || phase === PHASES.ERROR) && (
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
