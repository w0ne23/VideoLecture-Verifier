import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { PHASES } from '../components/verifier/verifierConstants'
import VerifierResults from '../components/verifier/VerifierResults'
import { useJobStream } from '../hooks/useJobStream'

export default function ResultPage() {
  const { lectureId } = useParams()
  const navigate = useNavigate()
  const videoRef = useRef(null)
  const goToList = () => navigate('/lectures')
  const [showVideo, setShowVideo] = useState(false)

  const {
    phase,
    lecture,
    verifier,
    currentStage,
    errorMessage,
    isLoading,
    isMutating,
    actions,
  } = useJobStream(lectureId, { onExit: goToList })

  // 결과 화면에 아직 파이프라인이 안 끝난 강의로 진입했거나, "다시 검증"으로 재실행된 경우
  // 진행 화면으로 되돌려보낸다.
  // useJobStream의 초기 phase는 PIPELINE이므로, 로딩이 끝난 뒤에만 판정해야
  // /result ↔ /verify 리다이렉트 루프가 생기지 않는다.
  useEffect(() => {
    if (isLoading) return
    if (phase === PHASES.PIPELINE || phase === PHASES.ERROR) {
      navigate(`/verify/${lectureId}`, { replace: true })
    }
  }, [phase, lectureId, navigate, isLoading])

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
        <button type="button" className="btn" onClick={goToList}>← 목록으로</button>
        <h2>{lecture.title || lectureId}</h2>
        {phase === PHASES.VERIFY_READY && (
          <button type="button" className="btn" onClick={() => navigate(`/result/${lectureId}/stages`)}>
            검증 과정 보기
          </button>
        )}
        {phase === PHASES.VERIFY_READY && lecture.video_url && (
          <button type="button" className="btn" onClick={() => setShowVideo(v => !v)}>
            {showVideo ? '강의 숨기기' : '강의 같이 보기'}
          </button>
        )}
      </div>

      {lecture.video_url && showVideo && (
        <video ref={videoRef} className="detail-video" src={lecture.video_url} controls preload="metadata" />
      )}

      {phase === PHASES.VERIFY_READY && (
        <VerifierResults verifier={verifier} onSeek={seekTo} />
      )}

      {phase === PHASES.VERIFY_READY && !verifier && (
        <p className="list-note">검증 결과 파일을 불러올 수 없습니다. 파이프라인 로그를 확인하세요.</p>
      )}

      {errorMessage && <p className="error-text">{errorMessage}</p>}

      <div className="button-row detail-actions">
        <button type="button" className="btn" disabled={isMutating} onClick={actions.restart}>
          다시 검증
        </button>
        <button type="button" className="btn btn--danger" disabled={isMutating} onClick={actions.remove}>
          삭제
        </button>
      </div>
    </div>
  )
}
