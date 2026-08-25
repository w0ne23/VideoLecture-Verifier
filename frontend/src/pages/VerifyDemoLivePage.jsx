import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import DiagramPipeline from '../components/verifier/DiagramPipeline'
import VerifierResults from '../components/verifier/VerifierResults'
import { DEMO_PHASES } from '../hooks/useDemoPipelineFlow'
import { useDemoDiagramFlow } from '../hooks/useDemoDiagramFlow'
import { useVideoThumbnail } from '../hooks/useVideoThumbnail'
import { getLectureResult } from '../api/pipeline'

// 실제 흐름(/upload → /verify/:id → /result/:id)과 화면·문구가 동일하게 보이도록 만든
// 데모. 실제 백엔드 파이프라인을 타지 않고도 어떤 영상을 넣어도 다이어그램 파이프라인이
// 시간에 따라 자동으로 진행된 뒤 결과 화면을 보여준다("데모"라는 표시를 화면에 노출하지
// 않는다). 개발자만 직접 URL로 들어가서 확인하는 용도라 메인 페이지 등 어디에도 링크하지
// 않는다.
// 완료 화면에 보여줄 결과물은 아직 실제 파이프라인이 없어서, 이미 검증이 끝난 강의 하나의
// 실제 결과를 그대로 재사용한다. 매번 /api/lectures/{id}/result를 호출하므로 그 강의의
// 결과가 바뀌면 데모에도 그대로 반영되고, 다른 강의로 바꾸고 싶을 땐 이 상수만 바꾸면 된다.
const DEMO_RESULT_LECTURE_ID = '9018dee3-a130-4c0e-a4a2-45caaf8c4136'

// 실제 파이프라인 단계는 몇 초~몇 분씩 걸리므로, 즉시 넘어가는 것보다 한 틱에 3초 정도
// 머무는 편이 진행되는 느낌이 더 자연스럽다.
const TICK_DELAY_MS = 3000

// 통합 텍스트·피드백은 "무언가를 만들어내는" 단계라 다른 단계와 달리 "진행 중" 대신
// "생성 중"으로 표현한다.
const GENERATION_STAGE_LABELS = new Set(['멀티모달 통합 텍스트', '피드백'])

function DemoUploadStep({ flow }) {
  const inputRef = useRef(null)
  const { file, title, actions } = flow
  const thumbnailUrl = useVideoThumbnail(file)

  function onDrop(event) {
    event.preventDefault()
    actions.selectFile(event.dataTransfer.files?.[0])
  }

  return (
    <div className="upload-card">
      <h2>강의 영상 업로드</h2>
      <div
        className={`dropzone${file ? ' dropzone--filled' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={event => event.preventDefault()}
        onDrop={onDrop}
      >
        {file
          ? (
            <div className="dropzone-preview">
              {thumbnailUrl && <img className="dropzone-thumbnail" src={thumbnailUrl} alt="" />}
              <span className="dropzone-filename">{file.name} ({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
            </div>
          )
          : <span>클릭하거나 영상 파일을 끌어다 놓으세요 (.mp4)</span>}
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          hidden
          onChange={event => actions.selectFile(event.target.files?.[0])}
        />
      </div>
      <div className="upload-title-row">
        <label className="field upload-title-field">
          <span>강의 제목</span>
          <input
            type="text"
            value={title}
            placeholder="미입력 시 파일명 사용"
            onChange={event => actions.setTitle(event.target.value)}
          />
        </label>
        <button type="button" className="btn btn--primary upload-title-submit" disabled={!file} onClick={actions.start}>
          검증 시작
        </button>
      </div>
    </div>
  )
}

function DemoPipelineStep({ flow, onViewResult }) {
  const { title, file, phase, status, activeLabels, autoPlay, actions } = flow
  const isDone = phase === DEMO_PHASES.DONE
  const stageMessage = isDone
    ? '모든 단계가 완료되었습니다.'
    : activeLabels.length === 1 && GENERATION_STAGE_LABELS.has(activeLabels[0])
      ? `${activeLabels[0]} 생성 중`
      : `${activeLabels.join(' · ')} 진행 중`

  // 발표용: 자동 진행·버튼 없이 방향키(← 이전 단계 / → 다음 단계)로만 넘긴다.
  useEffect(() => {
    if (autoPlay) actions.toggleAutoPlay()
  }, [autoPlay, actions])

  useEffect(() => {
    if (isDone) return undefined
    function handleKeyDown(event) {
      if (event.key === 'ArrowRight') actions.next()
      if (event.key === 'ArrowLeft') actions.prev()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isDone, actions])

  return (
    <div className="detail">
      <div className="page-header-row demo-standard-width">
        <h2 className="list-heading">{title || file?.name || '강의'}</h2>
        <button className="ms-back-btn" type="button" onClick={actions.backToUpload} aria-label="이전으로">
          ←
        </button>
      </div>

      <div className="vf-pipe demo-standard-width">
        <div className="vf-progress-message vf-progress-message--divided">{stageMessage}</div>
        <DiagramPipeline status={status} diffLine diffNode compact />
        {isDone && (
          <div className="button-row button-row--center">
            <button type="button" className="btn" onClick={actions.start}>다시하기</button>
            <button type="button" className="btn btn--primary" onClick={onViewResult}>피드백 보기</button>
          </div>
        )}
      </div>
    </div>
  )
}

function DemoResultStep({ flow, videoUrl, navigate }) {
  const { title, file } = flow
  const videoRef = useRef(null)
  const [verifier, setVerifier] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [showVideo, setShowVideo] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError('')
    getLectureResult(DEMO_RESULT_LECTURE_ID)
      .then(result => { if (!cancelled) setVerifier(result) })
      .catch(err => { if (!cancelled) setError(String(err?.message || err)) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [])

  function seekTo(seconds) {
    const video = videoRef.current
    if (!video) return
    video.currentTime = seconds
    video.play().catch(() => {})
    video.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  return (
    <div className="detail">
      <div className="detail-head">
        <button type="button" className="btn" onClick={() => navigate('/')}>← 메인으로</button>
        <h2>{title || file?.name || '강의'}</h2>
        <button
          type="button"
          className="btn"
          onClick={() => navigate(`/result/${DEMO_RESULT_LECTURE_ID}/stages`)}
        >
          검증 과정 보기
        </button>
        {videoUrl && (
          <button type="button" className="btn" onClick={() => setShowVideo(v => !v)}>
            {showVideo ? '강의 숨기기' : '강의 같이 보기'}
          </button>
        )}
      </div>

      {videoUrl && showVideo && (
        <video ref={videoRef} className="detail-video" src={videoUrl} controls preload="metadata" />
      )}

      {loading && <p className="list-note">불러오는 중...</p>}
      {error && <p className="error-text">{error}</p>}
      {!loading && !error && <VerifierResults verifier={verifier} onSeek={seekTo} />}
    </div>
  )
}

export default function VerifyDemoLivePage() {
  const navigate = useNavigate()
  const flow = useDemoDiagramFlow(TICK_DELAY_MS)
  const { file, phase } = flow
  // 파이프라인이 다 끝나도 바로 결과 화면으로 넘어가지 않고, "피드백 보기"를 눌러야만
  // 넘어가게 한다. "다시하기"로 reset하면 phase가 UPLOAD로 돌아가면서 이 상태도 초기화된다.
  const [showResult, setShowResult] = useState(false)

  const videoUrl = useMemo(() => (file ? URL.createObjectURL(file) : ''), [file])
  useEffect(() => () => { if (videoUrl) URL.revokeObjectURL(videoUrl) }, [videoUrl])
  useEffect(() => {
    if (phase === DEMO_PHASES.UPLOAD) setShowResult(false)
  }, [phase])

  if (phase === DEMO_PHASES.UPLOAD) {
    return (
      <section className="upload-page">
        <div className="page-header-row">
          <h2 className="list-heading">검증할 강의 업로드</h2>
          <button className="ms-back-btn" type="button" onClick={() => navigate('/')} aria-label="메인으로">
            ←
          </button>
        </div>
        <DemoUploadStep flow={flow} />
      </section>
    )
  }

  if (phase === DEMO_PHASES.PIPELINE || (phase === DEMO_PHASES.DONE && !showResult)) {
    return <DemoPipelineStep flow={flow} onViewResult={() => setShowResult(true)} />
  }

  return <DemoResultStep flow={flow} videoUrl={videoUrl} navigate={navigate} />
}
