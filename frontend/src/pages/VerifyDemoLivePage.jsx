import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import DiagramPipeline from '../components/verifier/DiagramPipeline'
import VerifierResults from '../components/verifier/VerifierResults'
import { DEMO_PHASES } from '../hooks/useDemoPipelineFlow'
import { useTimedDiagramFlow } from '../hooks/useTimedDiagramFlow'
import { useVideoThumbnail } from '../hooks/useVideoThumbnail'
import { getLectureResult } from '../api/pipeline'
import { NODE_BY_ID, NODE_SCHEDULE, bigPhaseFor, formatDuration, laneToneFor } from '../components/verifier/diagramPipelineConstants'

// 실제 흐름(/upload → /verify/:id → /result/:id)과 화면·문구가 동일하게 보이도록 만든
// 발표용 데모. 실제 백엔드 파이프라인을 타지 않고도 어떤 영상을 넣어도 다이어그램
// 파이프라인이 자동으로 진행된 뒤 결과 화면을 보여준다("데모"라는 표시를 화면에 노출하지
// 않는다). 개발자만 직접 URL로 들어가서 확인하는 용도라 메인 페이지 등 어디에도 링크하지
// 않는다.
// 방향키로 넘기던 방식 대신, diagramPipelineConstants의 NODE_SCHEDULE(노드별 시작·소요
// 시간)을 기준으로 자동 재생된다. 화면의 스톱워치는 그냥 경과 시간을 보여줄 뿐 목표
// 총 시간은 없고, 마지막 노드(피드백)까지 끝나면 자동으로 멈춘다.
// 완료 화면에 보여줄 결과물은 아직 실제 파이프라인이 없어서, 이미 검증이 끝난 강의 하나의
// 실제 결과를 그대로 재사용한다. 매번 /api/lectures/{id}/result를 호출하므로 그 강의의
// 결과가 바뀌면 데모에도 그대로 반영되고, 다른 강의로 바꾸고 싶을 땐 이 상수만 바꾸면 된다.
const DEMO_RESULT_LECTURE_ID = '9018dee3-a130-4c0e-a4a2-45caaf8c4136'

// 통합 텍스트·피드백은 "무언가를 만들어내는" 단계라 다른 단계와 달리 "진행 중" 대신
// "생성 중"으로 표현한다.
const GENERATION_STAGE_LABELS = new Set(['멀티모달 통합 텍스트', '피드백'])

// 위쪽엔 "큰 분류: 단계명 진행중" 텍스트 한 줄(+스톱워치)만 있고, 그 아래 구분선을 두고
// 활성 노드 수만큼(보통 1~2개, 한 단계만 진행 중이면 1개) 진행 바가 세로로 쌓인다. 바
// 앞엔 어떤 단계의 바인지 짧은 라벨을 붙인다 — CSS grid(2열)로 배치해서 라벨 길이가
// 서로 달라도(예: "슬라이드 추출" vs "오디오 품질 분석") 바들은 항상 같은 x에서 시작한다.
// 각 노드는 NODE_SCHEDULE에 정해진 자기 소요 시간만큼 채워지므로, 같은 구간에서 나란히
// 도는 두 바도 서로 다른 속도로 끝날 수 있다. key를 노드 id로 주면 다음 노드로 바뀔 때
// 컴포넌트가 새로 마운트되어 0%부터 다시 채워진다.
function StageBar({ id }) {
  const [filled, setFilled] = useState(false)
  useEffect(() => {
    const raf = requestAnimationFrame(() => setFilled(true))
    return () => cancelAnimationFrame(raf)
  }, [])
  const durationMs = NODE_SCHEDULE[id]?.duration ?? 1000
  return (
    <>
      <span className="vf-stage-bar-label">{NODE_BY_ID[id]?.label}</span>
      <div className="vf-stage-bar-track">
        <div
          className={`vf-stage-bar-fill vf-stage-bar-fill--${laneToneFor(id)}`}
          style={{ width: filled ? '100%' : '0%', transitionDuration: `${durationMs}ms` }}
        />
      </div>
    </>
  )
}

function PipelineStageBars({ phase, activeIds }) {
  if (phase === DEMO_PHASES.DONE) {
    return (
      <div className="vf-stage-bars">
        <span className="vf-stage-bar-label">완료</span>
        <div className="vf-stage-bar-track">
          <div className="vf-stage-bar-fill vf-stage-bar-fill--done" style={{ width: '100%' }} />
        </div>
      </div>
    )
  }
  return (
    <div className="vf-stage-bars">
      {activeIds.map(id => <StageBar key={id} id={id} />)}
    </div>
  )
}

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
  const { title, file, phase, status, activeIds, activeLabels, elapsedMs, actions } = flow
  const isDone = phase === DEMO_PHASES.DONE
  const stageMessage = isDone
    ? '모든 단계가 완료되었습니다.'
    : activeLabels.length === 1 && GENERATION_STAGE_LABELS.has(activeLabels[0])
      ? `${activeLabels[0]} 생성중`
      : `${activeLabels.join(' · ')} 진행중`
  // 진행 중인 노드가 전처리 쪽인지 검증 쪽인지로 큰 분류를 정하고, 완료 후에는
  // 마지막 구간(검증)의 분류를 그대로 보여준다.
  const bigPhase = bigPhaseFor(activeIds[0] || 'error_output')

  return (
    <div className="detail">
      <div className="page-header-row demo-standard-width">
        <h2 className="list-heading">{title || file?.name || '강의'}</h2>
        <button className="ms-back-btn" type="button" onClick={actions.backToUpload} aria-label="이전으로">
          ←
        </button>
      </div>

      <div className="vf-pipe demo-standard-width">
        <div className="vf-progress-row">
          <div className="vf-progress-message"><strong>{bigPhase}</strong>: {stageMessage}</div>
          <span className="vf-stopwatch">{formatDuration(elapsedMs)}</span>
        </div>
        <PipelineStageBars phase={phase} activeIds={activeIds} />
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
  const flow = useTimedDiagramFlow()
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
