import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import DiagramPipeline from '../components/verifier/DiagramPipeline'
import VerifierResults from '../components/verifier/VerifierResults'
import { DEMO_PHASES } from '../hooks/useDemoPipelineFlow'
import { useDemoDiagramFlow } from '../hooks/useDemoDiagramFlow'
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

function DemoUploadStep({ flow }) {
  const inputRef = useRef(null)
  const { file, title, actions } = flow

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
          ? <span>{file.name} ({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
          : <span>클릭하거나 영상 파일을 끌어다 놓으세요 (.mp4)</span>}
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          hidden
          onChange={event => actions.selectFile(event.target.files?.[0])}
        />
      </div>
      <label className="field">
        <span>강의 제목</span>
        <input
          type="text"
          value={title}
          placeholder="미입력 시 파일명 사용"
          onChange={event => actions.setTitle(event.target.value)}
        />
      </label>
      <div className="button-row">
        <button type="button" className="btn btn--primary" disabled={!file} onClick={actions.start}>
          검증 시작
        </button>
      </div>
    </div>
  )
}

function DemoPipelineStep({ flow, navigate }) {
  const { title, file, status, activeLabels } = flow
  const stageMessage = `${activeLabels.join(' · ')} 진행 중`

  return (
    <div className="detail">
      <div className="detail-head">
        <button type="button" className="btn" onClick={() => navigate('/')}>← 메인으로</button>
        <h2>{title || file?.name || '강의'}</h2>
      </div>

      <div className="vf-pipe diag-breakout">
        <div className="vf-progress-message">{stageMessage}</div>
        <DiagramPipeline status={status} diffLine diffNode />
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
      </div>

      {videoUrl && (
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

  const videoUrl = useMemo(() => (file ? URL.createObjectURL(file) : ''), [file])
  useEffect(() => () => { if (videoUrl) URL.revokeObjectURL(videoUrl) }, [videoUrl])

  if (phase === DEMO_PHASES.UPLOAD) {
    return (
      <section className="upload-page">
        <div className="page-header-row">
          <h2 className="list-heading">검증할 강의 업로드</h2>
          <button className="ms-back-btn" type="button" onClick={() => navigate('/')} aria-label="메인으로">
            ←
          </button>
        </div>
        <p className="upload-page-hint">강의 영상을 업로드하면 검증이 시작됩니다.</p>
        <DemoUploadStep flow={flow} />
      </section>
    )
  }

  if (phase === DEMO_PHASES.PIPELINE) {
    return <DemoPipelineStep flow={flow} navigate={navigate} />
  }

  return <DemoResultStep flow={flow} videoUrl={videoUrl} navigate={navigate} />
}
