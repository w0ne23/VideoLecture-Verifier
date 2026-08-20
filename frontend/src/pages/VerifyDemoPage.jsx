import { useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import PipelineProgress from '../components/verifier/PipelineProgress'
import DemoStageAnimation from '../components/verifier/DemoStageAnimation'
import { PIPELINE_NODES } from '../components/verifier/verifierConstants'
import { DEMO_PHASES, useDemoPipelineFlow } from '../hooks/useDemoPipelineFlow'

// 파이프라인 진행 UI만 확인/수정하기 위한 데모 라우트.
// 실제 업로드·검증 API를 호출하지 않고 로컬 타이머로만 stage를 넘긴다. 결과 화면은 만들지 않는다.
function DemoUploadStep({ flow }) {
  const inputRef = useRef(null)
  const { file, title, actions } = flow

  function onDrop(event) {
    event.preventDefault()
    actions.selectFile(event.dataTransfer.files?.[0])
  }

  return (
    <div className="upload-card">
      <h2>파이프라인 데모 업로드</h2>
      <p className="list-note">실제로 업로드되지 않습니다. 파일을 고르면 바로 데모 진행 화면으로 넘어갑니다.</p>
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
          검증 시작 (데모)
        </button>
      </div>
    </div>
  )
}

function DemoPipelineStep({ flow }) {
  const { phase, title, file, stageIndex, lastStageIndex, autoPlay, pipelineStages, currentStage, actions } = flow
  const isError = phase === DEMO_PHASES.ERROR

  return (
    <div className="detail">
      <div className="detail-head">
        <h2>{title || file?.name || '데모 강의'}</h2>
      </div>

      <PipelineProgress stages={pipelineStages} statusMessage={currentStage} />

      <DemoStageAnimation stageId={PIPELINE_NODES[stageIndex]?.id} phase={phase} />

      {isError && (
        <div className="error-box">
          <h3>분석 실패 (데모)</h3>
          <p>실제 오류가 아닙니다. 실패 상태 스타일을 확인하기 위한 데모 화면입니다.</p>
        </div>
      )}

      <div className="button-row detail-actions">
        <button type="button" className="btn" disabled={!isError && stageIndex === 0} onClick={actions.prev}>
          {isError ? '오류 복구' : '이전 단계'}
        </button>
        <button type="button" className="btn" disabled={isError} onClick={actions.next}>
          다음 단계
        </button>
        <label className="field" style={{ display: 'inline-flex', alignItems: 'center', gap: 6, margin: 0 }}>
          <input type="checkbox" checked={autoPlay} disabled={isError} onChange={actions.toggleAutoPlay} />
          <span style={{ margin: 0 }}>자동 재생</span>
        </label>
        <button type="button" className="btn btn--danger" disabled={isError} onClick={actions.simulateError}>
          에러 상태로 보기
        </button>
        <button type="button" className="btn" onClick={actions.reset}>
          처음부터
        </button>
      </div>
    </div>
  )
}

function DemoDoneStep({ flow }) {
  const { phase, title, file, pipelineStages, currentStage, actions } = flow

  return (
    <div className="detail">
      <div className="detail-head">
        <h2>{title || file?.name || '데모 강의'}</h2>
      </div>

      <PipelineProgress stages={pipelineStages} statusMessage={currentStage} />

      <DemoStageAnimation stageId={null} phase={phase} />

      <div className="button-row detail-actions">
        <button type="button" className="btn btn--primary" onClick={actions.reset}>
          처음부터
        </button>
      </div>
    </div>
  )
}

export default function VerifyDemoPage() {
  const navigate = useNavigate()
  const flow = useDemoPipelineFlow()

  return (
    <section className="upload-page">
      <div className="page-header-row">
        <h2 className="list-heading">파이프라인 UI 데모</h2>
        <button className="ms-back-btn" type="button" onClick={() => navigate('/')} aria-label="메인으로">
          ←
        </button>
      </div>

      {flow.phase === DEMO_PHASES.UPLOAD && <DemoUploadStep flow={flow} />}
      {(flow.phase === DEMO_PHASES.PIPELINE || flow.phase === DEMO_PHASES.ERROR) && <DemoPipelineStep flow={flow} />}
      {flow.phase === DEMO_PHASES.DONE && <DemoDoneStep flow={flow} />}
    </section>
  )
}
