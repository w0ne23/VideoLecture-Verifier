// [개발용] 아키텍처 다이어그램(전처리/검증 두 구간, 구간마다 두 갈래) 확인/수정 데모 (/dev/verify-demo-diagram)
// 실제 API 없이 로컬 타이머로만 진행, /dev/verify-demo 와 별개

import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import DiagramPipeline from '../components/verifier/DiagramPipeline'
import { DEMO_PHASES } from '../hooks/useDemoPipelineFlow'
import { useDemoDiagramFlow } from '../hooks/useDemoDiagramFlow'

// 발화 검증(rose)·슬라이드 검증(teal) 구간의 선/노드 색을 켜고 끄며 비교하는 실험용 컨트롤
// 상태를 페이지 최상단에 둬 진행/에러/완료 단계를 오가도 값 유지 — 색은 고정, "선만/노드만 다르게" 토글만 유지
function ColorLabControls({ diffLine, diffNode, onToggleLine, onToggleNode, noGrounding, onToggleGrounding }) {
  return (
    <div className="diag-colorlab">
      <div className="diag-colorlab-row">
        <span className="diag-colorlab-label">발화·슬라이드 색 비교</span>
        <label className="field" style={{ display: 'inline-flex', alignItems: 'center', gap: 6, margin: 0 }}>
          <input type="checkbox" checked={diffLine} onChange={onToggleLine} />
          <span style={{ margin: 0 }}>선 색 다르게</span>
        </label>
        <label className="field" style={{ display: 'inline-flex', alignItems: 'center', gap: 6, margin: 0 }}>
          <input type="checkbox" checked={diffNode} onChange={onToggleNode} />
          <span style={{ margin: 0 }}>노드 색 다르게</span>
        </label>
        <label className="field" style={{ display: 'inline-flex', alignItems: 'center', gap: 6, margin: 0 }}>
          <input type="checkbox" checked={noGrounding} onChange={onToggleGrounding} />
          <span style={{ margin: 0 }}>웹그라운딩 미포함 셋</span>
        </label>
      </div>
    </div>
  )
}

function DemoUploadStep({ flow }) {
  const inputRef = useRef(null)
  const { file, title, actions } = flow

  function onDrop(event) {
    event.preventDefault()
    actions.selectFile(event.dataTransfer.files?.[0])
  }

  return (
    <div className="upload-card">
      <h2>파이프라인 다이어그램 데모 업로드</h2>
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
          검증 시작 (데모)
        </button>
      </div>
    </div>
  )
}

function DemoPipelineStep({ flow, colorLab }) {
  const { phase, title, file, tickIndex, autoPlay, status, currentStage, actions } = flow
  const isError = phase === DEMO_PHASES.ERROR
  // worker.py 가 그라운딩 미포함 잡의 verifier_web_grounding stage 를 시작부터 'skip' 으로
  // 보고하는 신호를 데모에서 흉내냄 — DiagramPipeline 은 이 값만 보고 레이아웃 변경
  const displayStatus = colorLab.noGrounding ? { ...status, issue_filter: 'skip' } : status

  return (
    <div className="detail">
      <div className="detail-head">
        <h2>{title || file?.name || '데모 강의'}</h2>
      </div>

      <div className="vf-pipe diag-breakout">
        <div className="vf-progress-message">{currentStage}</div>
        <ColorLabControls {...colorLab} />
        <DiagramPipeline status={displayStatus} diffLine={colorLab.diffLine} diffNode={colorLab.diffNode} />
      </div>

      {isError && (
        <div className="error-box">
          <h3>분석 실패 (데모)</h3>
          <p>실제 오류가 아닙니다. 실패 상태 스타일을 확인하기 위한 데모 화면입니다.</p>
        </div>
      )}

      <div className="button-row detail-actions">
        <button type="button" className="btn" disabled={!isError && tickIndex === 0} onClick={actions.prev}>
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

function DemoDoneStep({ flow, colorLab }) {
  const { title, file, status, currentStage, actions } = flow
  const displayStatus = colorLab.noGrounding ? { ...status, issue_filter: 'skip' } : status

  return (
    <div className="detail">
      <div className="detail-head">
        <h2>{title || file?.name || '데모 강의'}</h2>
      </div>

      <div className="vf-pipe diag-breakout">
        <div className="vf-progress-message">{currentStage}</div>
        <ColorLabControls {...colorLab} />
        <DiagramPipeline status={displayStatus} diffLine={colorLab.diffLine} diffNode={colorLab.diffNode} />
      </div>

      <div className="button-row detail-actions">
        <button type="button" className="btn btn--primary" onClick={actions.reset}>
          처음부터
        </button>
      </div>
    </div>
  )
}

export default function VerifyDemoDiagramPage() {
  const navigate = useNavigate()
  const flow = useDemoDiagramFlow()
  const [diffLine, setDiffLine] = useState(false)
  const [diffNode, setDiffNode] = useState(false)
  const [noGrounding, setNoGrounding] = useState(false)
  const colorLab = {
    diffLine,
    diffNode,
    noGrounding,
    onToggleLine: () => setDiffLine(value => !value),
    onToggleNode: () => setDiffNode(value => !value),
    onToggleGrounding: () => setNoGrounding(value => !value),
  }

  return (
    <section className="upload-page">
      <div className="page-header-row">
        <h2 className="list-heading">파이프라인 다이어그램 데모</h2>
        <button className="ms-back-btn" type="button" onClick={() => navigate('/')} aria-label="메인으로">
          ←
        </button>
      </div>

      {flow.phase === DEMO_PHASES.UPLOAD && <DemoUploadStep flow={flow} />}
      {(flow.phase === DEMO_PHASES.PIPELINE || flow.phase === DEMO_PHASES.ERROR) && (
        <DemoPipelineStep flow={flow} colorLab={colorLab} />
      )}
      {flow.phase === DEMO_PHASES.DONE && <DemoDoneStep flow={flow} colorLab={colorLab} />}
    </section>
  )
}
