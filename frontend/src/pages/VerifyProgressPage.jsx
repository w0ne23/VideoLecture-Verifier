// 검증 진행 화면 — 다이어그램(DiagramPipeline) + 진행 중 단계별 진행 바 + 경과 시간
// job 상태는 useJobStream(SSE), 완료되면 결과 화면으로 리다이렉트

import { Fragment, useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { PHASES } from '../components/verifier/verifierConstants'
import DiagramPipeline from '../components/verifier/DiagramPipeline'
import {
  NODE_BY_ID,
  NODE_IDS,
  bigPhaseFor,
  formatDuration,
  laneToneFor,
  progressMapFromPipelineStages,
  statusMapFromPipelineStages,
} from '../components/verifier/diagramPipelineConstants'
import { useJobStream } from '../hooks/useJobStream'
import { useElapsedStopwatch } from '../hooks/useElapsedStopwatch'

// done 된 단계를 곧바로 목록에서 빼면 "끝까지 채워졌다"를 볼 틈이 없음
// → HOLD_MS 동안 100% 로 붙잡고, FADE_MS 동안 투명도를 0 으로 내린 뒤 목록에서 제거
const HOLD_MS = 900
const FADE_MS = 350
// 실측 진행률이 없는 단계가 진행 중 도달할 수 있는 최대치
// 100 이면 실제로 안 끝났는데 "완료"처럼 보이는 순간이 생기므로 진행 중에는 여기서 멈춤
const SIMULATED_CAP_PCT = 92

// 배치 진행률을 실제로 보고하는 단계들 (pipeline/*.py 가 progress 튜플을 보내는 stage 노드 id)
// 이 목록의 단계는 "아직 첫 진행 신호가 안 왔을 뿐" 배치가 없는 게 아님 —
// 구분하지 않으면 첫 신호 전(예: 첫 슬라이드 VLM 응답 대기 중)에 progress 가 null 이라
// "배치 없는 단계"로 오인해 SIMULATED_CAP_PCT 까지 채웠다가, 첫 실측값(1/12=8%)이 오면 뚝 떨어져 보임
// → 이 목록의 단계는 progress 가 없어도 0% 로 대기, 가짜로 채우지 않음
const STAGES_WITH_BATCH_PROGRESS = new Set([
  'slide_extract',
  'audio_quality',
  'slide_analyze',
  'voice_transcribe',
  'claim_extract',
  'issue_detect',
  'issue_classify',
  'issue_filter',
  'issue_judge',
  'slide_inspect',
  'syntax_verify',
])

// 진행 바가 몇 % 찬 상태로 "툭" 등장하지 않도록, 마운트 첫 프레임은 0 에서 시작해
// 다음 프레임에 실제 값으로 이동 — 이후 값 변경은 같은 DOM 노드라 CSS transition 이 처리
function useMountGrowth(target) {
  const [display, setDisplay] = useState(0)
  const mountedRef = useRef(false)
  useEffect(() => {
    if (!mountedRef.current) {
      mountedRef.current = true
      const raf = requestAnimationFrame(() => setDisplay(target))
      return () => cancelAnimationFrame(raf)
    }
    setDisplay(target)
  }, [target])
  return display
}

// 진행 바 하나 — 여러 상태(실측 % / 배치 없이 0→100% / 완료 유예)를 이 컴포넌트 하나로 처리
// 상태가 바뀌어도 같은 DOM 노드라 폭 변경 시 CSS transition 이 자연스럽게 이어짐
// - 배치 있는 단계: 완료/전체 비율 pct, 값 바뀔 때마다 짧게(450ms) 이동
// - 배치 없는 단계(예: 검증 입력 데이터 구성): 실측 % 없어 정해진 시간에 걸쳐 0→100%
// - 완료 유예(hold): pct=100 고정, 빠른 transition 으로 마무리
function FillBar({ toneClass, pct, transitionClass, fading }) {
  const displayPct = useMountGrowth(pct)
  return (
    <div className="vf-stage-bar-track" style={{ opacity: fading ? 0 : 1, transition: `opacity ${FADE_MS}ms ease` }}>
      <div className={`vf-stage-bar-fill ${transitionClass} ${toneClass}`} style={{ width: `${displayPct}%` }} />
    </div>
  )
}

// 단계 상태(completed / 실측 progress / 배치 대기 / 배치 없음)에 따라 FillBar 의 pct 결정
function StageProgressBar({ id, toneClass, progress, completed, fading }) {
  if (completed) {
    return <FillBar toneClass={toneClass} pct={100} transitionClass="vf-stage-bar-fill--metered" fading={fading} />
  }
  if (progress) {
    // 100 은 "완료"로 읽히므로 done 이 와서 completed 분기로 넘어가기 전까지는 99 에서 멈춤
    // 음성 전사처럼 배치(P2B 전사) 종료 후에도 done 안 나는 후처리(P3)가 남은 단계는
    // 이 캡이 없으면 한참 100% 에 멈춰 있는 것처럼 보임
    const pct = Math.min(99, Math.round((progress[0] / Math.max(1, progress[1])) * 100))
    return <FillBar toneClass={toneClass} pct={pct} transitionClass="vf-stage-bar-fill--metered" fading={false} />
  }
  if (STAGES_WITH_BATCH_PROGRESS.has(id)) {
    // 배치는 있으나 첫 진행 신호 대기 중 — 가짜로 채우면 첫 실측값 도착 시 뚝 떨어져 보이므로 0% 대기
    return <FillBar toneClass={toneClass} pct={0} transitionClass="vf-stage-bar-fill--metered" fading={false} />
  }
  // 배치 자체가 없는 단계(예: 검증 입력 데이터 구성 — LLM·병렬 워커 없는 단순 조립)
  // → 진행 중에는 SIMULATED_CAP_PCT 까지만, done 이 올 때만 100%
  return <FillBar toneClass={toneClass} pct={SIMULATED_CAP_PCT} transitionClass="vf-stage-bar-fill--simulated" fading={false} />
}

// 'run' 이던 노드가 다른 상태(done/error)로 바뀌는 순간을 감지해, HOLD_MS 동안 'hold'(100% 고정),
// 이어서 FADE_MS 동안 'fade'(투명도 감소) 처리 후 목록에서 제거
//
// 핵심: 방금 run 에서 벗어난 id 를 "이 렌더에서 곧바로" settling 에 포함
// useEffect 실행 후(한 렌더 늦게) 반영하면, 그 사이 렌더에서 diagramStatus 도 settling 도 그 id 를
// 빼먹어 FillBar 가 언마운트→재마운트되며 useMountGrowth 가 0% 부터 다시 애니메이션됨("완료 후 한 번 더 참")
// → prevStatusRef 를 렌더 중 직접 갱신하고, 방금 끝난 id 를 즉석 Map 에 포함해 반환 (언마운트 렌더 없음)
function useSettlingIds(diagramStatus) {
  const [settling, setSettling] = useState(() => new Map())
  const prevStatusRef = useRef({})
  const timersRef = useRef({})

  const prevStatus = prevStatusRef.current
  const justFinished = NODE_IDS.filter(
    id => prevStatus[id] === 'run' && diagramStatus[id] !== 'run' && !settling.has(id)
  )
  prevStatusRef.current = diagramStatus

  const effectiveSettling = justFinished.length === 0
    ? settling
    : (() => {
        const next = new Map(settling)
        justFinished.forEach(id => next.set(id, 'hold'))
        return next
      })()

  useEffect(() => {
    if (justFinished.length === 0) return

    setSettling(prev => {
      const next = new Map(prev)
      justFinished.forEach(id => next.set(id, 'hold'))
      return next
    })

    justFinished.forEach(id => {
      clearTimeout(timersRef.current[id]?.hold)
      clearTimeout(timersRef.current[id]?.fade)
      const holdTimer = setTimeout(() => {
        setSettling(prev => {
          if (!prev.has(id)) return prev
          const next = new Map(prev)
          next.set(id, 'fade')
          return next
        })
        const fadeTimer = setTimeout(() => {
          setSettling(prev => {
            if (!prev.has(id)) return prev
            const next = new Map(prev)
            next.delete(id)
            return next
          })
        }, FADE_MS)
        timersRef.current[id] = { ...timersRef.current[id], fade: fadeTimer }
      }, HOLD_MS)
      timersRef.current[id] = { hold: holdTimer }
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [justFinished.join(',')])

  useEffect(() => () => {
    Object.values(timersRef.current).forEach(t => {
      clearTimeout(t?.hold)
      clearTimeout(t?.fade)
    })
  }, [])

  return effectiveSettling
}

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

  // job.created_at = 서버가 찍은 실제 작업 시작 시각 — 마운트 시각이 아니라 이걸 기준으로 해야
  // 새로고침해도 경과 시간이 0 으로 돌아가지 않음
  const jobStartedAtMs = lecture.job?.created_at ? new Date(lecture.job.created_at).getTime() : undefined
  const elapsedMs = useElapsedStopwatch(!isLoading && phase === PHASES.PIPELINE, jobStartedAtMs)

  // 완료되면 결과 화면으로 이동 — 이미 완료된 강의로 진입해도 여기서 즉시 리다이렉트되므로
  // 목록에서는 상태 무관하게 항상 이 라우트로 보내면 됨
  useEffect(() => {
    if (phase === PHASES.VERIFY_READY) {
      navigate(`/result/${lectureId}`, { replace: true })
    }
  }, [phase, lectureId, navigate])

  // 훅 호출 순서 고정을 위해 아래 isLoading 이른 반환보다 앞에서 계산
  // isLoading 이면 pipelineStages 가 비어도 각 함수의 기본값(빈 배열)이 안전 처리
  const diagramStatus = statusMapFromPipelineStages(pipelineStages)
  const progressMap = progressMapFromPipelineStages(pipelineStages)
  const settlingIds = useSettlingIds(diagramStatus)

  if (isLoading) return <p className="list-note">불러오는 중...</p>

  return (
    <div className="detail">
      <div className="detail-head">
        <button type="button" className="btn" onClick={goToList}>← 목록으로</button>
        <h2>{lecture.title || lectureId}</h2>
      </div>

      {phase === PHASES.PIPELINE && (() => {
        // 같은 레인 안 단계들은 백엔드에서 순차 실행이라 동시에 진행되지 않음 —
        // 겹쳐 보이는 건 완료 유예 중인 이전 단계 + 막 시작한 다음 단계의 시각적 잔상
        // → 새로 'run' 된 단계는 같은 레인에 유예 중인 단계가 남아있으면 사라질 때까지 표시 보류
        // 레인이 다른 단계(영상/오디오, 발화검증/슬라이드검사)는 실제 동시 진행이라 이 지연 미적용
        const activeIds = NODE_IDS.filter(id => {
          if (settlingIds.has(id)) return true
          if (diagramStatus[id] !== 'run') return false
          const lane = NODE_BY_ID[id]?.lane
          if (!lane) return true
          const laneStillSettling = NODE_IDS.some(
            otherId => otherId !== id && NODE_BY_ID[otherId]?.lane === lane && settlingIds.has(otherId)
          )
          return !laneStillSettling
        })
        const runningIds = NODE_IDS.filter(id => diagramStatus[id] === 'run')
        const bigPhase = bigPhaseFor(runningIds[0] || 'error_output')
        const stageMessage = String(currentStage || '').trim() || '분석 준비 중...'
        return (
          <div className="vf-pipe">
            <div className="vf-progress-row">
              <div className="vf-progress-message"><strong>{bigPhase}</strong>: {stageMessage}</div>
              <span className="vf-stopwatch">{formatDuration(elapsedMs)}</span>
            </div>
            {activeIds.length > 0 && (
              <div className="vf-stage-bars">
                {activeIds.map(id => {
                  const settlePhase = settlingIds.get(id)
                  const fading = settlePhase === 'fade'
                  return (
                    <Fragment key={id}>
                      <span
                        className="vf-stage-bar-label"
                        style={{ opacity: fading ? 0 : 1, transition: `opacity ${FADE_MS}ms ease` }}
                      >
                        {NODE_BY_ID[id]?.label}
                      </span>
                      <StageProgressBar
                        id={id}
                        toneClass={`vf-stage-bar-fill--${laneToneFor(id)}`}
                        progress={progressMap[id]}
                        completed={diagramStatus[id] !== 'run'}
                        fading={fading}
                      />
                    </Fragment>
                  )
                })}
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
