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

// done으로 바뀐 단계를 곧바로 목록에서 빼면 "끝까지 채워졌다"는 걸 확인할 틈이 없다.
// HOLD_MS 동안 100%로 붙잡아 두고, 그 다음 FADE_MS 동안 옵아시티를 0으로 내린 뒤에야
// 목록에서 뺀다.
const HOLD_MS = 900
const FADE_MS = 350
// 실측 진행률이 없는 단계가 진행 중에 도달할 수 있는 최대치. 100으로 두면 실제로는
// 안 끝났는데 "완료"처럼 보이는 순간이 생기므로, 진행 중에는 여기서 멈춘다.
const SIMULATED_CAP_PCT = 92

// 배치 진행률을 실제로 보고하는 단계들(pipeline/*.py에서 progress 튜플을 보내는 stage에
// 대응하는 노드 id). 이 목록에 있는 단계는 "아직 첫 진행 신호가 안 왔을 뿐"이지 배치가
// 없는 게 아니다 — 이 차이를 구분하지 않으면, 아직 첫 신호가 오기 전(예: 첫 슬라이드
// VLM 응답이 6초 넘게 걸리는 동안)에는 progress가 null이라서 "배치 없는 단계"로 오인해
// SIMULATED_CAP_PCT까지 채워버리고, 막상 첫 실측값(예: 1/12=8%)이 도착하면 갑자기
// 아래로 떨어지는 것처럼 보인다. 그래서 이 목록에 있는 단계는 progress가 아직 없어도
// 0%로 대기할 뿐 가짜로 채우지 않는다.
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

// 진행 바가 처음 나타날 때 이미 몇 %가 찬 상태로 "툭" 등장하지 않도록, 마운트 첫 프레임은
// 항상 0에서 시작해서 다음 프레임에 실제 값으로 옮겨간다 — 이후 값이 바뀔 때는 그냥 그대로
// 반영하면 같은 DOM 노드라 CSS transition이 자연스럽게 애니메이션한다.
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

// 하나의 진행 바가 겪는 모든 상태(실측 %로 채워짐 / 배치 정보 없이 0→100%로 채워짐 /
// 완료 유예)를 전부 이 컴포넌트 하나로 그린다 — 상태가 바뀌어도 같은 DOM 노드를 그대로
// 쓰기 때문에 폭이 바뀔 때마다 CSS transition이 자연스럽게 애니메이션한다.
// - 배치가 있는 단계: 완료/전체 비율로 pct가 오고, 값이 바뀔 때마다 짧게(450ms) 움직인다.
// - 배치가 없는 단계(예: 검증 입력 데이터 구성): 실측 %를 모르므로 정해진 시간에 걸쳐
//   0%에서 100%까지 채운다 — 짧게 끝나는 실제 단계는 100%에 먼저 도달해 잠깐 멈춰
//   있을 수 있지만, 특정 %에서 멈춰 서있는 것보다는 계속 차오르는 쪽이 자연스럽다.
// - 완료 유예(hold): pct=100으로 고정, 항상 빠른(450ms) transition으로 마무리한다.
function FillBar({ toneClass, pct, transitionClass, fading }) {
  const displayPct = useMountGrowth(pct)
  return (
    <div className="vf-stage-bar-track" style={{ opacity: fading ? 0 : 1, transition: `opacity ${FADE_MS}ms ease` }}>
      <div className={`vf-stage-bar-fill ${transitionClass} ${toneClass}`} style={{ width: `${displayPct}%` }} />
    </div>
  )
}

function StageProgressBar({ id, toneClass, progress, completed, fading }) {
  if (completed) {
    return <FillBar toneClass={toneClass} pct={100} transitionClass="vf-stage-bar-fill--metered" fading={fading} />
  }
  if (progress) {
    // 100은 "완료됨"으로 읽히므로 실제로 done이 와서 completed 분기로 넘어가기 전까지는
    // 99에서 멈춘다. 대부분의 단계는 마지막 배치 항목이 끝나자마자 done도 같이 오니 거의
    // 안 보이지만, 음성 전사(voice_transcribe)처럼 배치(P2B 전사)가 다 끝난 뒤에도 done 안
    // 나는 후처리(P3 오디오 context 후처리)가 남아있는 단계는 이 캡이 없으면 배치가 끝난
    // 순간부터 done이 올 때까지 한참 100%에 멈춰 있는 것처럼 보인다.
    const pct = Math.min(99, Math.round((progress[0] / Math.max(1, progress[1])) * 100))
    return <FillBar toneClass={toneClass} pct={pct} transitionClass="vf-stage-bar-fill--metered" fading={false} />
  }
  if (STAGES_WITH_BATCH_PROGRESS.has(id)) {
    // 배치는 있지만 아직 첫 진행 신호가 도착하지 않은 상태(예: 첫 슬라이드 VLM 응답
    // 대기 중) — 배치가 아예 없는 단계와 달리, 가짜로 채우면 첫 실측값이 도착하는
    // 순간 아래로 떨어지는 것처럼 보이므로 0%에서 정직하게 대기한다.
    return <FillBar toneClass={toneClass} pct={0} transitionClass="vf-stage-bar-fill--metered" fading={false} />
  }
  // 이 단계는 배치 자체가 없어서(예: 검증 입력 데이터 구성 — LLM 호출도 병렬 워커도 없는
  // 단순 조립) "몇 개 중 몇 개"라는 실측값이 원천적으로 없다. 그래서 진행 중에는 100%
  // 직전(SIMULATED_CAP_PCT)까지만 채워서 "아직 안 끝났다"는 걸 거짓말하지 않고,
  // 실제로 done이 오는 순간에만(completed 분기) 100%로 넘어간다.
  return <FillBar toneClass={toneClass} pct={SIMULATED_CAP_PCT} transitionClass="vf-stage-bar-fill--simulated" fading={false} />
}

// diagramStatus에서 'run'이던 노드가 다른 상태(done/error)로 바뀌는 순간을 감지해,
// HOLD_MS 동안은 'hold'(100% 고정 표시), 그 다음 FADE_MS 동안은 'fade'(옵아시티 감소)로
// 표시하고 나서야 목록에서 뺀다.
//
// 방금 run에서 벗어난 id를 "이 렌더에서 곧바로" settling에 포함시키는 게 핵심이다.
// 예전 버전처럼 useEffect가 실행된 뒤에야(한 렌더 늦게) settling state에 반영하면, 그
// 사이의 한 렌더에서 diagramStatus도 'run'이 아니고 settling에도 아직 없어서 activeIds가
// 그 id를 빼먹는다 — 그 순간 FillBar가 언마운트됐다가 다음 렌더에서 다시 마운트되면서
// useMountGrowth가 처음부터(0%) 다시 애니메이션을 트는 것처럼 보인다("완료되면 한 번 더
// 0→100으로 채워짐"). 그래서 prevStatusRef를 렌더 중에 직접 갱신하고, 방금 끝난 id를
// 그 자리에서 만든 Map에 즉시 포함시켜 반환한다 — 언마운트되는 렌더가 아예 없다.
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

  // job.created_at은 서버가 찍은, 이 작업이 실제로 시작된 시각이다 — 컴포넌트 마운트
  // 시각이 아니라 이걸 기준으로 삼아야 새로고침해도 경과 시간이 0으로 돌아가지 않는다.
  const jobStartedAtMs = lecture.job?.created_at ? new Date(lecture.job.created_at).getTime() : undefined
  const elapsedMs = useElapsedStopwatch(!isLoading && phase === PHASES.PIPELINE, jobStartedAtMs)

  // 파이프라인이 완료되면 결과 화면으로 이동한다. 이미 완료된 강의로 바로 진입한 경우도
  // 여기서 즉시 리다이렉트되므로, 목록에서는 상태와 무관하게 항상 이 라우트로 보내면 된다.
  useEffect(() => {
    if (phase === PHASES.VERIFY_READY) {
      navigate(`/result/${lectureId}`, { replace: true })
    }
  }, [phase, lectureId, navigate])

  // 훅은 항상 같은 순서로 호출되어야 하므로, 아래 isLoading 이른 반환보다 앞에서 계산한다.
  // isLoading일 때는 pipelineStages가 비어 있어도 각 함수의 기본값(빈 배열)이 안전하게 처리한다.
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
        // 같은 레인(예: 발화검증 체인의 claim 추출→오류 탐지) 안의 단계들은 백엔드에서
        // 실제로 순차 실행이라 절대 동시에 진행되지 않는다 — 지금 겹쳐 보이는 건 완료
        // 유예(hold/fade) 중인 이전 단계와 막 시작한 다음 단계가 잠깐 같이 목록에 있어서
        // 생기는 시각적 잔상일 뿐이다. 그래서 새로 'run'이 된 단계는, 같은 레인에 아직
        // 유예 중인(사라지는 중인) 단계가 남아있으면 그게 다 사라질 때까지 표시를 미룬다.
        // 레인이 다른 단계(영상/오디오, 발화검증/슬라이드검사)는 실제로 동시에 진행되므로
        // 이 지연을 적용하지 않는다 — 그러면 진짜 진행 중인 바가 가려지는 문제가 생긴다.
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
