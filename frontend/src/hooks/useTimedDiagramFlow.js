import { useEffect, useRef, useState } from 'react'
import { DEMO_PHASES } from './useDemoPipelineFlow'
import { NODE_BY_ID, NODE_IDS, NODE_SCHEDULE, TOTAL_DURATION_MS } from '../components/verifier/diagramPipelineConstants'

// VerifyDemoLivePage 전용. useDemoDiagramFlow(틱 인덱스를 setTimeout으로 하나씩 넘기는
// 방식)와 달리, 노드마다 정해진 시작·소요 시간(NODE_SCHEDULE)과 "지금까지 흐른 시간"만
// 비교해서 상태를 구한다 — 같은 구간에서 나란히 도는 두 노드가 서로 다른 속도로 끝나도
// (예: 슬라이드 검증이 발화 검증보다 먼저 끝남) 자연스럽게 표현된다.
const POLL_MS = 250

function fileTitle(file) {
  return file?.name ? file.name.replace(/\.[^.]+$/, '') : ''
}

function statusForNode(elapsedMs, entry) {
  if (!entry) return 'wait'
  if (elapsedMs >= entry.start + entry.duration) return 'done'
  if (elapsedMs >= entry.start) return 'run'
  return 'wait'
}

function buildStatus(elapsedMs) {
  const status = { video: 'done' }
  for (const id of NODE_IDS) {
    if (id === 'video') continue
    status[id] = statusForNode(elapsedMs, NODE_SCHEDULE[id])
  }
  return status
}

const ALL_DONE_STATUS = Object.fromEntries(NODE_IDS.map(id => [id, 'done']))

export function useTimedDiagramFlow() {
  const [phase, setPhase] = useState(DEMO_PHASES.UPLOAD)
  const [file, setFile] = useState(null)
  const [title, setTitle] = useState('')
  const [isTitleManual, setIsTitleManual] = useState(false)
  const [elapsedMs, setElapsedMs] = useState(0)

  const intervalRef = useRef(null)

  // 스톱워치이자 상태 갱신 타이머 — 시작 시각과의 실제 차이를 매번 다시 계산해
  // setInterval 누적 오차를 피한다. 전체 일정(TOTAL_DURATION_MS)이 끝나면 완료 처리한다.
  useEffect(() => {
    if (phase !== DEMO_PHASES.PIPELINE) return undefined
    const startedAt = Date.now() - elapsedMs
    intervalRef.current = setInterval(() => {
      const next = Date.now() - startedAt
      if (next >= TOTAL_DURATION_MS) {
        setElapsedMs(TOTAL_DURATION_MS)
        setPhase(DEMO_PHASES.DONE)
        return
      }
      setElapsedMs(next)
    }, POLL_MS)
    return () => clearInterval(intervalRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase])

  function selectFile(nextFile) {
    if (!nextFile) return
    setFile(nextFile)
    // 사용자가 제목을 직접 수정한 적이 없을 때만 파일명으로 자동 갱신한다.
    setTitle(prev => (isTitleManual && prev.trim() ? prev : fileTitle(nextFile)))
  }

  function setTitleManual(value) {
    setIsTitleManual(true)
    setTitle(value)
  }

  function start() {
    if (!file) return
    setElapsedMs(0)
    setPhase(DEMO_PHASES.PIPELINE)
  }

  function reset() {
    setPhase(DEMO_PHASES.UPLOAD)
    setFile(null)
    setTitle('')
    setIsTitleManual(false)
    setElapsedMs(0)
  }

  // 파이프라인 화면에서 "이전으로" — reset과 달리 이미 고른 파일·제목은 그대로 두고
  // 업로드 화면으로만 돌아간다.
  function backToUpload() {
    setPhase(DEMO_PHASES.UPLOAD)
    setElapsedMs(0)
  }

  const status = phase === DEMO_PHASES.DONE ? ALL_DONE_STATUS : buildStatus(elapsedMs)
  const activeIds = phase === DEMO_PHASES.DONE ? [] : NODE_IDS.filter(id => status[id] === 'run')
  const activeLabels = activeIds.map(id => NODE_BY_ID[id]?.label).filter(Boolean)

  return {
    phase,
    file,
    title,
    elapsedMs,
    status,
    activeIds,
    activeLabels,
    actions: {
      selectFile,
      setTitle: setTitleManual,
      start,
      reset,
      backToUpload,
    },
  }
}
