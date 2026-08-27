import { useEffect, useRef, useState } from 'react'
import { DEMO_PHASES } from './useDemoPipelineFlow'
import { NODE_BY_ID, NODE_IDS, TICKS } from '../components/verifier/diagramPipelineConstants'

// 다이어그램 파이프라인 데모(/dev/verify-demo-diagram, /dev/verify-demo) 공용. 구조는
// useDemoPipelineFlow와 같지만(업로드→진행→완료, 로컬 타이머만 사용) 한 틱에 여러 노드가
// 동시에 활성화될 수 있다는 점이 다르다. 틱 간격은 페이지마다 다르게 쓸 수 있게 인자로 뺐다.
const DEFAULT_TICK_DELAY_MS = 1400
const LAST_TICK_INDEX = TICKS.length - 1

// 통합 멀티모달 텍스트 생성은 실제로 오래 걸리는 단계가 아니라서, 다른 단계처럼
// 오래 머무르지 않고 한두 번만 깜빡인 뒤 바로 다음 단계로 넘어가게 짧게 잡는다.
const QUICK_TICK_NODE_IDS = new Set(['integrated_text'])
const QUICK_TICK_DELAY_MS = 1500

function fileTitle(file) {
  return file?.name ? file.name.replace(/\.[^.]+$/, '') : ''
}

function buildStatus(tickIndex, { errorAtCurrent = false } = {}) {
  // video는 파이프라인 진입 전(업로드 시점)에 이미 끝난 단계라 항상 done으로 표시한다.
  const status = Object.fromEntries(NODE_IDS.map(id => [id, id === 'video' ? 'done' : 'wait']))
  TICKS.forEach((ids, i) => {
    ids.forEach(id => {
      if (i < tickIndex) status[id] = 'done'
      else if (i === tickIndex) status[id] = errorAtCurrent ? 'error' : 'run'
    })
  })
  return status
}

const ALL_DONE_STATUS = Object.fromEntries(NODE_IDS.map(id => [id, 'done']))

export function useDemoDiagramFlow(tickDelayMs = DEFAULT_TICK_DELAY_MS) {
  const [phase, setPhase] = useState(DEMO_PHASES.UPLOAD)
  const [file, setFile] = useState(null)
  const [title, setTitle] = useState('')
  const [isTitleManual, setIsTitleManual] = useState(false)
  const [tickIndex, setTickIndex] = useState(0)
  const [autoPlay, setAutoPlay] = useState(true)

  const timerRef = useRef(null)

  useEffect(() => {
    if (phase !== DEMO_PHASES.PIPELINE || !autoPlay) return undefined

    const currentIds = TICKS[tickIndex] || []
    const delay = currentIds.some(id => QUICK_TICK_NODE_IDS.has(id)) ? QUICK_TICK_DELAY_MS : tickDelayMs

    timerRef.current = setTimeout(() => {
      setTickIndex(prev => {
        if (prev >= LAST_TICK_INDEX) {
          setPhase(DEMO_PHASES.DONE)
          return prev
        }
        return prev + 1
      })
    }, delay)

    return () => clearTimeout(timerRef.current)
  }, [phase, autoPlay, tickIndex, tickDelayMs])

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
    setTickIndex(0)
    setAutoPlay(true)
    setPhase(DEMO_PHASES.PIPELINE)
  }

  function reset() {
    setPhase(DEMO_PHASES.UPLOAD)
    setFile(null)
    setTitle('')
    setIsTitleManual(false)
    setTickIndex(0)
    setAutoPlay(true)
  }

  // 파이프라인 화면에서 "이전으로" — reset과 달리 이미 고른 파일·제목은 그대로 두고
  // 업로드 화면으로만 돌아간다.
  function backToUpload() {
    setPhase(DEMO_PHASES.UPLOAD)
    setTickIndex(0)
    setAutoPlay(true)
  }

  function next() {
    if (phase === DEMO_PHASES.ERROR) return
    setTickIndex(prev => {
      if (prev >= LAST_TICK_INDEX) {
        setPhase(DEMO_PHASES.DONE)
        return prev
      }
      return prev + 1
    })
  }

  function prev() {
    if (phase === DEMO_PHASES.ERROR) {
      setPhase(DEMO_PHASES.PIPELINE)
      return
    }
    setTickIndex(current => Math.max(0, current - 1))
  }

  function toggleAutoPlay() {
    setAutoPlay(value => !value)
  }

  function simulateError() {
    setAutoPlay(false)
    setPhase(DEMO_PHASES.ERROR)
  }

  function recoverFromError() {
    setPhase(DEMO_PHASES.PIPELINE)
  }

  const status = phase === DEMO_PHASES.DONE
    ? ALL_DONE_STATUS
    : buildStatus(tickIndex, { errorAtCurrent: phase === DEMO_PHASES.ERROR })

  const activeLabels = (TICKS[tickIndex] || []).map(id => NODE_BY_ID[id]?.label).filter(Boolean)
  const currentStage =
    phase === DEMO_PHASES.DONE
      ? '모든 단계가 완료되었습니다. (데모)'
      : phase === DEMO_PHASES.ERROR
        ? `${activeLabels.join(' · ')} 단계에서 오류가 발생했습니다. (데모)`
        : `${activeLabels.join(' · ')} 진행 중 (데모)`

  return {
    phase,
    file,
    title,
    tickIndex,
    lastTickIndex: LAST_TICK_INDEX,
    autoPlay,
    status,
    currentStage,
    activeLabels,
    actions: {
      selectFile,
      setTitle: setTitleManual,
      start,
      reset,
      backToUpload,
      next,
      prev,
      toggleAutoPlay,
      simulateError,
      recoverFromError,
    },
  }
}
