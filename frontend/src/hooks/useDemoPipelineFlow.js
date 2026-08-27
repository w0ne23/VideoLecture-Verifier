import { useEffect, useRef, useState } from 'react'
import { PHASES, PIPELINE_NODES, VERIFY_STAGE_KEYS, normalizePipelineStages } from '../components/verifier/verifierConstants'

// 파이프라인 진행 UI만 반복해서 확인하기 위한 데모 전용 훅.
// 실제 업로드/작업 API를 호출하지 않고, 로컬 타이머와 수동 버튼으로 stage를 넘긴다.
const STAGE_DELAY_MS = 1500
const LAST_STAGE_INDEX = VERIFY_STAGE_KEYS.length - 1

export const DEMO_PHASES = {
  UPLOAD: 'upload',
  PIPELINE: PHASES.PIPELINE,
  ERROR: PHASES.ERROR,
  DONE: 'done',
}

function stagesAt(stageIndex, { errorAtCurrent = false } = {}) {
  return normalizePipelineStages(
    VERIFY_STAGE_KEYS.map((stage, index) => {
      if (index < stageIndex) return { stage, status: 'done' }
      if (index === stageIndex) return { stage, status: errorAtCurrent ? 'error' : 'run' }
      return { stage, status: 'wait' }
    })
  )
}

const ALL_DONE_STAGES = normalizePipelineStages(VERIFY_STAGE_KEYS.map(stage => ({ stage, status: 'done' })))

function fileTitle(file) {
  return file?.name ? file.name.replace(/\.[^.]+$/, '') : ''
}

export function useDemoPipelineFlow() {
  const [phase, setPhase] = useState(DEMO_PHASES.UPLOAD)
  const [file, setFile] = useState(null)
  const [title, setTitle] = useState('')
  const [isTitleManual, setIsTitleManual] = useState(false)
  const [stageIndex, setStageIndex] = useState(0)
  const [autoPlay, setAutoPlay] = useState(true)

  const timerRef = useRef(null)

  useEffect(() => {
    if (phase !== DEMO_PHASES.PIPELINE || !autoPlay) return undefined

    timerRef.current = setTimeout(() => {
      setStageIndex(prev => {
        if (prev >= LAST_STAGE_INDEX) {
          setPhase(DEMO_PHASES.DONE)
          return prev
        }
        return prev + 1
      })
    }, STAGE_DELAY_MS)

    return () => clearTimeout(timerRef.current)
  }, [phase, autoPlay, stageIndex])

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
    setStageIndex(0)
    setAutoPlay(true)
    setPhase(DEMO_PHASES.PIPELINE)
  }

  function reset() {
    setPhase(DEMO_PHASES.UPLOAD)
    setFile(null)
    setTitle('')
    setIsTitleManual(false)
    setStageIndex(0)
    setAutoPlay(true)
  }

  function next() {
    if (phase === DEMO_PHASES.ERROR) return
    setStageIndex(prev => {
      if (prev >= LAST_STAGE_INDEX) {
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
    setStageIndex(current => Math.max(0, current - 1))
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

  const pipelineStages =
    phase === DEMO_PHASES.DONE ? ALL_DONE_STAGES : stagesAt(stageIndex, { errorAtCurrent: phase === DEMO_PHASES.ERROR })

  const activeNode = PIPELINE_NODES[stageIndex]
  const currentStage =
    phase === DEMO_PHASES.DONE
      ? '모든 단계가 완료되었습니다. (데모)'
      : phase === DEMO_PHASES.ERROR
        ? `${activeNode?.stageLabel || ''} 단계에서 오류가 발생했습니다. (데모)`
        : `${activeNode?.stageLabel || ''} 진행 중 (데모)`

  return {
    phase,
    file,
    title,
    stageIndex,
    lastStageIndex: LAST_STAGE_INDEX,
    autoPlay,
    pipelineStages,
    currentStage,
    actions: {
      selectFile,
      setTitle: setTitleManual,
      start,
      reset,
      next,
      prev,
      toggleAutoPlay,
      simulateError,
      recoverFromError,
    },
  }
}
