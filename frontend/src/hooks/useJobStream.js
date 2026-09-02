import { useEffect, useMemo, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { PHASES, normalizePipelineStages } from '../components/verifier/verifierConstants'
import {
  deleteLecture,
  getLectureDetail,
  getLectureResult,
  jobStreamUrl,
  checkHealth,
  retryLecture as retryJobRequest,
} from '../api/pipeline'
import { isTerminalStatus, markAllStages, mergeStageStatus, phaseFromStatus } from '../lib/jobStreamUtils'

const EMPTY_LECTURE = {
  id: '',
  title: '',
  video_url: '',
  status: '',
}

// VLVerifier의 useJobStream을 verify 전용으로 축소 이식한 버전.
// 라우터 의존을 없애고, 목록 복귀는 onExit 콜백으로 위임한다.
export function useJobStream(lectureId, { onExit } = {}) {
  const queryClient = useQueryClient()
  const eventSourceRef = useRef(null)
  const verifierLoadedRef = useRef(false)
  const ignoreNextStreamErrorRef = useRef(false)
  const healthTimerRef = useRef(null)

  const [lecture, setLecture] = useState(EMPTY_LECTURE)
  const [verifier, setVerifier] = useState(null)
  const [pipelineStages, setPipelineStages] = useState(() => normalizePipelineStages())
  const [phase, setPhase] = useState(PHASES.PIPELINE)
  const [currentStage, setCurrentStage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [isLoading, setIsLoading] = useState(true)
  const [isMutating, setIsMutating] = useState(false)

  function stopHealthCheck() {
    if (!healthTimerRef.current) return
    clearInterval(healthTimerRef.current)
    healthTimerRef.current = null
  }

  function closeEventSource({ ignoreStreamError = false } = {}) {
    if (eventSourceRef.current) {
      if (ignoreStreamError) ignoreNextStreamErrorRef.current = true
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
    stopHealthCheck()
  }

  function handleServerDown() {
    closeEventSource({ ignoreStreamError: true })
    setCurrentStage('서버와의 연결이 끊어졌습니다.')
    setErrorMessage('서버와의 연결이 끊어졌습니다.')
    setPhase(PHASES.ERROR)
  }

  function startHealthCheck() {
    if (healthTimerRef.current) return
    healthTimerRef.current = setInterval(async () => {
      if (!eventSourceRef.current) {
        stopHealthCheck()
        return
      }
      const controller = new AbortController()
      const timeoutId = window.setTimeout(() => controller.abort(), 2500)
      try {
        await checkHealth(controller.signal)
      } catch {
        handleServerDown()
      } finally {
        window.clearTimeout(timeoutId)
      }
    }, 3000)
  }

  // result는 GET /lectures/{id}/result의 독립 응답이다. done 도달 시에만 이 엔드포인트를 부른다.
  function loadVerificationIfReady(status) {
    if (phaseFromStatus(status) !== PHASES.VERIFY_READY || verifierLoadedRef.current) return
    verifierLoadedRef.current = true
    getLectureResult(lectureId)
      .then(result => setVerifier(result))
      .catch(() => {
        verifierLoadedRef.current = false
      })
  }

  function connectJob() {
    if (!lectureId) return

    closeEventSource()
    ignoreNextStreamErrorRef.current = false

    const eventSource = new EventSource(jobStreamUrl(lectureId))
    eventSourceRef.current = eventSource
    startHealthCheck()

    eventSource.onmessage = event => {
      try {
        const payload = JSON.parse(event.data)
        if (payload.error) throw new Error(payload.error)

        const nextPhase = phaseFromStatus(payload.status)
        setPhase(nextPhase)
        setCurrentStage(payload.current_stage || '')
        setErrorMessage(prev =>
          payload.error_message || (nextPhase === PHASES.ERROR ? payload.current_stage || prev : '')
        )
        setLecture(prev => ({
          ...prev,
          id: prev.id || lectureId,
          job_id: payload.job_id || prev.job_id,
          status: payload.status || prev.status,
        }))
        setPipelineStages(prev => mergeStageStatus(prev, payload.pipeline_stages || []))

        loadVerificationIfReady(payload.status)

        // verify 완료 → 통계 페이지가 열려 있으면 자동 갱신.
        // 백엔드가 done 표시 전에 verification_stats 를 넣으므로 여기서 행이 보장된다.
        if (payload.status === 'done') {
          queryClient.invalidateQueries({ queryKey: ['stats'] })
        }

        if (isTerminalStatus(payload.status)) {
          closeEventSource({ ignoreStreamError: true })
        }
      } catch (error) {
        closeEventSource()
        setErrorMessage(String(error?.message || error))
        setPhase(PHASES.ERROR)
      }
    }

    eventSource.onerror = () => {
      if (ignoreNextStreamErrorRef.current) {
        ignoreNextStreamErrorRef.current = false
        return
      }
      closeEventSource()
      setErrorMessage('서버와의 연결이 끊어졌습니다.')
      setPhase(PHASES.ERROR)
    }
  }

  useEffect(() => {
    if (!lectureId) return undefined

    let cancelled = false
    setIsLoading(true)
    setErrorMessage('')

    async function loadInitialState() {
      try {
        const detail = await getLectureDetail(lectureId)
        if (cancelled) return

        const job = detail.job || {}
        const nextPhase = phaseFromStatus(job.status)

        // lectureId가 바뀌면 이전 강의의 result가 남지 않도록 초기화한다.
        verifierLoadedRef.current = false
        setVerifier(null)
        setLecture({ ...EMPTY_LECTURE, ...detail, status: job.status || '' })
        setPhase(nextPhase)
        setCurrentStage(job.current_stage || '')
        setErrorMessage(job.error_message || (nextPhase === PHASES.ERROR ? job.current_stage || '' : ''))

        if (Array.isArray(job.pipeline_stages) && job.pipeline_stages.length > 0) {
          setPipelineStages(mergeStageStatus(markAllStages('wait'), job.pipeline_stages))
        } else {
          setPipelineStages(markAllStages(job.status === 'done' ? 'done' : 'wait'))
        }

        loadVerificationIfReady(job.status)
        if (!isTerminalStatus(job.status)) connectJob()
      } catch (error) {
        if (!cancelled) {
          setErrorMessage(String(error?.message || error))
          setPhase(PHASES.ERROR)
        }
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }

    loadInitialState()
    return () => {
      cancelled = true
      closeEventSource()
    }
  }, [lectureId])

  async function restart() {
    if (!lectureId || isMutating) return

    setIsMutating(true)
    setErrorMessage('')
    verifierLoadedRef.current = false
    setVerifier(null)
    setPhase(PHASES.PIPELINE)
    setCurrentStage('검증 파이프라인을 다시 시작합니다.')
    setPipelineStages(markAllStages('wait'))

    try {
      const result = await retryJobRequest(lectureId)
      // 새 job이 지금 막 시작됐으므로, 경과 시간 기준점(job.created_at)도 지금 시각으로
      // 갱신한다 — 갱신하지 않으면 스톱워치가 실패했던 이전 시도의 경과 시간을 이어받는다.
      setLecture(prev => ({
        ...prev,
        job_id: result.job_id || prev.job_id,
        status: 'pending',
        job: { ...prev.job, created_at: new Date().toISOString() },
      }))
      setIsMutating(false)
      connectJob()
    } catch (error) {
      setIsMutating(false)
      setErrorMessage(String(error?.message || error))
      setPhase(PHASES.ERROR)
    }
  }

  async function remove() {
    if (!lectureId || isMutating) return
    if (!window.confirm('이 강의와 생성된 파일을 삭제할까요?')) return

    closeEventSource()
    setIsMutating(true)
    setErrorMessage('')
    try {
      await deleteLecture(lectureId)
      onExit?.()
    } catch (error) {
      setIsMutating(false)
      setErrorMessage(String(error?.message || error))
      setPhase(PHASES.ERROR)
    }
  }

  return {
    phase,
    lecture,
    verifier,
    pipelineStages,
    currentStage,
    errorMessage,
    isLoading,
    isMutating,
    actions: { restart, remove },
  }
}
