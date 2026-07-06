import { useEffect, useMemo, useRef, useState } from 'react'
import { PHASES, normalizePipelineStages } from '../components/verifier/verifierConstants'
import {
  confirmLectureVerification,
  deleteLecture,
  getLectureDetail,
  getLectureVerifier,
  jobStreamUrl,
  checkHealth,
  retryLecture as retryJobRequest,
} from '../lib/api'
import { isTerminalStatus, markAllStages, mergeStageStatus, phaseFromStatus } from '../lib/jobStreamUtils'

const EMPTY_LECTURE = {
  id: '',
  title: '',
  video_url: '',
  is_verified: false,
  status: '',
}

// graphLec의 useJobStream을 verify 전용으로 축소 이식한 버전.
// 라우터 의존을 없애고, 목록 복귀는 onExit 콜백으로 위임한다.
export function useJobStream(lectureId, { onExit } = {}) {
  const eventSourceRef = useRef(null)
  const activeJobIdRef = useRef(null)
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

  const isVerified = Boolean(lecture.is_verified)

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

  function loadVerifierIfReady(status) {
    if (phaseFromStatus(status) !== PHASES.VERIFY_READY || verifierLoadedRef.current) return
    verifierLoadedRef.current = true
    getLectureVerifier(lectureId)
      .then(result => {
        if (result) setVerifier(result)
        else verifierLoadedRef.current = false
      })
      .catch(() => {
        verifierLoadedRef.current = false
      })
  }

  function connectJob(jobId = '') {
    if (!lectureId) return

    closeEventSource()
    activeJobIdRef.current = jobId || null
    ignoreNextStreamErrorRef.current = false

    const eventSource = new EventSource(jobStreamUrl(lectureId, jobId))
    eventSourceRef.current = eventSource
    startHealthCheck()

    eventSource.onmessage = event => {
      try {
        const payload = JSON.parse(event.data)
        if (payload.error) throw new Error(payload.error)
        if (activeJobIdRef.current && payload.job_id && payload.job_id !== activeJobIdRef.current) return

        const nextPhase = phaseFromStatus(payload.lecture_status)
        setPhase(nextPhase)
        setCurrentStage(payload.current_stage || '')
        setErrorMessage(prev =>
          payload.error_message || (nextPhase === PHASES.ERROR ? payload.current_stage || prev : '')
        )
        setLecture(prev => ({
          ...prev,
          id: prev.id || lectureId,
          job_id: payload.job_id || prev.job_id,
          status: payload.lecture_status || prev.status,
        }))
        setPipelineStages(prev => mergeStageStatus(prev, payload.pipeline_stages || []))

        loadVerifierIfReady(payload.lecture_status)

        if (isTerminalStatus(payload.lecture_status)) {
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

        verifierLoadedRef.current = false
        setLecture({ ...EMPTY_LECTURE, ...detail, status: job.status || '' })
        setVerifier(null)
        setPhase(nextPhase)
        setCurrentStage(job.current_stage || '')
        setErrorMessage(job.error_message || (nextPhase === PHASES.ERROR ? job.current_stage || '' : ''))

        if (Array.isArray(job.pipeline_stages) && job.pipeline_stages.length > 0) {
          setPipelineStages(mergeStageStatus(markAllStages('wait'), job.pipeline_stages))
        } else {
          setPipelineStages(markAllStages(job.status === 'done' ? 'done' : 'wait'))
        }

        loadVerifierIfReady(job.status)
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
      setLecture(prev => ({ ...prev, job_id: result.job_id || prev.job_id, status: 'pending' }))
      setIsMutating(false)
      connectJob(result.job_id)
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

  async function confirmReview() {
    if (!lectureId || isMutating) return
    setIsMutating(true)
    setErrorMessage('')
    try {
      await confirmLectureVerification(lectureId)
      setLecture(prev => ({ ...prev, is_verified: true }))
      setIsMutating(false)
    } catch (error) {
      setIsMutating(false)
      setErrorMessage(String(error?.message || error))
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
    isVerified,
    actions: { restart, remove, confirmReview },
  }
}
