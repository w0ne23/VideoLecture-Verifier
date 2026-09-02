import { useEffect, useRef, useState } from 'react'
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

// 검증 job 진행 상태 훅 — 초기 상태 로드 + SSE 스트림 구독 + 재시작/삭제 액션
// verify 워크플로우 전용, 라우터 의존 없이 목록 복귀는 onExit 콜백에 위임
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

  // 헬스체크 타이머 정지
  function stopHealthCheck() {
    if (!healthTimerRef.current) return
    clearInterval(healthTimerRef.current)
    healthTimerRef.current = null
  }

  // SSE 커넥션 종료 + 헬스체크 정지
  // ignoreStreamError: 뒤이어 발생할 EventSource onerror 를 한 번 무시 (의도적 종료 구분용)
  function closeEventSource({ ignoreStreamError = false } = {}) {
    if (eventSourceRef.current) {
      if (ignoreStreamError) ignoreNextStreamErrorRef.current = true
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
    stopHealthCheck()
  }

  // 헬스체크 실패(서버 다운) 처리 — 연결 종료 후 에러 phase 로 전환
  function handleServerDown() {
    closeEventSource({ ignoreStreamError: true })
    setCurrentStage('서버와의 연결이 끊어졌습니다.')
    setErrorMessage('서버와의 연결이 끊어졌습니다.')
    setPhase(PHASES.ERROR)
  }

  // SSE 는 서버가 죽어도 즉시 에러를 내지 않으므로, 3초마다 /health 로 생존 확인
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

  // done 도달 시 한 번만 GET /lectures/{id}/result 를 호출해 최종 결과 로드
  function loadVerificationIfReady(status) {
    if (phaseFromStatus(status) !== PHASES.VERIFY_READY || verifierLoadedRef.current) return
    verifierLoadedRef.current = true
    getLectureResult(lectureId)
      .then(result => setVerifier(result))
      .catch(() => {
        verifierLoadedRef.current = false
      })
  }

  // SSE 연결 + 이벤트 핸들러 등록
  // onmessage 마다 phase/스테이지/에러/강의 상태를 갱신하고, 종료 상태면 연결 정리
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

        // verify 완료 → 통계 페이지가 열려 있으면 자동 갱신
        // 백엔드가 done 표시 전에 verification_stats 를 적재하므로 여기서 행이 보장됨
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

  // lectureId 변경 시: 상세 조회로 초기 상태를 채우고, 진행 중이면 SSE 연결
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

        // lectureId 가 바뀌면 이전 강의의 result 가 남지 않도록 초기화
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

  // 검증 재실행 — 새 job 생성 후 SSE 재연결
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
      // 새 job 이 방금 시작됐으므로 경과 시간 기준점(job.created_at)도 현재 시각으로 갱신 —
      // 갱신하지 않으면 스톱워치가 실패한 이전 시도의 경과 시간을 이어받음
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

  // 강의 + 산출물 삭제 후 onExit 로 목록 복귀
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
