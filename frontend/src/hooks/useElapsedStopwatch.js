import { useEffect, useState } from 'react'

// 경과 시간 스톱워치 (1초 단위 갱신, ms 반환)
// startedAtMs(작업이 실제 시작된 서버 타임스탬프)가 있으면 그 기준으로 경과 시간 계산 —
// 새로고침으로 컴포넌트가 재마운트돼도 시작 시각은 그대로라 시계가 0으로 리셋되지 않음
// startedAtMs 가 없으면(서버 타임스탬프 수신 전) 현재 시각 기준으로 임시 측정
export function useElapsedStopwatch(isRunning, startedAtMs) {
  const [elapsedMs, setElapsedMs] = useState(0)

  useEffect(() => {
    if (!isRunning) return undefined

    const anchor = startedAtMs ?? Date.now()
    const tick = () => setElapsedMs(Date.now() - anchor)
    tick()
    const intervalId = setInterval(tick, 1000)
    return () => clearInterval(intervalId)
  }, [isRunning, startedAtMs])

  return elapsedMs
}
