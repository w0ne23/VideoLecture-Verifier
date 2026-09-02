import { useEffect, useState } from 'react'

// startedAtMs(작업이 실제로 시작된 서버 타임스탬프)가 있으면 그걸 기준으로 경과 시간을
// 계산한다 — 새로고침해도 컴포넌트가 다시 마운트될 뿐 작업 시작 시각은 그대로라서,
// 시계가 0으로 초기화되지 않고 실제 경과 시간을 이어서 보여준다. startedAtMs가 없으면
// (예: 서버 타임스탬프를 아직 못 받아온 순간) 지금 이 순간을 기준으로 임시로 잰다.
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
