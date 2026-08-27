import { useEffect, useRef, useState } from 'react'

// isRunning이 true가 될 때마다 0부터 다시 재고, false가 되면 그 시점 값에서 멈춘다.
export function useElapsedStopwatch(isRunning) {
  const startedAtRef = useRef(0)
  const [elapsedMs, setElapsedMs] = useState(0)

  useEffect(() => {
    if (!isRunning) return undefined

    startedAtRef.current = Date.now()
    setElapsedMs(0)
    const tick = () => setElapsedMs(Date.now() - startedAtRef.current)
    const intervalId = setInterval(tick, 1000)
    return () => clearInterval(intervalId)
  }, [isRunning])

  return elapsedMs
}
