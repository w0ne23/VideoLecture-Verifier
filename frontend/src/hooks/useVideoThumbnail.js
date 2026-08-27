import { useEffect, useState } from 'react'

// 선택된 영상 파일에서 프레임을 하나 캡처해 썸네일 데이터 URL을 만든다.
// loadeddata 시점에 첫 프레임을 우선 캡처해두고(항상 무언가는 뜨도록),
// 가능하면 좀 더 보기 좋은 중간 지점 프레임으로 교체한다.
export function useVideoThumbnail(file) {
  const [thumbnailUrl, setThumbnailUrl] = useState('')

  useEffect(() => {
    if (!file) {
      setThumbnailUrl('')
      return undefined
    }

    let cancelled = false
    let capturedOnce = false
    const objectUrl = URL.createObjectURL(file)
    const video = document.createElement('video')
    video.muted = true
    video.playsInline = true
    video.preload = 'auto'
    video.style.position = 'fixed'
    video.style.width = '1px'
    video.style.height = '1px'
    video.style.opacity = '0'
    video.style.pointerEvents = 'none'
    video.src = objectUrl
    document.body.appendChild(video)

    const capture = () => {
      if (cancelled || !video.videoWidth) return
      const canvas = document.createElement('canvas')
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)
      setThumbnailUrl(canvas.toDataURL('image/jpeg', 0.85))
    }

    const cleanup = () => {
      video.removeEventListener('loadeddata', onLoadedData)
      video.removeEventListener('seeked', onSeeked)
      video.removeEventListener('error', onError)
      video.remove()
      URL.revokeObjectURL(objectUrl)
    }

    function onLoadedData() {
      capture()
      capturedOnce = true
      const seekTime = Math.min(1, (video.duration || 2) / 2)
      if (seekTime > 0) {
        video.currentTime = seekTime
      }
    }

    function onSeeked() {
      capture()
      cleanup()
    }

    function onError() {
      if (!capturedOnce) setThumbnailUrl('')
      cleanup()
    }

    video.addEventListener('loadeddata', onLoadedData)
    video.addEventListener('seeked', onSeeked)
    video.addEventListener('error', onError)

    return () => {
      cancelled = true
      cleanup()
    }
  }, [file])

  return thumbnailUrl
}
