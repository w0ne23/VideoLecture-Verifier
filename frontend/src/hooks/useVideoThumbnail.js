import { useEffect, useState } from 'react'

// 선택된 영상 파일에서 프레임 하나를 캡처해 썸네일 데이터 URL 생성
// loadeddata 시점에 첫 프레임을 먼저 캡처(항상 무언가는 표시되도록)하고,
// 가능하면 더 보기 좋은 중간 지점 프레임으로 교체
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
    // 화면에 보이지 않는 1px 비디오 엘리먼트로 디코딩만 수행
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

    // 현재 프레임을 canvas 로 옮겨 JPEG 데이터 URL 로 변환
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

    // 첫 프레임 캡처 후 중간 지점(최대 1초)으로 seek
    function onLoadedData() {
      capture()
      capturedOnce = true
      const seekTime = Math.min(1, (video.duration || 2) / 2)
      if (seekTime > 0) {
        video.currentTime = seekTime
      }
    }

    // seek 완료된 프레임으로 교체하고 정리
    function onSeeked() {
      capture()
      cleanup()
    }

    // 디코딩 실패 — 첫 프레임도 못 잡았으면 빈 값
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
