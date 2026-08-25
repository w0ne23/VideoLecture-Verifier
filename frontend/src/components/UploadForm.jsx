import { useEffect, useRef, useState } from 'react'
import { useUploadForm } from '../hooks/useUploadForm'

export default function UploadForm({ onUploaded }) {
  const inputRef = useRef(null)
  const { file, title, errorMessage, isSubmitting, actions } = useUploadForm({ onUploaded })
  const [thumbnailUrl, setThumbnailUrl] = useState('')

  function onDrop(event) {
    event.preventDefault()
    actions.selectFile(event.dataTransfer.files?.[0])
  }

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
      // 첫 프레임을 즉시 캡처해두고, 가능하면 좀 더 보기 좋은 중간 지점 프레임으로 교체한다.
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

  return (
    <div className="upload-card">
      <h2>강의 영상 업로드</h2>
      <div
        className={`dropzone${file ? ' dropzone--filled' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={event => event.preventDefault()}
        onDrop={onDrop}
      >
        {file
          ? (
            <div className="dropzone-preview">
              {thumbnailUrl && <img className="dropzone-thumbnail" src={thumbnailUrl} alt="" />}
              <span className="dropzone-filename">{file.name} ({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
            </div>
          )
          : <span>클릭하거나 영상 파일을 끌어다 놓으세요 (.mp4)</span>}
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          hidden
          onChange={event => actions.selectFile(event.target.files?.[0])}
        />
      </div>
      <label className="field">
        <span>강의 제목</span>
        <input
          type="text"
          value={title}
          placeholder="미입력 시 파일명 사용"
          onChange={event => actions.setTitle(event.target.value)}
        />
      </label>
      {errorMessage && <p className="error-text">{errorMessage}</p>}
      <div className="button-row">
        <button
          type="button"
          className="btn btn--primary"
          disabled={!file || isSubmitting}
          onClick={actions.submit}
        >
          {isSubmitting ? '업로드 중...' : '검증 시작'}
        </button>
        {file && (
          <button type="button" className="btn" disabled={isSubmitting} onClick={actions.reset}>
            초기화
          </button>
        )}
      </div>
    </div>
  )
}
