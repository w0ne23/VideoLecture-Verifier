import { useRef } from 'react'
import { useUploadForm } from '../hooks/useUploadForm'
import { useVideoThumbnail } from '../hooks/useVideoThumbnail'

export default function UploadForm({ onUploaded }) {
  const inputRef = useRef(null)
  const { file, title, errorMessage, isSubmitting, actions } = useUploadForm({ onUploaded })
  const thumbnailUrl = useVideoThumbnail(file)

  function onDrop(event) {
    event.preventDefault()
    actions.selectFile(event.dataTransfer.files?.[0])
  }

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
      <div className="upload-title-row">
        <label className="field upload-title-field">
          <span>강의 제목</span>
          <input
            type="text"
            value={title}
            placeholder="미입력 시 파일명 사용"
            onChange={event => actions.setTitle(event.target.value)}
          />
        </label>
        <button
          type="button"
          className="btn btn--primary upload-title-submit"
          disabled={!file || isSubmitting}
          onClick={actions.submit}
        >
          {isSubmitting ? '업로드 중...' : '검증 시작'}
        </button>
      </div>
      {errorMessage && <p className="error-text">{errorMessage}</p>}
    </div>
  )
}
