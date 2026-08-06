import { useRef } from 'react'
import { LECTURE_SOURCE_TAGS } from '../constants/lectureTags'
import { useUploadForm } from '../hooks/useUploadForm'

export default function UploadForm({ onUploaded }) {
  const inputRef = useRef(null)
  const { file, title, sourceTag, errorMessage, isSubmitting, actions } = useUploadForm({ onUploaded })

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
          ? <span>{file.name} ({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
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
      <fieldset className="field tag-fieldset">
        <legend>출처 태그</legend>
        <div className="tag-options" role="radiogroup" aria-label="출처 태그">
          {LECTURE_SOURCE_TAGS.map(tag => (
            <label key={tag.value} className={`tag-option${sourceTag === tag.value ? ' is-selected' : ''}`}>
              <input
                type="radio"
                name="source_tag"
                value={tag.value}
                checked={sourceTag === tag.value}
                onChange={() => actions.setSourceTag(tag.value)}
              />
              {tag.label}
            </label>
          ))}
        </div>
      </fieldset>
      {errorMessage && <p className="error-text">{errorMessage}</p>}
      <div className="button-row">
        <button
          type="button"
          className="btn btn--primary"
          disabled={!file || !sourceTag || isSubmitting}
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
