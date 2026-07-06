import { useState } from 'react'
import { uploadLecture } from '../lib/api'

function fileTitle(file) {
  return file?.name ? file.name.replace(/\.[^.]+$/, '') : ''
}

// graphLec의 useUploadForm을 verify 전용으로 축소한 버전.
// 업로드 성공 시 onUploaded(lectureId)로 상세 화면 전환을 위임한다.
export function useUploadForm({ onUploaded } = {}) {
  const [file, setFile] = useState(null)
  const [title, setTitle] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  function selectFile(nextFile) {
    if (!nextFile) return
    setFile(nextFile)
    setTitle(prev => (prev.trim() ? prev : fileTitle(nextFile)))
  }

  async function submit() {
    if (!file || isSubmitting) return

    setErrorMessage('')
    setIsSubmitting(true)
    try {
      const created = await uploadLecture({
        file,
        title: title.trim() || fileTitle(file),
      })
      setFile(null)
      setTitle('')
      setIsSubmitting(false)
      onUploaded?.(created.id)
    } catch (error) {
      setIsSubmitting(false)
      setErrorMessage(String(error?.message || error))
    }
  }

  function reset() {
    setFile(null)
    setTitle('')
    setErrorMessage('')
    setIsSubmitting(false)
  }

  return {
    file,
    title,
    errorMessage,
    isSubmitting,
    actions: { selectFile, setTitle, submit, reset },
  }
}
