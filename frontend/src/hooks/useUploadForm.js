import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { uploadLecture } from '../api/pipeline'

function fileTitle(file) {
  return file?.name ? file.name.replace(/\.[^.]+$/, '') : ''
}

// graphLec의 useUploadForm을 verify 전용으로 축소한 버전.
// 업로드 성공 시 onUploaded(lectureId)로 상세 화면 전환을 위임한다.
export function useUploadForm({ onUploaded } = {}) {
  const [file, setFile] = useState(null)
  const [title, setTitle] = useState('')

  const mutation = useMutation({
    mutationFn: uploadLecture,
    onSuccess: created => {
      setFile(null)
      setTitle('')
      onUploaded?.(created.id)
    },
  })

  function selectFile(nextFile) {
    if (!nextFile) return
    setFile(nextFile)
    setTitle(prev => (prev.trim() ? prev : fileTitle(nextFile)))
  }

  function submit() {
    if (!file || mutation.isPending) return
    mutation.mutate({ file, title: title.trim() || fileTitle(file) })
  }

  function reset() {
    setFile(null)
    setTitle('')
    mutation.reset()
  }

  return {
    file,
    title,
    errorMessage: mutation.error ? String(mutation.error?.message || mutation.error) : '',
    isSubmitting: mutation.isPending,
    actions: { selectFile, setTitle, submit, reset },
  }
}
