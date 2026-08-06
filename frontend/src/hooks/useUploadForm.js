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
  const [sourceTag, setSourceTag] = useState('')
  const [localError, setLocalError] = useState('')

  const mutation = useMutation({
    mutationFn: uploadLecture,
    onSuccess: created => {
      setFile(null)
      setTitle('')
      setSourceTag('')
      setLocalError('')
      onUploaded?.(created.id)
    },
  })

  function selectFile(nextFile) {
    if (!nextFile) return
    setFile(nextFile)
    setLocalError('')
    setTitle(prev => (prev.trim() ? prev : fileTitle(nextFile)))
  }

  function submit() {
    if (!file || mutation.isPending) return
    if (!sourceTag) {
      setLocalError('출처 태그를 선택해 주세요.')
      return
    }
    setLocalError('')
    mutation.mutate({
      file,
      title: title.trim() || fileTitle(file),
      sourceTag,
    })
  }

  function reset() {
    setFile(null)
    setTitle('')
    setSourceTag('')
    setLocalError('')
    mutation.reset()
  }

  const mutationError = mutation.error ? String(mutation.error?.message || mutation.error) : ''

  return {
    file,
    title,
    sourceTag,
    errorMessage: localError || mutationError,
    isSubmitting: mutation.isPending,
    actions: { selectFile, setTitle, setSourceTag, submit, reset },
  }
}
