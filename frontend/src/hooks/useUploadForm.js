import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { uploadLecture } from '../api/pipeline'

// 업로드 화면에서 출처 태그 선택 UI를 없앤 대신, 백엔드가 요구하는 값을 채우기 위해
// 항상 '기타'로 보낸다. 기존 강의의 출처 필터·통계는 그대로 유지된다.
const DEFAULT_SOURCE_TAG = 'etc'

function fileTitle(file) {
  return file?.name ? file.name.replace(/\.[^.]+$/, '') : ''
}

// graphLec의 useUploadForm을 verify 전용으로 축소한 버전.
// 업로드 성공 시 onUploaded(lectureId)로 상세 화면 전환을 위임한다.
export function useUploadForm({ onUploaded } = {}) {
  const [file, setFile] = useState(null)
  const [title, setTitle] = useState('')
  const [isTitleManual, setIsTitleManual] = useState(false)
  const [localError, setLocalError] = useState('')

  const mutation = useMutation({
    mutationFn: uploadLecture,
    onSuccess: created => {
      setFile(null)
      setTitle('')
      setIsTitleManual(false)
      setLocalError('')
      onUploaded?.(created.id)
    },
  })

  function selectFile(nextFile) {
    if (!nextFile) return
    setFile(nextFile)
    setLocalError('')
    // 사용자가 제목을 직접 수정한 적이 없을 때만 파일명으로 자동 갱신한다.
    // (그냥 비어있는지만 보면, 이전 파일에서 자동 채워진 제목이 새 파일을 골라도 그대로 남는다.)
    setTitle(prev => (isTitleManual && prev.trim() ? prev : fileTitle(nextFile)))
  }

  function setTitleManual(value) {
    setIsTitleManual(true)
    setTitle(value)
  }

  function submit() {
    if (!file || mutation.isPending) return
    setLocalError('')
    mutation.mutate({
      file,
      title: title.trim() || fileTitle(file),
      sourceTag: DEFAULT_SOURCE_TAG,
    })
  }

  function reset() {
    setFile(null)
    setTitle('')
    setIsTitleManual(false)
    setLocalError('')
    mutation.reset()
  }

  const mutationError = mutation.error ? String(mutation.error?.message || mutation.error) : ''

  return {
    file,
    title,
    errorMessage: localError || mutationError,
    isSubmitting: mutation.isPending,
    actions: { selectFile, setTitle: setTitleManual, submit, reset },
  }
}
