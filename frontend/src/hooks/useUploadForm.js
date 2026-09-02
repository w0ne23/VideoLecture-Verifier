// 강의 업로드 폼 상태 관리 + 제출 (react-query mutation)

import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { uploadLecture } from '../api/pipeline'

// 업로드 화면에 출처 태그 선택 UI 가 없어, 백엔드 필수값을 채우려고 항상 '기타'로 전송
// 기존 강의의 출처 필터·통계는 그대로 유지
const DEFAULT_SOURCE_TAG = 'etc'

// 파일명에서 확장자를 뗀 문자열
function fileTitle(file) {
  return file?.name ? file.name.replace(/\.[^.]+$/, '') : ''
}

// 업로드 성공 시 onUploaded(lectureId)로 상세 화면 전환 위임
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
    // 사용자가 제목을 직접 수정한 적 없을 때만 파일명으로 자동 갱신
    // (빈 값 여부만 보면, 이전 파일에서 자동 채워진 제목이 새 파일 선택 후에도 남음)
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
