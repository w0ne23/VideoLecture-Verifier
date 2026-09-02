// 강의 영상 출처 태그 — 업로드 폼 선택지 + 통계 그룹 기준

export const LECTURE_SOURCE_TAGS = [
  { value: 'youtube', label: 'YouTube' },
  { value: 'kmooc', label: 'K-MOOC' },
  { value: 'kocw', label: 'KOCW' },
  { value: 'instructor', label: '교수자 제공' },
  { value: 'etc', label: '기타' },
]

export const LECTURE_SOURCE_TAG_VALUES = LECTURE_SOURCE_TAGS.map(tag => tag.value)

// 태그 값 → 한글 라벨 (미등록 값은 원본, 빈 값은 '—')
export function lectureTagLabel(value) {
  return LECTURE_SOURCE_TAGS.find(tag => tag.value === value)?.label || value || '—'
}
