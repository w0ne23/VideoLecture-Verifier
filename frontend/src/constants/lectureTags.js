export const LECTURE_SOURCE_TAGS = [
  { value: 'youtube', label: 'YouTube' },
  { value: 'kmooc', label: 'K-MOOC' },
  { value: 'kocw', label: 'KOCW' },
  { value: 'instructor', label: '교수자 제공' },
  { value: 'etc', label: '기타' },
]

export const LECTURE_SOURCE_TAG_VALUES = LECTURE_SOURCE_TAGS.map(tag => tag.value)

export function lectureTagLabel(value) {
  return LECTURE_SOURCE_TAGS.find(tag => tag.value === value)?.label || value || '—'
}
