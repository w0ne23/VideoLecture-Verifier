// 강의자가 오류 항목별로 남기는 동의/중립/비동의 평가 — 조회 + 저장(upsert)
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { getLectureReviews, saveLectureReview } from '../api/pipeline'

export function useLectureReviews(lectureId) {
  const queryClient = useQueryClient()
  const queryKey = ['lecture-reviews', lectureId]

  const query = useQuery({
    queryKey,
    queryFn: () => getLectureReviews(lectureId),
    enabled: !!lectureId,
    staleTime: Infinity,
  })

  const mutation = useMutation({
    mutationFn: ({ itemId, rating }) => saveLectureReview(lectureId, itemId, rating),
    onSuccess: (_result, { itemId, rating }) => {
      queryClient.setQueryData(queryKey, prev => ({ ...(prev || {}), [itemId]: rating }))
    },
  })

  return {
    reviews: query.data || {},
    isLoading: query.isLoading,
    saveReview: (itemId, rating) => mutation.mutate({ itemId, rating }),
    // 현재 저장 요청 중인 항목만 pending 취급 (다른 항목 버튼은 계속 조작 가능)
    isSaving: itemId => mutation.isPending && mutation.variables?.itemId === itemId,
  }
}
