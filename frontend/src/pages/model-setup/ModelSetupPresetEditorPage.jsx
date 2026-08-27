import { useMemo } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import ModelSetupConsole from '../../components/model-setup/ModelSetupConsole'
import {
  createModelSettingProfile,
  getModelSettingProfile,
  updateModelSettingProfile,
} from '../../api/modelSetupProfiles'

export default function ModelSetupPresetEditorPage({ mode }) {
  const navigate = useNavigate()
  const { profileId } = useParams()
  const queryClient = useQueryClient()
  const isEdit = mode === 'edit'

  const profileQuery = useQuery({
    queryKey: ['model-setting-profile', profileId],
    queryFn: () => getModelSettingProfile(profileId),
    enabled: isEdit && Boolean(profileId),
  })

  const mutation = useMutation({
    mutationFn: payload => (
      isEdit
        ? updateModelSettingProfile(profileId, payload)
        : createModelSettingProfile(payload)
    ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['model-setting-profiles'] })
    },
  })

  const initialData = useMemo(() => ({
    name: profileQuery.data?.name || '',
    editorState: profileQuery.data?.editor_state || null,
  }), [profileQuery.data])

  if (isEdit && profileQuery.isLoading) {
    return (
      <section className="model-setup">
        <div className="ms-card"><p className="ms-empty">프리셋을 불러오는 중이에요…</p></div>
      </section>
    )
  }

  if (isEdit && profileQuery.error) {
    return (
      <section className="model-setup">
        <div className="ms-card"><p className="ms-save-error">{String(profileQuery.error.message || profileQuery.error)}</p></div>
      </section>
    )
  }

  return (
    <ModelSetupConsole
      headerTitle={isEdit ? '프리셋 수정' : '새 프리셋 만들기'}
      initialName={initialData.name}
      initialEditorState={initialData.editorState}
      onBack={() => navigate(isEdit ? '/model-setup/presets' : '/model-setup')}
      confirmMessage="저장하시겠습니까?"
      confirmActionLabel="확인"
      cancelActionLabel="취소"
      showPostSaveDialog={!isEdit}
      postSaveMessage="저장 완료했습니다. 프리셋을 확인해 보시겠습니까?"
      isSubmitting={mutation.isPending}
      errorMessage={mutation.error ? String(mutation.error.message || mutation.error) : ''}
      onRequestSubmit={async payload => {
        await mutation.mutateAsync(payload)
        if (isEdit) navigate('/model-setup/presets')
      }}
      onGoList={() => navigate('/model-setup/presets')}
      onGoHome={() => navigate('/')}
    />
  )
}
