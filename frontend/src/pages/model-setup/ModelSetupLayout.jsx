// /model-setup 중첩 라우트의 레이아웃 (현재는 자식 라우트를 그대로 렌더)

import { Outlet } from 'react-router-dom'

export default function ModelSetupLayout() {
  return <Outlet />
}
