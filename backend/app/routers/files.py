# 파일 메타데이터 관련 헬스체크
from fastapi import APIRouter

router = APIRouter(prefix='/files-meta')


# files-meta 라우터 상태 확인
@router.get('/health')
def files_health():
    return {'status': 'ok'}
