# 백엔드 전체 헬스체크
from fastapi import APIRouter

router = APIRouter()


# 서버 기동 여부 확인
@router.get('/health')
def health_check():
    return {'status': 'ok'}
