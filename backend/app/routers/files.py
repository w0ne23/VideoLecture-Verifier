from fastapi import APIRouter

router = APIRouter(prefix='/files-meta')


@router.get('/health')
def files_health():
    return {'status': 'ok'}
