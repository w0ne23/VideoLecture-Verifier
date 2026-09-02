# 환경 변수 기반 전역 설정 값 정의
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
root_env = PROJECT_ROOT / '.env'
# 프로젝트 루트에 .env가 있으면 그 파일을 사용, 없으면 기본 탐색 경로 사용
if root_env.exists():
    load_dotenv(root_env)
else:
    load_dotenv()

# 파이프라인 산출물, 업로드 영상 등을 저장하는 로컬 경로
LOCAL_STORAGE_DIR = Path(os.getenv('LOCAL_STORAGE_DIR', str(PROJECT_ROOT / 'storage'))).resolve()
# 검증 파이프라인 스크립트가 위치한 루트 경로
PIPELINE_ROOT = Path(os.getenv('PIPELINE_ROOT', str(PROJECT_ROOT))).resolve()
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+asyncpg://user:pass@localhost/verilec')
# 프론트엔드 개발 서버 등 CORS 허용 origin 목록
CORS_ORIGINS = [
    item.strip()
    for item in os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000').split(',')
    if item.strip()
]
# 백엔드 워커 프로세스 개수, 최소 1
BACKEND_WORKERS = max(1, int(os.getenv('VLVERIFIER_BACKEND_WORKERS', '1') or '1'))
# LiteLLM 모델 카탈로그 조회 URL
LITELLM_MODEL_CATALOG_URL = os.getenv(
    'LITELLM_MODEL_CATALOG_URL',
    'https://api.litellm.ai/model_catalog',
).rstrip('/')
