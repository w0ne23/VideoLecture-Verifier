# VLVerifier 백엔드 FastAPI 애플리케이션 엔트리포인트
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import CORS_ORIGINS, LOCAL_STORAGE_DIR
from app.lifecycle import lifespan
from app.routers import files, health, lectures, llm_catalog, llm_credentials, model_settings, ocr, stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

app = FastAPI(title='VLVerifier API', lifespan=lifespan)

# 프론트엔드 개발 서버 등 허용된 origin에서의 요청만 통과
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# 파이프라인 산출물, 업로드 영상 등을 정적 파일로 /files 경로에 노출
LOCAL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount('/files', StaticFiles(directory=str(LOCAL_STORAGE_DIR)), name='files')

# 기능별 라우터 등록
app.include_router(health.router)
app.include_router(lectures.router)
app.include_router(files.router)
app.include_router(ocr.router)
app.include_router(model_settings.router)
app.include_router(llm_catalog.router)
app.include_router(llm_credentials.router)
app.include_router(stats.router)
