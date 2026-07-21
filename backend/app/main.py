import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import CORS_ORIGINS, LOCAL_STORAGE_DIR
from app.lifecycle import lifespan
from app.routers import files, health, jobs, lectures, ocr, results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

app = FastAPI(title='VeriLec API', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

LOCAL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount('/files', StaticFiles(directory=str(LOCAL_STORAGE_DIR)), name='files')

app.include_router(health.router)
app.include_router(jobs.router)
app.include_router(lectures.router)
app.include_router(results.router)
app.include_router(files.router)
app.include_router(ocr.router)
