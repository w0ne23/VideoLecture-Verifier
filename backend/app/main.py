import logging
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Try to load .env from project root (../../.env) if it exists, or from the current directory
root_env = Path(__file__).resolve().parent.parent.parent / ".env"
if root_env.exists():
    load_dotenv(dotenv_path=root_env)
else:
    load_dotenv()

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.lifecycle import lifespan, LOCAL_STORAGE_DIR
from app.routers import jobs, results, recommend

app = FastAPI(lifespan=lifespan)

# CORS configuration: Load from environment variable CORS_ORIGINS
# Defaults to local dev environments
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount local storage for serving result files
app.mount("/files", StaticFiles(directory=LOCAL_STORAGE_DIR), name="files")

app.include_router(jobs.router)
app.include_router(results.router)
app.include_router(recommend.router)

@app.get("/health")
def health_check():
    return {"status": "ok"}
