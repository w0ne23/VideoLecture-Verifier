import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
root_env = PROJECT_ROOT / '.env'
if root_env.exists():
    load_dotenv(root_env)
else:
    load_dotenv()

LOCAL_STORAGE_DIR = Path(os.getenv('LOCAL_STORAGE_DIR', str(PROJECT_ROOT / 'storage'))).resolve()
PIPELINE_ROOT = Path(os.getenv('PIPELINE_ROOT', str(PROJECT_ROOT))).resolve()
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+asyncpg://user:pass@localhost/verilec')
CORS_ORIGINS = [
    item.strip()
    for item in os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000').split(',')
    if item.strip()
]
BACKEND_WORKERS = max(1, int(os.getenv('VERILEC_BACKEND_WORKERS', '1') or '1'))
