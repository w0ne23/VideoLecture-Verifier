from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import LOCAL_STORAGE_DIR
from app.services.rapid_ocr import rapid_ocr_runtime

router = APIRouter()


class OCRRequest(BaseModel):
    image_path: str = Field(..., description="Path visible inside the backend container")
    lang: str | None = None
    model_dir: str | None = None


@router.get('/ocr/health')
def health() -> dict[str, str]:
    return {'status': 'ok', 'model': 'rapidocr-pp-ocrv5-korean-mobile'}


@router.post('/ocr')
def ocr(req: OCRRequest) -> dict:
    image_path = Path(req.image_path)
    if not image_path.is_absolute():
        image_path = LOCAL_STORAGE_DIR / image_path
    try:
        return rapid_ocr_runtime.infer(image_path, lang=req.lang)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f'image not found: {image_path}') from None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
