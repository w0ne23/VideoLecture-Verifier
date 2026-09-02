# RapidOCR 기반 슬라이드 이미지 텍스트 인식 API
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import LOCAL_STORAGE_DIR
from app.services.rapid_ocr import rapid_ocr_runtime

router = APIRouter()


class OCRRequest(BaseModel):
    image_path: str = Field(..., description="백엔드 컨테이너 내부 기준 경로")
    lang: str | None = None
    model_dir: str | None = None


# OCR 런타임 상태 확인
@router.get('/ocr/health')
def health() -> dict[str, str]:
    return {'status': 'ok', 'model': 'rapidocr-pp-ocrv5-korean-mobile'}


# 이미지 경로를 받아 OCR 실행, 상대경로면 LOCAL_STORAGE_DIR 기준으로 해석
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
