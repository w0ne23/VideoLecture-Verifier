from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

LOCAL_STORAGE_DIR = Path(os.getenv("LOCAL_STORAGE_DIR", "/app/storage"))


class OCRRequest(BaseModel):
    image_path: str = Field(..., description="Path visible inside the OCR container")
    lang: str | None = None
    model_dir: str | None = None


class OCRResponse(BaseModel):
    image_path: str
    text: str
    lines: list[str]
    model: str = "rapidocr-pp-ocrv5-korean-mobile"
    lang: str
    elapsed_sec: float | None = None


def _compact_lines(lines: object, *, max_lines: int = 80, max_chars: int = 6000) -> list[str]:
    compact: list[str] = []
    previous = ""
    for value in lines if isinstance(lines, (list, tuple)) else ():
        line = " ".join(str(value).split()).strip()
        if not line or line == previous:
            continue
        compact.append(line)
        previous = line
        if len(compact) >= max_lines:
            compact.append("...")
            break
    text = "\n".join(compact)
    if len(text) > max_chars:
        return [text[:max_chars].rstrip() + "..."]
    return compact


class RapidOCRRuntime:
    def __init__(self) -> None:
        self._ocr = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._ocr is not None:
            return
        with self._lock:
            if self._ocr is not None:
                return
            from rapidocr import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

            # PP-OCRv5 has an explicit Korean recognizer while its detector is
            # the shared Chinese/multilingual mobile detector.  ONNX Runtime is
            # the portable engine for Linux CPU and macOS Docker Desktop.
            params = {
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.lang_type": LangDet.CH,
                "Det.model_type": ModelType.MOBILE,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Cls.engine_type": EngineType.ONNXRUNTIME,
                "Cls.lang_type": LangCls.CH,
                "Cls.model_type": ModelType.MOBILE,
                "Cls.ocr_version": OCRVersion.PPOCRV5,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.lang_type": LangRec.KOREAN,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
            }
            log.info("Loading RapidOCR PP-OCRv5 Korean mobile (ONNX Runtime)")
            self._ocr = RapidOCR(params=params)

    def infer(self, image_path: Path, lang: str | None = None) -> OCRResponse:
        if not image_path.exists():
            raise FileNotFoundError(str(image_path))
        self.load()
        result = self._ocr(str(image_path), use_det=True, use_cls=True, use_rec=True)
        lines = _compact_lines(getattr(result, "txts", ()))
        return OCRResponse(
            image_path=str(image_path),
            text="\n".join(lines),
            lines=lines,
            lang=(lang or "korean").strip().lower(),
            elapsed_sec=float(getattr(result, "elapse", 0.0) or 0.0),
        )


runtime = RapidOCRRuntime()


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOCAL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="VeriLec RapidOCR Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "rapidocr-pp-ocrv5-korean-mobile"}


@app.post("/ocr", response_model=OCRResponse)
def ocr(req: OCRRequest) -> OCRResponse:
    image_path = Path(req.image_path)
    if not image_path.is_absolute():
        image_path = LOCAL_STORAGE_DIR / image_path
    try:
        return runtime.infer(image_path, lang=req.lang)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"image not found: {image_path}")
    except Exception as exc:
        log.exception("RapidOCR inference failed for %s", image_path)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
