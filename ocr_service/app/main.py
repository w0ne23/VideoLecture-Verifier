from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

LOCAL_STORAGE_DIR = Path(os.getenv("LOCAL_STORAGE_DIR", "/app/storage"))
OCR_LANG = os.getenv("VLVERIFIER_SLIDE_OCR_LANG", "multilingual").strip().lower()
OCR_MODEL_DIR = os.getenv("VLVERIFIER_SLIDE_OCR_MODEL_DIR", "").strip()


class OCRRequest(BaseModel):
    image_path: str = Field(..., description="Path visible inside the OCR container")
    lang: str | None = None
    model_dir: str | None = None


class OCRResponse(BaseModel):
    image_path: str
    text: str
    lines: list[str]
    model: str = "nemotron-ocr-v2"
    lang: str


def _normalize_lines(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, (list, tuple)):
        lines: list[str] = []
        for item in value:
            lines.extend(_normalize_lines(item))
        return lines
    if isinstance(value, dict):
        lines: list[str] = []
        for key in ("lines", "text", "ocr_txts", "texts", "paragraphs", "content"):
            if key in value:
                lines.extend(_normalize_lines(value.get(key)))
        return lines
    for attr in ("lines", "text", "ocr_txts", "texts", "paragraphs", "content"):
        if hasattr(value, attr):
            return _normalize_lines(getattr(value, attr))
    if hasattr(value, "to_dict"):
        try:
            return _normalize_lines(value.to_dict())
        except Exception:
            return []
    text = str(value).strip()
    return [text] if text else []


def _compact_lines(lines: list[str], *, max_lines: int = 80, max_chars: int = 6000) -> list[str]:
    compact: list[str] = []
    prev = ""
    for line in lines:
        line = " ".join(str(line).split()).strip()
        if not line or line == prev:
            continue
        compact.append(line)
        prev = line
        if len(compact) >= max_lines:
            compact.append("...")
            break
    if len("\n".join(compact)) > max_chars:
        joined = "\n".join(compact)
        return [joined[:max_chars].rstrip() + "..."]
    return compact


class OCRRuntime:
    def __init__(self) -> None:
        self.ocr = None
        self.ready = False
        self.loaded_lang = OCR_LANG
        self.loaded_model_dir = OCR_MODEL_DIR

    def load(self, *, lang: str | None = None, model_dir: str | None = None) -> None:
        requested_lang = (lang or OCR_LANG).strip().lower()
        requested_model_dir = (model_dir or OCR_MODEL_DIR).strip()
        if self.ready and requested_lang == self.loaded_lang and requested_model_dir == self.loaded_model_dir:
            return
        try:
            from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
        except Exception as exc:
            raise RuntimeError(f"Nemotron OCR package unavailable: {exc}") from exc

        kwargs: dict[str, Any] = {}
        if requested_model_dir:
            kwargs["model_dir"] = requested_model_dir
        if requested_lang in {"en", "english"}:
            kwargs["lang"] = "en"

        log.info("Loading Nemotron OCR v2 (lang=%s, model_dir=%s)", kwargs.get("lang", requested_lang), requested_model_dir or "<hub>")
        self.ocr = NemotronOCRV2(**kwargs)
        self.ready = True
        self.loaded_lang = requested_lang
        self.loaded_model_dir = requested_model_dir

    def infer(self, image_path: Path, lang: str | None = None, model_dir: str | None = None) -> OCRResponse:
        if not image_path.exists():
            raise FileNotFoundError(str(image_path))
        self.load(lang=lang, model_dir=model_dir)

        ocr = self.ocr
        raw = None
        image_arg = str(image_path)
        if hasattr(ocr, "predict") and callable(getattr(ocr, "predict")):
            raw = ocr.predict(image_arg)
        elif callable(ocr):
            raw = ocr(image_arg)
        else:
            raise RuntimeError("Nemotron OCR runtime does not expose predict/call")

        lines = _compact_lines(_normalize_lines(raw))
        text = "\n".join(lines).strip()
        return OCRResponse(
            image_path=str(image_path),
            text=text,
            lines=lines,
            lang=(lang or OCR_LANG).strip().lower(),
        )


runtime = OCRRuntime()


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOCAL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="VLVerifier OCR Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ocr", response_model=OCRResponse)
def ocr(req: OCRRequest) -> OCRResponse:
    image_path = Path(req.image_path)
    if not image_path.is_absolute():
        image_path = LOCAL_STORAGE_DIR / req.image_path
    try:
        return runtime.infer(image_path, lang=req.lang, model_dir=req.model_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"image not found: {image_path}")
    except Exception as exc:
        log.exception("OCR inference failed for %s", image_path)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
