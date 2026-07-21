from __future__ import annotations

import threading
from pathlib import Path


class RapidOCRRuntime:
    """Lazily load one shared RapidOCR instance inside the backend process."""

    def __init__(self) -> None:
        self._ocr = None
        self._lock = threading.Lock()

    def _load(self):
        if self._ocr is not None:
            return self._ocr
        with self._lock:
            if self._ocr is None:
                from rapidocr import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

                self._ocr = RapidOCR(params={
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
                })
        return self._ocr

    @staticmethod
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
        return [text[:max_chars].rstrip() + "..."] if len(text) > max_chars else compact

    def infer(self, image_path: Path, *, lang: str | None = None) -> dict:
        if not image_path.exists():
            raise FileNotFoundError(str(image_path))
        result = self._load()(str(image_path), use_det=True, use_cls=True, use_rec=True)
        lines = self._compact_lines(getattr(result, "txts", ()))
        return {
            "image_path": str(image_path),
            "text": "\n".join(lines),
            "lines": lines,
            "model": "rapidocr-pp-ocrv5-korean-mobile",
            "lang": (lang or "korean").strip().lower(),
            "elapsed_sec": float(getattr(result, "elapse", 0.0) or 0.0),
        }


rapid_ocr_runtime = RapidOCRRuntime()
