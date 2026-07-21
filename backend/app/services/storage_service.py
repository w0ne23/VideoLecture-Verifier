import shutil
import uuid
from pathlib import Path
from fastapi import UploadFile

from app.config import LOCAL_STORAGE_DIR


def lecture_input_dir(lecture_id: uuid.UUID | str) -> Path:
    return LOCAL_STORAGE_DIR / 'inputs' / str(lecture_id)


def lecture_output_dir(lecture_id: uuid.UUID | str) -> Path:
    return LOCAL_STORAGE_DIR / 'results' / str(lecture_id)


def make_file_url(path: str | Path | None) -> str:
    if not path:
        return ''
    raw = Path(path)
    try:
        rel = raw.resolve().relative_to(LOCAL_STORAGE_DIR.resolve())
    except Exception:
        return ''
    return '/files/' + '/'.join(rel.parts)


def save_upload(video: UploadFile, lecture_id: uuid.UUID) -> tuple[Path, Path]:
    input_dir = lecture_input_dir(lecture_id)
    output_dir = lecture_output_dir(lecture_id)
    input_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extension = Path(video.filename or '').suffix or '.mp4'
    input_path = input_dir / f'{lecture_id}{extension}'
    with input_path.open('wb') as handle:
        shutil.copyfileobj(video.file, handle)
    return input_path, output_dir
