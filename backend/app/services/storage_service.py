# 업로드 영상과 파이프라인 산출물의 저장 경로 관리
import shutil
import uuid
from pathlib import Path
from fastapi import UploadFile

from app.config import LOCAL_STORAGE_DIR


# 강의 업로드 원본 저장 디렉터리 경로
def lecture_input_dir(lecture_id: uuid.UUID | str) -> Path:
    return LOCAL_STORAGE_DIR / 'inputs' / str(lecture_id)


# 강의 파이프라인 산출물 저장 디렉터리 경로
def lecture_output_dir(lecture_id: uuid.UUID | str) -> Path:
    return LOCAL_STORAGE_DIR / 'results' / str(lecture_id)


def storage_relpath(path: str | Path) -> str:
    """LOCAL_STORAGE_DIR 기준 상대경로 문자열, DB에는 이 형태로 저장"""
    raw = Path(path)
    if not raw.is_absolute():
        return str(raw)
    return str(raw.resolve().relative_to(LOCAL_STORAGE_DIR.resolve()))


def resolve_storage_path(path_value: str | Path | None) -> Path | None:
    """DB에 저장된 스토리지 경로를 현재 환경의 절대경로로 해석

    - 상대경로('inputs/…' 등)는 LOCAL_STORAGE_DIR 기준으로 결합
    - 절대경로인데 파일이 없으면(다른 호스트/컨테이너에서 저장된 레거시 행)
      'storage/' 뒤쪽을 LOCAL_STORAGE_DIR로 rebase해서 재시도
    """
    if not path_value:
        return None
    raw = Path(path_value)
    if not raw.is_absolute():
        return LOCAL_STORAGE_DIR / raw
    if raw.exists():
        return raw
    parts = raw.parts
    if 'storage' in parts:
        suffix = parts[parts.index('storage') + 1:]
        rebased = LOCAL_STORAGE_DIR.joinpath(*suffix)
        if rebased.exists():
            return rebased
    return raw


# 스토리지 경로를 /files/ 하위 공개 URL로 변환, 해석 실패 시 빈 문자열
def make_file_url(path: str | Path | None) -> str:
    if not path:
        return ''
    resolved = resolve_storage_path(path)
    try:
        rel = resolved.resolve().relative_to(LOCAL_STORAGE_DIR.resolve())
    except Exception:
        return ''
    return '/files/' + '/'.join(rel.parts)


# 업로드 영상을 강의 입력 디렉터리에 저장, 기존 산출물 디렉터리는 초기화 후 재생성
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
