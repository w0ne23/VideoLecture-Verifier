# 강의별 파이프라인 산출물 경로 계산
from pathlib import Path

from app.config import LOCAL_STORAGE_DIR, PIPELINE_ROOT


# lecture_id에 대응하는 파이프라인 산출물 디렉터리/파일 경로 모음
def pipeline_paths(lecture_id: str) -> dict[str, Path]:
    output_dir = LOCAL_STORAGE_DIR / 'results' / lecture_id
    return {
        'pipeline_root': PIPELINE_ROOT,
        'output_dir': output_dir,
        'slides_dir': output_dir / 'slides',
        'log_file': output_dir / 'pipeline.log',
    }
