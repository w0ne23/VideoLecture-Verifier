import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Iterator, TextIO


PIPELINE_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def reset_root_logging_handlers() -> None:
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def attach_pipeline_log_handler(log_file: TextIO) -> logging.Handler:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(log_file)
    handler.setFormatter(logging.Formatter(PIPELINE_LOG_FORMAT))
    root_logger.addHandler(handler)
    return handler


def detach_pipeline_log_handler(handler: logging.Handler) -> None:
    root_logger = logging.getLogger()
    root_logger.removeHandler(handler)
    handler.close()


@contextmanager
def pipeline_log_context(output_dir: Path | str) -> Iterator[Path]:
    """CLI로 직접 실행할 때도 백그라운드 job과 동일하게 {output_dir}/pipeline.log를
    남긴다. 이 파이프라인은 진행 로그 대부분을 logging이 아니라 print()로 찍으므로,
    logging 핸들러만 붙이는 것으로는 부족해 stdout/stderr 자체를 파일로 리다이렉트한다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_file_path = output_path / "pipeline.log"

    reset_root_logging_handlers()
    with log_file_path.open("w", encoding="utf-8", buffering=1) as log_file:
        log_handler = attach_pipeline_log_handler(log_file)
        try:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                yield log_file_path
        finally:
            detach_pipeline_log_handler(log_handler)
