# 파이프라인 로그를 콘솔과 pipeline.log 파일에 동시에 남기는 유틸
import logging
import os
import sys
import threading
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Iterator, TextIO


PIPELINE_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class _LivePipelineTee:
    """호출자의 콘솔과 pipeline.log에 즉시 동시 기록하는 스트림 래퍼"""

    def __init__(self, console: TextIO, log_file: TextIO, lock: threading.RLock):
        self.console = console
        self.log_file = log_file
        self.lock = lock
        self.encoding = getattr(console, "encoding", None) or "utf-8"
        self.errors = getattr(console, "errors", None) or "replace"

    def write(self, data: str) -> int:
        if not data:
            return 0
        with self.lock:
            console_written = self.console.write(data)
            self.console.flush()
            self.log_file.write(data)
            self.log_file.flush()
        return int(console_written) if isinstance(console_written, int) else len(data)

    def flush(self) -> None:
        with self.lock:
            self.console.flush()
            self.log_file.flush()

    def fileno(self) -> int:
        return self.console.fileno()

    def isatty(self) -> bool:
        return bool(getattr(self.console, "isatty", lambda: False)())

    def __getattr__(self, name: str):
        return getattr(self.console, name)


# 콘솔 동시 출력 활성화 여부, PIPELINE_LOG_CONSOLE=0/false/no/off면 비활성화
def _console_tee_enabled() -> bool:
    value = os.getenv("PIPELINE_LOG_CONSOLE", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


# 루트 로거에 붙은 기존 핸들러를 모두 제거하고 정리
def reset_root_logging_handlers() -> None:
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


# 루트 로거에 파일 핸들러를 붙여 로그를 log_file로 기록
def attach_pipeline_log_handler(log_file: TextIO) -> logging.Handler:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(log_file)
    handler.setFormatter(logging.Formatter(PIPELINE_LOG_FORMAT))
    root_logger.addHandler(handler)
    return handler


# 붙였던 로그 핸들러를 제거하고 닫음
def detach_pipeline_log_handler(handler: logging.Handler) -> None:
    root_logger = logging.getLogger()
    root_logger.removeHandler(handler)
    handler.close()


@contextmanager
def pipeline_log_context(output_dir: Path | str) -> Iterator[Path]:
    """CLI로 직접 실행할 때도 백그라운드 job과 동일하게 {output_dir}/pipeline.log를
    남김, 기본값은 stdout/stderr를 터미널과 파일 양쪽에 즉시 출력, 콘솔 출력이
    필요 없는 백그라운드 환경에서는 PIPELINE_LOG_CONSOLE=0으로 이전의 파일 전용
    동작을 사용 가능
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_file_path = output_path / "pipeline.log"

    reset_root_logging_handlers()
    with log_file_path.open("w", encoding="utf-8", buffering=1) as log_file:
        if _console_tee_enabled():
            write_lock = threading.RLock()
            stdout_target: TextIO = _LivePipelineTee(sys.stdout, log_file, write_lock)
            stderr_target: TextIO = _LivePipelineTee(sys.stderr, log_file, write_lock)
            logging_target = stderr_target
        else:
            stdout_target = log_file
            stderr_target = log_file
            logging_target = log_file

        log_handler = attach_pipeline_log_handler(logging_target)
        try:
            with redirect_stdout(stdout_target), redirect_stderr(stderr_target):
                yield log_file_path
        finally:
            detach_pipeline_log_handler(log_handler)
