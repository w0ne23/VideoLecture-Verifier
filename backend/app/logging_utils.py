import logging
from typing import TextIO


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
