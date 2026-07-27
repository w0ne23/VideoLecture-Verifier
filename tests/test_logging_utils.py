import io
import logging

from app.logging_utils import (
    attach_pipeline_log_handler,
    detach_pipeline_log_handler,
    reset_root_logging_handlers,
)


def test_reset_root_logging_handlers_removes_stale_closed_stream(monkeypatch):
    root_logger = logging.getLogger()
    stale_stream = io.StringIO()
    stale_handler = logging.StreamHandler(stale_stream)
    stale_stream.close()
    monkeypatch.setattr(root_logger, 'handlers', [stale_handler])

    reset_root_logging_handlers()

    assert root_logger.handlers == []


def test_pipeline_log_handler_is_detached_before_stream_is_reused(monkeypatch):
    root_logger = logging.getLogger()
    monkeypatch.setattr(root_logger, 'handlers', [])

    log_stream = io.StringIO()
    handler = attach_pipeline_log_handler(log_stream)
    logging.getLogger('test.pipeline').info('first job')

    detach_pipeline_log_handler(handler)

    assert 'first job' in log_stream.getvalue()
    assert root_logger.handlers == []
