"""
공통 유틸리티 함수
"""

import os
import inspect
import subprocess
import time
from pathlib import Path


def anthropic_structured_output_request_kwargs(schema, create_method=None):
    """Return SDK-version-compatible Anthropic Structured Output arguments.

    Current Anthropic SDKs expose ``output_config`` directly on
    ``messages.create``. Older SDKs do not expose that keyword, but do expose
    ``extra_body`` for forwarding newly added API fields. In both cases the
    HTTP request body contains the same top-level ``output_config`` object.
    """
    output_config = {
        "format": {
            "type": "json_schema",
            "schema": schema,
        }
    }

    if create_method is None:
        return {"output_config": output_config}

    try:
        parameters = inspect.signature(create_method).parameters
    except (TypeError, ValueError):
        # Prefer the current official SDK contract when introspection is not
        # available (for example, with some generated/mocked clients).
        return {"output_config": output_config}

    if "output_config" in parameters:
        return {"output_config": output_config}

    if "extra_body" in parameters:
        return {"extra_body": {"output_config": output_config}}

    # A generic **kwargs client is expected to support the current contract.
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return {"output_config": output_config}

    # Keep the failure explicit for an SDK too old to forward unknown fields.
    return {"output_config": output_config}


def resolve_backend_root() -> Path:
    """
    subprocess cwd / PYTHONPATH 기준 프로젝트 루트.

    - 로컬 graphLec: 저장소 루트 (`app/backend/pipeline` 경로)
    - Docker: /app (형제 패키지 `app`, `pipeline` — 위와 겹치지 않음)

    로컬 `.../app/backend` 는 `pipeline/` 과 `app/main.py` 를 동시에 가지므로,
    monorepo 판별을 Docker 평면 레이아웃보다 먼저 수행한다.
    """
    env_root = os.getenv("VERILEC_ROOT") or os.getenv("PIPELINE_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve().parent  # pipeline/
    chain = [here.parent, *here.parents]
    for p in chain:
        if (p / "app" / "backend" / "pipeline").is_dir():
            return p
    for p in chain:
        if (p / "pipeline").is_dir() and (p / "app" / "main.py").exists():
            return p
    return here.parent


def resolve_pipeline_package_root() -> Path:
    """`pipeline` 패키지가 있는 디렉터리 (`python -m pipeline...` 실행 시 cwd)."""
    root = resolve_backend_root()
    nested = root / "app" / "backend"
    if (nested / "pipeline").is_dir():
        return nested
    return root


def api_call_with_retry(func, max_retries=None, initial_wait=None):
    """API 호출 재시도 (429, 503, 500 에러 처리)"""
    if max_retries is None:
        max_retries = int(os.getenv("VERIFIER_API_MAX_RETRIES", "5"))
    if initial_wait is None:
        initial_wait = float(os.getenv("VERIFIER_API_INITIAL_WAIT", "10"))
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            error_msg = str(e)
            retry_errors = [
                "429",
                "503",
                "500",
                "RESOURCE_EXHAUSTED",
                "UNAVAILABLE",
                "overloaded",
                "timeout",
                "timed out",
                "APITimeout",
                "ReadTimeout",
                "Connection error",
                "APIConnectionError",
                "ConnectTimeout",
                "connect timeout",
            ]
            if any(code in error_msg for code in retry_errors) and attempt < max_retries - 1:
                print(f"API ERROR: {error_msg}")
                wait = initial_wait * (attempt + 1)
                print(f"  ↺ {wait:.1f}s 후 재시도 ({attempt + 1}/{max_retries - 1})")
                time.sleep(wait)
            else:
                raise e
    if last_error is not None:
        raise last_error
    raise Exception("API 호출 실패")


def is_retryable_api_error(error) -> bool:
    error_msg = str(error)
    retry_errors = [
        "429",
        "503",
        "500",
        "RESOURCE_EXHAUSTED",
        "UNAVAILABLE",
        "overloaded",
        "timeout",
        "timed out",
        "APITimeout",
        "ReadTimeout",
        "Connection error",
        "APIConnectionError",
        "ConnectTimeout",
        "connect timeout",
    ]
    return any(code in error_msg for code in retry_errors)


def get_video_duration(file_path: str) -> float:
    """영상 길이 추출 (초)"""
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ], capture_output=True, text=True)
    return float(result.stdout.strip())
