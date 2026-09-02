"""
공통 유틸리티 함수
"""

import os
import inspect
import subprocess
import time
from pathlib import Path


def anthropic_structured_output_request_kwargs(schema, create_method=None):
    """SDK 버전에 관계없이 호환되는 Anthropic Structured Output 인자 반환

    최신 Anthropic SDK는 messages.create에 output_config를 직접 노출, 구버전
    SDK는 이 키워드는 없지만 새로 추가된 API 필드를 전달하는 extra_body는 노출,
    두 경우 모두 HTTP 요청 본문에는 동일한 최상위 output_config 객체가 들어감
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
        # introspection이 불가능할 때(예: 일부 generated/mocked 클라이언트)는 최신 공식 SDK 규격을 우선 적용
        return {"output_config": output_config}

    if "output_config" in parameters:
        return {"output_config": output_config}

    if "extra_body" in parameters:
        return {"extra_body": {"output_config": output_config}}

    # 범용 **kwargs 클라이언트는 최신 규격을 지원한다고 가정
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return {"output_config": output_config}

    # 알 수 없는 필드를 전달하지 못하는 구식 SDK면, 실패를 감추지 않고 그대로 드러나게 둠
    return {"output_config": output_config}


def resolve_backend_root() -> Path:
    """subprocess cwd / PYTHONPATH 기준 프로젝트 루트

    - 로컬 VLVerifier: 저장소 루트 (`app/backend/pipeline` 경로)
    - Docker: /app (형제 패키지 `app`, `pipeline` — 위와 겹치지 않음)

    로컬 `.../app/backend`는 `pipeline/`과 `app/main.py`를 동시에 가지므로,
    monorepo 판별을 Docker 평면 레이아웃보다 먼저 수행
    """
    env_root = os.getenv("VLVERIFIER_ROOT") or os.getenv("PIPELINE_ROOT")
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
    """`pipeline` 패키지가 있는 디렉터리 (`python -m pipeline...` 실행 시 cwd)"""
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


# 에러 메시지에 재시도 가능한 패턴(429/503/500/타임아웃 등)이 포함되는지 확인
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
