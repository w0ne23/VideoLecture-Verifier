"""classified verifier 파이프라인이 공유하는 헬퍼"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


# analyzer 디렉터리를 가진 상위 경로를 저장소 루트로 판별
def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "analyzer").is_dir():
            return candidate
    return start


# subprocess로 실행될 때도 pipeline 패키지를 import할 수 있도록 루트를 sys.path에 추가
_ROOT = _find_repo_root(Path(__file__).resolve())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .claim_common import _empty_token_usage, _merge_token_usage


# 검증 단계 subprocess에 그대로 전달할 환경변수 allowlist
_ENV_KEYS = [
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY_1",
    "GOOGLE_API_KEY_2",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "XAI_BASE_URL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_THINKING",
    "DEEPSEEK_REASONING_EFFORT",
    "DEEPSEEK_TIMEOUT_SEC",
    "VERIFIER_DEEPSEEK_THINKING",
    "VERIFIER_DEEPSEEK_REASONING_EFFORT",
    "VERIFIER_DEEPSEEK_TIMEOUT_SEC",
    "VERIFIER_DEEPSEEK_API_MAX_RETRIES",
    "VERIFIER_DEEPSEEK_API_INITIAL_WAIT",
    "GROQ_API_KEY",
    "ISSUE_JUDGE_MAX_WORKERS",
    "VERIFIER_ISSUE_JUDGE_MIN_CONFIDENCE",
    "VERIFIER_ISSUE_JUDGE_SINGLE_MODEL_KEEP_CONFIDENCE",
    "VLVERIFIER_LLM_CONFIG_JSON",
    "VLVERIFIER_CREDENTIALS_JSON",
    "LITELLM_ENABLED",
    "LITELLM_BASE_URL",
    "LITELLM_API_KEY",
    "VERIFIER_CLAIM_EXTRACT_BATCH_SIZE",
    "VERIFIER_CLAIM_EXTRACT_MAX_WORKERS",
    "VERIFIER_CLAIM_EXTRACT_PROMPT_PROFILE",
    "VERIFIER_CLAIM_JUDGE_MODEL",
    "VERIFIER_ISSUE_JUDGE_PROMPT_LAYOUT",
    "VERIFIER_BATCH_SIZE",
    "VERIFIER_TEMPERATURE",
    "VERIFIER_PARSE_RETRIES",
    "VERIFIER_BATCH_RECOVERY_RETRIES",
    "VERIFIER_REQUIRE_COMPLETE",
    "VERIFIER_OPENAI_PROMPT_CACHE_KEY",
    "VERIFIER_OPENAI_PROMPT_CACHE_RETENTION",
    "OPENAI_PROMPT_CACHE_KEY",
    "OPENAI_PROMPT_CACHE_RETENTION",
    "VERIFIER_ANTHROPIC_PROMPT_CACHE",
    "VERIFIER_ANTHROPIC_PROMPT_CACHE_TTL",
    "ANTHROPIC_PROMPT_CACHE",
    "ANTHROPIC_PROMPT_CACHE_TTL",
    "VERIFIER_LLM_ISSUE_CLUSTERING",
    "VERIFIER_ISSUE_CLUSTER_MODEL",
    "VERIFIER_ISSUE_CLUSTER_MAX",
]

# 토큰 사용량 dict를 로그 출력용 요약 문자열로 변환
def _format_token_summary(usage: dict) -> str:
    total = (usage or {}).get("total", {})
    parts = [
        f"input {int(total.get('input_tokens', 0) or 0):,}",
        f"output {int(total.get('output_tokens', 0) or 0):,}",
        f"reasoning {int(total.get('reasoning_tokens', 0) or 0):,}",
        f"total {int(total.get('total_tokens', 0) or 0):,}",
    ]
    cached = int(total.get("cached_input_tokens", 0) or 0)
    if cached:
        parts.append(f"cached {cached:,}")
    cache_write = int(total.get("cache_creation_input_tokens", 0) or 0)
    if cache_write:
        parts.append(f"cache_write {cache_write:,}")
    return " / ".join(parts)


# _ENV_KEYS 중 값이 설정된 환경변수만 수집 (subprocess 전달용)
def _collect_env_vars() -> dict:
    from dotenv import load_dotenv

    load_dotenv()
    return {k: os.environ.get(k) for k in _ENV_KEYS if os.environ.get(k)}


# claim 목록을 jsonl 파일로 저장, 출력 파일명은 verification_final.json 접두어 기준
def _write_claims_jsonl(claims: list[dict], output_json_path: str | Path) -> str | None:
    if claims is None:
        return None
    output_json_path = Path(output_json_path)
    stem = output_json_path.name
    if stem.endswith("_verification_final.json"):
        prefix = stem[: -len("_verification_final.json")]
    else:
        prefix = output_json_path.stem
    out_path = output_json_path.with_name(f"{prefix}_claims.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for claim in claims:
            f.write(json.dumps(claim, ensure_ascii=False) + "\n")
    return str(out_path)


# jsonl 파일에서 claim 목록 로드, 파싱 실패 라인은 무시
def _load_claims_jsonl(path: str | Path) -> list[dict]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"claims jsonl 파일 없음: {jsonl_path}")

    claims = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            claims.append(payload)
    return claims


def _setup_worker(root: str, env_vars: dict, model: str):
    """subprocess 공통 초기화"""
    if root not in sys.path:
        sys.path.insert(0, root)
    for k, v in env_vars.items():
        if v is not None:
            os.environ[k] = v

    os.environ["VERIFIER_CLAIM_JUDGE_MODEL"] = model
    os.environ.setdefault("VERIFIER_TEMPERATURE", "0.0")
