"""analyzer issue-judge 출력을 가중치 기반 앙상블로 4가지 issue 유형으로 분류하는 독립 실행형 분류기"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "issue_types.v3"
# LLM이 확률을 내는 4유형, composite_issue는 별도 routing 결과로 유지
ISSUE_TYPES = (
    "temporal_error",
    "scope_overclaim",
    "factual_error",
    "confusing_explanation",
)
COMPOSITE_ISSUE_TYPE = "composite_issue"
ALL_ISSUE_TYPES = ISSUE_TYPES + (COMPOSITE_ISSUE_TYPE,)
DEFAULT_LIST_KEYS = ("issues",)
ISSUE_TYPE_LABELS = {
    "factual_error": "사실 오류",
    "temporal_error": "오래된 내용",
    "scope_overclaim": "과도한 일반화",
    "confusing_explanation": "혼동 가능 설명",
    COMPOSITE_ISSUE_TYPE: "복합 오류",
}
ISSUE_TYPE_SHORT_LABELS = {
    "factual_error": "factual",
    "temporal_error": "temporal",
    "scope_overclaim": "scope",
    "confusing_explanation": "confusing",
}
TOKEN_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "tool_input_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "total_tokens",
)

DEFAULT_LOW_MARGIN_THRESHOLD = 0.10


# .env 파일 로드 (dotenv 미설치면 무시)
def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


# 현재 시각을 ISO 8601 문자열로 반환
def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# LLM 호출 seed 값 조회, 환경변수 미설정/파싱 실패 시 None
def _env_seed() -> int | None:
    raw = str(os.getenv("VERIFIER_SEED", "") or "").strip()
    if not raw:
        raw = str(os.getenv("ISSUE_TYPE_CLASSIFIER_SEED", "") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


# 콤마/공백 구분 문자열을 리스트로 분리
def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part for part in re.split(r"[\s,]+", str(value).strip()) if part]


# issue_classify 스테이지에 설정된 모델 목록 조회
def _default_models() -> list[str]:
    _load_env()
    try:
        from .runtime_llm import configured_stage_models
    except ImportError:
        from runtime_llm import configured_stage_models
    return configured_stage_models("issue_classify")


# issue의 고유 식별자 조회, issue_id/claim_id가 없으면 내용 기반 해시로 생성
def _issue_identity(list_key: str, index: int, issue: dict[str, Any]) -> str:
    issue_id = str(issue.get("issue_id") or issue.get("claim_id") or "").strip()
    if issue_id:
        return issue_id
    seed = {
        "list_key": list_key,
        "index": index,
        "claim_text": issue.get("claim_text") or issue.get("resolved_claim") or "",
        "issue": issue.get("issue", ""),
    }
    digest = hashlib.sha1(json.dumps(seed, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{list_key}:{index + 1}:{digest}"


# payload의 지정 list_key들에서 issue 목록을 식별자와 함께 수집
def collect_issues(payload: dict[str, Any], list_keys: list[str]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for list_key in list_keys:
        rows = payload.get(list_key, [])
        if not isinstance(rows, list):
            continue
        for index, issue in enumerate(rows):
            if not isinstance(issue, dict):
                continue
            refs.append(
                {
                    "id": _issue_identity(list_key, index, issue),
                    "list_key": list_key,
                    "index": index,
                    "issue": issue,
                }
            )
    return refs


# 입력 경로의 상위 디렉터리들에서 *_merged_clean.json 파일 탐색
def _guess_merged_clean_path(input_path: Path) -> Path | None:
    for parent in [input_path.parent, *input_path.parents]:
        candidates = sorted(parent.glob("*_merged_clean.json"))
        if candidates:
            return candidates[0]
    return None


# 리스트를 지정 크기로 분할
def _chunk(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# LLM 프롬프트에 넣을 issue 요약(식별자+claim 텍스트) 추출
def _issue_brief(ref: dict[str, Any]) -> dict[str, Any]:
    issue = ref["issue"]
    return {
        "id": ref["id"],
        "issue_id": issue.get("issue_id", ""),
        "claim_id": issue.get("claim_id", ""),
        "resolved_claim": issue.get("resolved_claim", ""),
    }


# issue 유형 분류용 LLM 프롬프트 생성
def _build_prompt(items: list[dict[str, Any]], current_date: str) -> str:
    rows = [_issue_brief(item) for item in items]
    return f"""당신은 강의 verifier가 선별한 issue 후보를 4가지 유형으로 재분류하는 심사자입니다.

기준일: {current_date}

각 입력 issue에 대해 아래 네 유형에 해당할 가능성을 확률 분포로 평가하세요.
확률은 네 유형 전체 합이 1.0이 되도록 작성하세요.
애매한 경우에는 가장 그럴듯한 한 유형에만 몰지 말고, 가능한 유형들에 확률을 나누어 주세요.

분류:
- factual_error: 정의, 용어, 동작 원리, 관계, 순서, 메커니즘 등 객관적으로 틀린 사실 오류. 기준일과 무관하게 명제 자체가 틀린 경우.
- temporal_error: 현재 기준으로 업데이트되지 않은 정보. 과거 어느 시점에는 맞았거나 자연스러웠을 수 있지만, 현재 기준으로는 부족하거나, 더 이상 맞지 않는 정보인 경우.
- confusing_explanation: 명제가 명백히 틀렸다고 단정하기보다는, 비유/예시/생략/표현 방식 때문에 학생이 해당 명제를 다른 의미로 해석할 위험이 있는 설명.
- scope_overclaim: 조건, 예외, 범위, 적용 대상을 닫아버려 과도하게 일반화한 오류. “항상/오직/모든/유일한/전부/완전히/~만” 같은 범위 표현을 제거하거나 완화하면 대체로 맞는 명제가 되는 경우.

판단 기준:
- resolved_claim만 근거로 판단하세요.
- 원문 문맥, 슬라이드, 앞뒤 설명을 추정하지 마세요.
- 이 단계는 issue가 맞는지 최종 판정하는 단계가 아니라, 후속 verifier가 어떤 기준으로 검증해야 하는지 정하는 routing 단계입니다.

중요:
- 단순히 날짜나 시점 표현이 들어갔다고 temporal_error가 아니다. 제시된 시점에서도 틀린 정의/원리/관계/메커니즘 오류는 factual_error로 본다.
- factual_error와 temporal_error가 모두 가능하면, 그 정보가 과거에는 맞았거나 당시에는 합리적이었지만 현재 기준으로 낡은 경우 temporal_error에 더 높은 확률을 주세요. 제시된 시점이나 과거 기준에서도 틀린 명제라면 factual_error에 더 높은 확률을 주세요.
- 단순히 더 자세한 설명이 가능하다는 이유만으로 confusing_explanation을 선택하지 마세요.
- "항상", "모든", "오직", "유일한" 같은 단어가 있다는 이유만으로 scope_overclaim로 올리지 마세요.
- factual_error와 scope_overclaim이 모두 가능하면, 제한 표현이나 범위 단정만 완화하면 대체로 맞는 문장이 되는 경우 scope_overclaim에 더 높은 확률을 주세요. 명제의 핵심 내용 자체가 틀리면 factual_error에 더 높은 확률을 주세요.
- temporal_error와 scope_overclaim이 모두 가능하면, 현재 기술 생태계 변화로 인해 최신 사례나 대안이 빠져 현재 기준으로 부족한 정보이면 temporal_error에 더 높은 확률을 주세요.


응답은 JSON만 출력하세요.
모든 입력 id에 대해 classifications 항목을 하나씩 포함하세요.
각 probabilities 객체는 네 키를 모두 포함해야 하며, 값은 0.0 이상 1.0 이하 숫자여야 합니다.
probabilities의 합은 1.0이 되도록 하세요.
confidence는 해당 확률 분포 전체에 대한 모델 자신의 신뢰도입니다.

```json
{{
  "classifications": [
    {{
      "id": "입력 id",
      "probabilities": {{
        "factual_error": 0.0,
        "temporal_error": 0.0,
        "confusing_explanation": 0.0,
        "scope_overclaim": 0.0
      }},
      "reason": "한두 문장 근거",
      "confidence": 0.0
    }}
  ]
}}
```

입력 issue:
{json.dumps(rows, ensure_ascii=False, indent=2)}
"""


# 코드펜스(```json ... ```) 제거
def _strip_json_fence(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


# LLM 응답에서 classifications 배열 파싱
def _parse_response(text: str) -> list[dict[str, Any]]:
    payload = json.loads(_strip_json_fence(text))
    rows = payload.get("classifications", [])
    return rows if isinstance(rows, list) else []


# 문자열을 4개 정규 유형 중 하나로 정규화, 아니면 None
def _normalize_issue_type(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    return raw if raw in ISSUE_TYPES else None


# issue_type에 대응하는 한글 라벨 조회
def _issue_type_label(issue_type: str | None) -> str:
    if not issue_type:
        return "분류 실패"
    normalized = str(issue_type or "").strip().lower()
    return ISSUE_TYPE_LABELS.get(normalized, str(issue_type))


# 값을 float로 안전 변환, 실패 시 default
def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# 확률 값 검증(숫자/NaN/inf/음수 여부), 오류 메시지와 함께 반환
def _safe_probability(value: Any) -> tuple[float | None, str | None]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, f"숫자가 아닌 확률값: {value!r}"
    if number != number or number in {float("inf"), float("-inf")}:
        return None, f"유효하지 않은 확률값: {value!r}"
    if number < 0:
        return None, f"음수 확률값: {value!r}"
    return number, None


# LLM이 낸 확률 dict를 4개 유형 합이 1이 되도록 정규화, 검증 실패 시 오류 메시지 반환
def _normalize_probabilities(raw: Any) -> tuple[dict[str, float] | None, str | None]:
    if not isinstance(raw, dict):
        return None, "probabilities 객체가 없거나 dict가 아닙니다."
    values: dict[str, float] = {}
    errors = []
    for issue_type in ISSUE_TYPES:
        number, error = _safe_probability(raw.get(issue_type, 0.0))
        if error:
            errors.append(f"{issue_type}: {error}")
        else:
            values[issue_type] = number if number is not None else 0.0
    if errors:
        return None, "; ".join(errors)
    total = sum(values.values())
    if total <= 0:
        return None, "probabilities 합계가 0입니다."
    return {key: round(value / total, 6) for key, value in values.items()}, None


# 확률 dict에서 최고 확률 유형과 그 값 조회
def _top_probability_type(probabilities: dict[str, float]) -> tuple[str | None, float]:
    if not probabilities:
        return None, 0.0
    top_type = max(ISSUE_TYPES, key=lambda issue_type: probabilities.get(issue_type, 0.0))
    return top_type, float(probabilities.get(top_type, 0.0) or 0.0)


# 여러 환경변수 이름 중 값이 있는 첫 번째를 반환
def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def resolve_low_margin_threshold(override: float | None = None) -> float:
    """1위·2위 유형 점수 차이가 이 값 미만이면 low_margin으로 표시"""
    if override is not None:
        return max(0.0, float(override))
    raw = _env_first(
        "ISSUE_TYPE_CLASSIFIER_LOW_MARGIN_THRESHOLD",
        "VERIFIER_ISSUE_TYPE_CLASSIFIER_LOW_MARGIN_THRESHOLD",
        default=str(DEFAULT_LOW_MARGIN_THRESHOLD),
    )
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_LOW_MARGIN_THRESHOLD


# 환경변수를 float로 읽되 최솟값 이상으로 clamp
def _env_float(name: str, default: float, *, min_value: float = 1.0) -> float:
    try:
        value = float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(min_value, value)


# 환경변수를 int로 읽되 최솟값 이상으로 clamp
def _env_int(name: str, default: int, *, min_value: int = 0) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(min_value, value)


# 축약된 별칭을 실제 Anthropic 모델 ID로 변환
def _resolve_anthropic_model(model_name: str) -> str:
    aliases = {
        "haiku-4.5": "claude-haiku-4-5-20251001",
        "claude-haiku-4.5": "claude-haiku-4-5-20251001",
        "claude-haiku-4-5": "claude-haiku-4-5-20251001",
        "sonnet-4.5": "claude-sonnet-4-5-20250929",
        "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
        "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
        "opus-4.5": "claude-3-opus-latest",
        "claude-opus-4.5": "claude-3-opus-latest",
        "claude-opus-4-5": "claude-3-opus-latest",
    }
    return aliases.get(str(model_name or "").strip(), str(model_name or "").strip())


def _resolve_vllm_model(model_spec: str) -> str:
    """OpenAI 호환 local/remote vLLM 모델 스펙 해석"""
    raw = str(model_spec or "").strip()
    lowered = raw.lower()
    if lowered in {"vllm", "qwen", "local", "local-llm"}:
        return _env_first(
            "LOCAL_LLM_MODEL",
            "VLLM_MODEL",
            "QWEN_MODEL",
            default="qwen3.8-27b",
        )
    for prefix in ("vllm:", "vllm/", "qwen:", "qwen/"):
        if lowered.startswith(prefix):
            return raw[len(prefix):].strip()
    return raw


# 모델 스펙 문자열을 provider/alias/실제 모델명으로 해석
def _resolve_model_spec(model_spec: str) -> dict[str, str]:
    raw = str(model_spec or "").strip()
    lowered = raw.lower()
    if (
        lowered in {"vllm", "qwen", "qwen3.8", "local", "local-llm"}
        or lowered.startswith(("vllm:", "vllm/", "qwen:", "qwen/", "qwen-"))
    ):
        return {
            "provider": "vllm",
            "alias": raw,
            "resolved_model": (
                _env_first(
                    "ISSUE_TYPE_CLASSIFIER_QWEN_MODEL",
                    "VERIFIER_ISSUE_TYPE_CLASSIFIER_QWEN_MODEL",
                    default="qwen3.8-27b",
                )
                if lowered == "qwen3.8"
                else _resolve_vllm_model(raw)
            ) or _resolve_vllm_model(raw),
        }
    if lowered in {"deepseek", "deepseek-default"}:
        return {
            "provider": "deepseek",
            "alias": raw,
            "resolved_model": _env_first(
                "ISSUE_TYPE_CLASSIFIER_DEEPSEEK_MODEL",
                "VERIFIER_ISSUE_TYPE_CLASSIFIER_DEEPSEEK_MODEL",
                default="deepseek-v4-flash",
            ),
        }
    if lowered in {"gemini", "google"}:
        return {
            "provider": "gemini",
            "alias": raw,
            "resolved_model": _env_first(
                "ISSUE_TYPE_CLASSIFIER_GEMINI_MODEL",
                "VERIFIER_ISSUE_TYPE_CLASSIFIER_GEMINI_MODEL",
                default="gemini-2.5-flash",
            ),
        }
    if lowered.startswith(("gpt", "o1", "o3")):
        return {"provider": "openai", "alias": raw, "resolved_model": raw}
    if lowered.startswith("xai:"):
        return {"provider": "xai", "alias": raw, "resolved_model": raw.split(":", 1)[1].strip()}
    if lowered.startswith("xai/"):
        return {"provider": "xai", "alias": raw, "resolved_model": raw.split("/", 1)[1].strip()}
    if lowered.startswith("grok"):
        return {"provider": "xai", "alias": raw, "resolved_model": raw}
    if lowered.startswith("deepseek:"):
        return {"provider": "deepseek", "alias": raw, "resolved_model": raw.split(":", 1)[1].strip()}
    if lowered.startswith("deepseek/"):
        return {"provider": "deepseek", "alias": raw, "resolved_model": raw.split("/", 1)[1].strip()}
    if lowered.startswith("deepseek"):
        return {"provider": "deepseek", "alias": raw, "resolved_model": raw}
    if lowered.startswith("google:"):
        return {"provider": "gemini", "alias": raw, "resolved_model": raw.split(":", 1)[1].strip()}
    if lowered.startswith("google/"):
        return {"provider": "gemini", "alias": raw, "resolved_model": raw.split("/", 1)[1].strip()}
    if lowered.startswith("gemini"):
        return {"provider": "gemini", "alias": raw, "resolved_model": raw}
    if lowered.startswith("claude") or lowered.startswith(("sonnet", "haiku", "opus")):
        return {"provider": "anthropic", "alias": raw, "resolved_model": _resolve_anthropic_model(raw)}
    if lowered.startswith("ollama:"):
        return {"provider": "ollama", "alias": raw, "resolved_model": raw.split(":", 1)[1].strip()}
    if lowered.startswith("ollama/"):
        return {"provider": "ollama", "alias": raw, "resolved_model": raw.split("/", 1)[1].strip()}
    # 설정된 runtime/LiteLLM 바인딩이 호출을 담당할 때는 알 수 없는 구체적 모델 ID도
    # 유효함, 여기서는 라우팅하거나 치환하지 않음
    return {"provider": "runtime", "alias": raw, "resolved_model": raw}


def _call_llm(
    *,
    model_spec: str,
    prompt: str,
    max_tokens: int,
    web_search: bool = False,
    web_search_max_calls: int = 2,
    web_search_force: bool = False,
    web_search_context_size: str | None = None,
    structured_schema: dict[str, Any] | None = None,
    stage: str = "issue_classify",
) -> tuple[str, dict[str, Any], dict[str, str]]:
    """선택된 classifier 모델을 런타임 게이트웨이로 호출"""
    try:
        from .runtime_llm import call_runtime_llm, resolve_runtime_binding
    except ImportError:
        from runtime_llm import call_runtime_llm, resolve_runtime_binding

    runtime_binding = resolve_runtime_binding(stage, model_spec)
    if not runtime_binding:
        raise RuntimeError(
            f"{stage} 단계의 선택 모델을 런타임 바인딩으로 해석하지 못했습니다: {model_spec}"
        )
    runtime_result = call_runtime_llm(
        runtime_binding,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        response_format=(
            None
            if web_search
            else (
                {"type": "json_schema", "schema": structured_schema}
                if structured_schema
                else {"type": "json_object"}
            )
        ),
        model_spec=model_spec,
        stage=stage,
        web_search=web_search,
        web_search_max_calls=web_search_max_calls,
        web_search_force=web_search_force,
        web_search_context_size=web_search_context_size,
    )
    if runtime_result is None:
        raise RuntimeError(
            f"{stage} 단계가 지원하지 않는 런타임 프로토콜입니다: {model_spec}"
        )
    text, usage = runtime_result
    resolved = {
        "provider": runtime_binding.get("provider", ""),
        "alias": model_spec,
        "resolved_model": runtime_binding.get("resolved_model", model_spec),
        "endpoint_ref": runtime_binding.get("endpoint_ref", ""),
    }
    return text, usage, resolved


# 모델 1개가 담당하는 전체 배치를 순차 처리
def _model_worker(args: tuple) -> dict[str, Any]:
    model, issues, batch_size, current_date, max_tokens = args
    resolved = _resolve_model_spec(model)
    batches = _chunk(issues, batch_size)
    classifications: list[dict[str, Any]] = []
    token_usage_by_batch: list[dict[str, Any]] = []
    print(
        f"  [{model}] 시작: provider={resolved['provider']}, model={resolved['resolved_model']}, "
        f"issues={len(issues)}, batches={len(batches)}",
        flush=True,
    )
    for batch_index, batch in enumerate(batches, start=1):
        ids = f"{batch[0]['id']}..{batch[-1]['id']}" if batch else "-"
        print(f"  [{model}] batch {batch_index}/{len(batches)} 요청 중: {ids}", flush=True)
        rows, usage = _call_model_for_batch(
            model=model,
            batch=batch,
            current_date=current_date,
            max_tokens=max_tokens,
        )
        for row in rows:
            row["batch_index"] = batch_index
        classifications.extend(rows)
        token_usage_by_batch.append(usage)
        ok_count = sum(1 for row in rows if row.get("status") == "ok")
        print(
            f"  [{model}] batch {batch_index}/{len(batches)} 완료: "
            f"{ok_count}/{len(rows)} probability vectors parsed",
            flush=True,
        )
    print(f"  [{model}] 완료: {len(classifications)}건", flush=True)
    return {
        "model": model,
        "provider": resolved["provider"],
        "resolved_model": resolved["resolved_model"],
        "status": "ok",
        "classifications": classifications,
        "token_usage_by_batch": token_usage_by_batch,
    }


# 배치 1개를 처리, 파싱 실패 항목만 골라 재시도, 최종 실패한 issue는 parse_failed로 채움
def _batch_worker(args: tuple) -> dict[str, Any]:
    model, batch, batch_index, total_batches, current_date, max_tokens = args
    resolved = _resolve_model_spec(model)
    attempts = _env_int("ISSUE_TYPE_CLASSIFIER_BATCH_RETRIES", 2, min_value=0) + 1
    retry_wait = _env_float("ISSUE_TYPE_CLASSIFIER_BATCH_RETRY_WAIT_SEC", 0.0, min_value=0.0)
    last_exc: Exception | None = None
    pending = list(batch)
    rows_by_id: dict[str, dict[str, Any]] = {}
    usages: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        ids = f"{pending[0]['id']}..{pending[-1]['id']}" if pending else "-"
        suffix = f" (시도 {attempt}/{attempts})" if attempts > 1 else ""
        print(f"  [{model}] batch {batch_index}/{total_batches} 요청 중: {ids}{suffix}", flush=True)
        try:
            rows, usage = _call_model_for_batch(
                model=model,
                batch=pending,
                current_date=current_date,
                max_tokens=max_tokens,
            )
            usages.append(usage)
            returned = {str(row.get("id") or ""): row for row in rows}
            for item in pending:
                issue_id = str(item.get("id") or "")
                if issue_id in returned:
                    rows_by_id[issue_id] = returned[issue_id]

            failed = [
                item
                for item in pending
                if rows_by_id.get(str(item.get("id") or ""), {}).get("status") != "ok"
            ]
            if not failed or attempt >= attempts:
                break

            ok_count = len(pending) - len(failed)
            last_exc = ValueError(f"probability vectors parsed {ok_count}/{len(pending)}")
            print(
                f"    [{model}] batch {batch_index}/{total_batches} 재시도 "
                f"{attempt}/{attempts - 1}: {last_exc}",
                flush=True,
            )
            pending = failed
            if retry_wait:
                time.sleep(retry_wait)
        except Exception as exc:
            last_exc = exc
            # anthropic.APIConnectionError.__str__()은 실제 원인과 무관하게 항상 고정된
            # "Connection error." 문자열을 반환함 — 진짜 근본 원인(TLS, DNS, proxy 등)이
            # 보이도록 감싸진 원인(cause)도 함께 표면화
            cause = getattr(exc, "__cause__", None)
            detail = f"{type(exc).__name__}: {exc}" + (f" (caused by {type(cause).__name__}: {cause})" if cause else "")
            if attempt >= attempts:
                raise
            print(
                f"    [{model}] batch {batch_index}/{total_batches} 재시도 "
                f"{attempt}/{attempts - 1}: {detail}",
                flush=True,
            )
            if retry_wait:
                time.sleep(retry_wait)
    else:
        raise RuntimeError("batch 분류 실패") from last_exc

    rows = []
    for item in batch:
        issue_id = str(item.get("id") or "")
        row = rows_by_id.get(issue_id)
        if row is None:
            row = {
                "id": issue_id,
                "status": "parse_failed",
                "parse_error": "모델 응답에서 해당 이슈 결과가 누락되었습니다.",
            }
        rows.append(row)
    for row in rows:
        row["batch_index"] = batch_index
    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    print(
        f"  [{model}] batch {batch_index}/{total_batches} 완료: "
        f"{ok_count}/{len(rows)} probability vectors parsed",
        flush=True,
    )
    return {
        "model": model,
        "provider": resolved["provider"],
        "resolved_model": resolved["resolved_model"],
        "batch_index": batch_index,
        "classifications": rows,
        "token_usage": _aggregate_token_usage(usages),
    }


# 배치 처리 성공 결과를 모델별 결과 dict에 누적
def _append_batch_result(model_results: dict[str, dict[str, Any]], result: dict[str, Any]) -> None:
    model = result["model"]
    target = model_results[model]
    target["provider"] = result["provider"]
    target["resolved_model"] = result["resolved_model"]
    target["_batch_results"].append((result["batch_index"], result["classifications"]))
    target["token_usage_by_batch"].append(result["token_usage"])


# 배치 처리 실패를 모델별 결과 dict에 기록 (모델 항목이 없으면 새로 생성)
def _append_batch_error(model_results: dict[str, dict[str, Any]], args: tuple, exc: Exception) -> None:
    model, _batch, batch_index, _total_batches, _current_date, _max_tokens = args
    try:
        resolved = _resolve_model_spec(model)
    except Exception:
        resolved = {"provider": "unknown", "resolved_model": model}
    target = model_results.setdefault(model, {
        "model": model,
        "provider": resolved["provider"],
        "resolved_model": resolved["resolved_model"],
        "classifications": [],
        "token_usage_by_batch": [],
        "_batch_results": [],
        "batch_errors": [],
    })
    target["batch_errors"].append({"batch_index": batch_index, "error": str(exc)})


# 여러 배치의 토큰 사용량을 필드별로 합산
def _aggregate_token_usage(usages: list[dict[str, Any]]) -> dict[str, int]:
    totals = Counter()
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for key in TOKEN_USAGE_FIELDS:
            totals[key] += int(usage.get(key, 0) or 0)
    return dict(totals)


def _parse_model_weights(value: str | None, models: list[str], model_results: dict[str, dict[str, Any]]) -> dict[str, float]:
    """선택된 모든 모델에 동일한 투표 가중치 부여"""
    del value, model_results
    if not models:
        return {}
    equal = 1.0 / len(models)
    return {model: round(equal, 6) for model in models}


# 모델별 확률에 가중치를 곱해 유형별 가중 점수 합산, 실패/파싱 오류 모델의 가중치는 missing_weight로 집계
def _weighted_scores(
    verdicts: list[dict[str, Any]],
    model_weights: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], float]:
    scores = {issue_type: 0.0 for issue_type in ISSUE_TYPES}
    used_weights: dict[str, float] = {}
    missing_weight = 0.0
    for verdict in verdicts:
        model = str(verdict.get("model", "") or "")
        weight = float(model_weights.get(model, 0.0) or 0.0)
        if verdict.get("status") != "ok":
            missing_weight += weight
            continue
        probabilities = verdict.get("probabilities")
        if not isinstance(probabilities, dict):
            missing_weight += weight
            continue
        used_weights[model] = weight
        for issue_type in ISSUE_TYPES:
            scores[issue_type] += float(probabilities.get(issue_type, 0.0) or 0.0) * weight
    return {key: round(value, 6) for key, value in scores.items()}, used_weights, round(missing_weight, 6)


# 가중 점수가 가장 높은 유형을 잠정 채택, 1위/2위 점수 차이가 임계값 미만이면 low_margin으로 표시
def _choose_final_type(
    verdicts: list[dict[str, Any]],
    model_weights: dict[str, float],
    low_margin_threshold: float,
) -> tuple[str | None, float, dict[str, float], dict[str, float], float, bool, float]:
    scores, used_weights, missing_weight = _weighted_scores(verdicts, model_weights)
    if not used_weights:
        return None, 0.0, scores, used_weights, missing_weight, False, 0.0
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    final_type, final_score = sorted_scores[0]
    runner_up_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    margin = round(final_score - runner_up_score, 6)
    return (
        final_type,
        final_score,
        scores,
        used_weights,
        missing_weight,
        margin < low_margin_threshold,
        margin,
    )


# 정상 응답한 모델들의 최고 확률 유형만 모아 리스트로 반환
def _model_top_types(verdicts: list[dict[str, Any]]) -> list[str]:
    tops: list[str] = []
    for verdict in verdicts:
        if verdict.get("status") != "ok":
            continue
        top = str(verdict.get("top_issue_type") or "").strip()
        if top in ISSUE_TYPES:
            tops.append(top)
    return tops


# 참여 모델 수만큼 서로 다른 top_issue_type이 나왔는지(전원 의견 불일치) 확인
def _all_models_disagree(verdicts: list[dict[str, Any]], *, expected_model_count: int) -> bool:
    # 모델이 하나뿐이면 다른 모델과 의견이 갈릴 수 없음, 이 가드가 없으면 모델 1개로
    # 실행했을 때 유효한 top type 1개만 있어도 ``len(set(tops)) == expected_model_count``
    # 조건이 성립해 잘못 composite_issue로 라우팅됨
    if expected_model_count < 2:
        return False
    tops = _model_top_types(verdicts)
    return len(tops) >= expected_model_count and len(set(tops)) == expected_model_count


# low_margin 또는 전 모델 의견 불일치 시 최종 유형을 composite_issue로 재라우팅
def _apply_composite_routing(
    *,
    provisional_type: str | None,
    low_margin: bool,
    verdicts: list[dict[str, Any]],
    expected_model_count: int,
) -> tuple[str | None, list[str]]:
    reasons: list[str] = []
    if low_margin:
        reasons.append("low_margin")
    if _all_models_disagree(verdicts, expected_model_count=expected_model_count):
        reasons.append("model_disagreement")
    if reasons and provisional_type:
        return COMPOSITE_ISSUE_TYPE, reasons
    return provisional_type, reasons


# 모델 실패나 누락된 issue별 판정이 없는지 검증, 문제가 있으면 예외로 전체 실행 중단
def _validate_classification_completeness(
    *,
    models: list[str],
    model_results: dict[str, dict[str, Any]],
    refs: list[dict[str, Any]],
    verdicts_by_id: dict[str, list[dict[str, Any]]],
) -> None:
    failed_models = [model for model, row in model_results.items() if row.get("status") == "failed"]
    if failed_models:
        errors = [
            f"{model}: {model_results[model].get('error', 'failed')}"
            for model in failed_models
        ]
        raise RuntimeError("issue type classifier model failure: " + "; ".join(errors))

    expected = len(models)
    for ref in refs:
        issue_id = ref["id"]
        verdicts = verdicts_by_id.get(issue_id, [])
        if len(verdicts) < expected:
            raise RuntimeError(
                f"issue type classifier incomplete classifications for {issue_id}: "
                f"{len(verdicts)}/{expected} model verdicts"
            )
        bad_models = [
            str(verdict.get("model") or "?")
            for verdict in verdicts
            if verdict.get("status") != "ok"
        ]
        if bad_models:
            raise RuntimeError(
                f"issue type classifier parse failure for {issue_id}: "
                f"non-ok models={bad_models}"
            )


# issue 1건에 대해 최종 유형/점수/composite 라우팅 여부를 포함한 분류 레코드 구성
def _classification_record(
    ref: dict[str, Any],
    verdicts: list[dict[str, Any]],
    *,
    model_weights: dict[str, float],
    low_margin_threshold: float,
    expected_model_count: int,
) -> dict[str, Any]:
    issue = ref["issue"]
    (
        provisional_type,
        ensemble_confidence,
        weighted_scores,
        used_model_weights,
        missing_model_weight,
        low_margin,
        margin,
    ) = _choose_final_type(verdicts, model_weights, low_margin_threshold)
    final_type, routing_reasons = _apply_composite_routing(
        provisional_type=provisional_type,
        low_margin=low_margin,
        verdicts=verdicts,
        expected_model_count=expected_model_count,
    )
    return {
        "id": ref["id"],
        "list_key": ref["list_key"],
        "index": ref["index"],
        "issue_id": issue.get("issue_id", ""),
        "claim_id": issue.get("claim_id", ""),
        "resolved_claim": issue.get("resolved_claim", ""),
        "claim_text": issue.get("claim_text", ""),
        "issue": issue.get("issue", ""),
        "basis_code": issue.get("basis_code", ""),
        "context_id": issue.get("context_id", ""),
        "context_ids": issue.get("context_ids", []),
        "slide_number": issue.get("slide_number"),
        "start_time": issue.get("start_time"),
        "end_time": issue.get("end_time"),
        "final_issue_type": final_type,
        "final_issue_type_label": _issue_type_label(final_type),
        "ensemble_confidence": ensemble_confidence,
        "weighted_scores": weighted_scores,
        "model_weights": used_model_weights,
        "missing_model_weight": missing_model_weight,
        "low_margin": bool(provisional_type and low_margin),
        "margin": margin,
        "routing_reasons": routing_reasons,
        "routed_to_composite": final_type == COMPOSITE_ISSUE_TYPE,
        "model_count": len(verdicts),
        "model_classifications": verdicts,
    }


# 분류 레코드를 최종 유형별로 그룹화, 유형이 없는 것은 unclassified로 분류
def _group_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = {issue_type: [] for issue_type in ALL_ISSUE_TYPES}
    unclassified = []
    for record in records:
        issue_type = record.get("final_issue_type")
        target = {
            "id": record.get("id"),
            "issue_id": record.get("issue_id"),
            "claim_id": record.get("claim_id"),
            "issue": record.get("issue"),
            "ensemble_confidence": record.get("ensemble_confidence"),
            "routing_reasons": record.get("routing_reasons", []),
            "routed_to_composite": bool(record.get("routed_to_composite")),
            "low_margin": record.get("low_margin", False),
            "margin": record.get("margin", 0.0),
        }
        if issue_type in by_type:
            by_type[issue_type].append(target)
        elif not issue_type:
            unclassified.append(target)
    return {"by_type": by_type, "unclassified": unclassified}


# 확률 dict를 짧은 라벨 기반 한 줄 문자열로 포맷
def _format_probability_vector(probabilities: dict[str, Any] | None) -> str:
    if not isinstance(probabilities, dict):
        return "-"
    parts = []
    for issue_type in ISSUE_TYPES:
        value = float(probabilities.get(issue_type, 0.0) or 0.0)
        short = ISSUE_TYPE_SHORT_LABELS.get(issue_type, issue_type)
        parts.append(f"{short}={value:.2f}")
    return ", ".join(parts)


# issue별 모델 확률/앙상블 점수를 콘솔에 보기 좋게 출력
def _print_classification_score_report(records: list[dict[str, Any]]) -> None:
    if not records:
        print("점수 리포트: 분류 결과 없음")
        return
    print("\n=== issue별 모델 점수 ===")
    for record in records:
        issue_id = record.get("issue_id") or record.get("id") or "?"
        issue_text = str(record.get("issue") or record.get("resolved_claim") or "").strip()
        if len(issue_text) > 72:
            issue_text = issue_text[:69] + "..."
        final_type = record.get("final_issue_type_label") or record.get("final_issue_type") or "?"
        ensemble = float(record.get("ensemble_confidence", 0.0) or 0.0)
        margin = float(record.get("margin", 0.0) or 0.0)
        flags = []
        if record.get("routed_to_composite"):
            flags.append("composite")
        if record.get("low_margin"):
            flags.append("low_margin")
        if record.get("routing_reasons"):
            flags.append("reasons=" + ",".join(record.get("routing_reasons") or []))
        flag_text = f" [{', '.join(flags)}]" if flags else ""
        print(f"\n{issue_id} → {final_type} (ensemble={ensemble:.2f}, margin={margin:.2f}){flag_text}")
        if issue_text:
            print(f"  issue: {issue_text}")
        for verdict in record.get("model_classifications") or []:
            if not isinstance(verdict, dict):
                continue
            model = str(verdict.get("model") or "?")
            top_type = verdict.get("top_issue_type_label") or verdict.get("top_issue_type") or "?"
            top_prob = float(verdict.get("top_probability", 0.0) or 0.0)
            confidence = float(verdict.get("confidence", 0.0) or 0.0)
            probs = _format_probability_vector(verdict.get("probabilities"))
            reason = str(verdict.get("reason") or "").strip()
            if len(reason) > 96:
                reason = reason[:93] + "..."
            print(
                f"  - {model:6s}: {probs} → {top_type} ({top_prob:.2f}), "
                f"confidence={confidence:.2f}"
            )
            if reason:
                print(f"           {reason}")
        weighted = record.get("weighted_scores") or {}
        if weighted:
            weighted_text = ", ".join(
                f"{key.split('_')[0]}={float(weighted.get(key, 0.0) or 0.0):.2f}"
                for key in ISSUE_TYPES
            )
            print(f"  weighted: {weighted_text}")


# 후속 단계로 넘길 최소 필드만 남긴 model_classifications 요약
def _compact_model_classifications(verdicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for verdict in verdicts:
        if not isinstance(verdict, dict):
            continue
        compact.append(
            {
                "model": verdict.get("model", ""),
                "top_issue_type": verdict.get("top_issue_type", ""),
                "top_probability": verdict.get("top_probability", 0.0),
                # 이 reason은 후속 grounding 단계를 위한 검증되지 않은 entity/검색 힌트일 뿐,
                # 외부 사실 근거가 아님
                "reason": str(verdict.get("reason") or "")[:320],
            }
        )
    return compact


# 분류 레코드를 후속 단계 입력용 축소된 item 형식으로 변환
def _next_stage_item(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "issue_id": record.get("issue_id", ""),
        "claim_id": record.get("claim_id", ""),
        "resolved_claim": record.get("resolved_claim", ""),
        "claim_text": record.get("claim_text", ""),
        "basis_code": record.get("basis_code", ""),
        "final_issue_type": record.get("final_issue_type"),
        "final_issue_type_label": record.get("final_issue_type_label", ""),
        "weighted_scores": record.get("weighted_scores", {}),
        "ensemble_confidence": record.get("ensemble_confidence", 0.0),
        "routing_reasons": record.get("routing_reasons", []),
        "routed_to_composite": bool(record.get("routed_to_composite")),
        "low_margin": bool(record.get("low_margin")),
        "margin": record.get("margin", 0.0),
        "model_classifications": _compact_model_classifications(record.get("model_classifications") or []),
        "location": {
            "slide_number": record.get("slide_number"),
            "start_time": record.get("start_time"),
            "end_time": record.get("end_time"),
        },
        "context": {
            "context_id": record.get("context_id", ""),
            "context_ids": record.get("context_ids", []),
        },
    }


# 분류 결과 전체를 유형별로 묶은 후속 단계 입력 JSON으로 변환
def build_next_stage_input(result: dict[str, Any], *, classification_path: str | Path) -> dict[str, Any]:
    issues_by_type = {issue_type: [] for issue_type in ALL_ISSUE_TYPES}
    unclassified = []
    for record in result.get("classifications", []) or []:
        item = _next_stage_item(record)
        issue_type = record.get("final_issue_type")
        if issue_type in issues_by_type:
            issues_by_type[issue_type].append(item)
        else:
            unclassified.append(item)

    summary = result.get("summary") or {}
    return {
        "schema_version": "classified_issue_input.v2",
        "source_classification_path": str(classification_path),
        "source_issue_path": result.get("input_path", ""),
        "generated_at": _now_iso(),
        "categories": result.get("categories", {}),
        "summary": {
            "input_issue_count": summary.get("input_issue_count", 0),
            "breakdown_by_type": summary.get("breakdown_by_type", {}),
            "low_margin_count": summary.get("low_margin_count", 0),
            "composite_count": summary.get("composite_count", 0),
            "routing_reason_breakdown": summary.get("routing_reason_breakdown", {}),
            "unclassified_count": len(unclassified),
        },
        "issues_by_type": issues_by_type,
        "unclassified": unclassified,
    }


# 모델별 판정 개수를 유형별로 집계
def _model_breakdown(model_results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    breakdown = {}
    for model, result in model_results.items():
        rows = result.get("classifications", []) or []
        counts = Counter(
            row.get("top_issue_type") if row.get("status") == "ok" else row.get("status", "unclassified")
            for row in rows
        )
        breakdown[model] = {
            "status": result.get("status", ""),
            "provider": result.get("provider", ""),
            "resolved_model": result.get("resolved_model", ""),
            "classified_count": sum(counts.get(issue_type, 0) for issue_type in ISSUE_TYPES),
            "parse_failed_count": counts.get("parse_failed", 0),
            "breakdown_by_type": {key: counts.get(key, 0) for key in ISSUE_TYPES},
        }
        if counts.get("parse_failed", 0):
            breakdown[model]["breakdown_by_type"]["parse_failed"] = counts.get("parse_failed", 0)
        if result.get("error"):
            breakdown[model]["error"] = result.get("error")
    return breakdown


# issue 목록을 여러 모델로 병렬 분류하고, 가중 앙상블로 최종 유형 결정까지 수행하는 전체 파이프라인
def classify_issues(
    payload: dict[str, Any],
    *,
    input_path: str | Path,
    merged_clean_path: str | Path | None,
    models: list[str],
    list_keys: list[str],
    batch_size: int,
    current_date: str,
    max_tokens: int,
    max_workers: int,
    limit: int | None = None,
    dry_run: bool = False,
    model_weights_spec: str | None = None,
    low_margin_threshold: float | None = None,
    progress_notify: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    _load_env()
    if not models:
        raise RuntimeError("오류 유형 분류에 사용할 모델이 선택되지 않았습니다.")
    resolved_low_margin_threshold = resolve_low_margin_threshold(low_margin_threshold)
    refs = collect_issues(payload, list_keys)
    if limit is not None:
        refs = refs[: max(0, limit)]

    model_results: dict[str, dict[str, Any]] = {}
    if dry_run:
        for model in models:
            try:
                resolved = _resolve_model_spec(model)
            except Exception:
                resolved = {"provider": "unknown", "resolved_model": model}
            model_results[model] = {
                "model": model,
                "provider": resolved["provider"],
                "resolved_model": resolved["resolved_model"],
                "status": "dry_run",
                "classifications": [],
                "token_usage": {},
            }
    else:
        worker_args = []
        for model in models:
            try:
                resolved = _resolve_model_spec(model)
            except Exception:
                resolved = {"provider": "unknown", "resolved_model": model}
            batches = _chunk(refs, batch_size)
            model_results[model] = {
                "model": model,
                "provider": resolved["provider"],
                "resolved_model": resolved["resolved_model"],
                "status": "ok",
                "classifications": [],
                "token_usage_by_batch": [],
                "_batch_results": [],
                "batch_errors": [],
            }
            print(
                f"  [{model}] 준비: provider={resolved['provider']}, model={resolved['resolved_model']}, "
                f"issues={len(refs)}, batches={len(batches)}",
                flush=True,
            )
            for batch_index, batch in enumerate(batches, start=1):
                worker_args.append((model, batch, batch_index, len(batches), current_date, max_tokens))

        print(
            f"\n  ── issue type ensemble 분류 시작 "
            f"({len(refs)}건, {len(models)}모델, batch_size={batch_size}, "
            f"workers_per_model={max_workers}) ──",
            flush=True,
        )
        failed_batch_args: list[tuple] = []
        total_batch_count = len(worker_args)
        progress_lock = threading.Lock()
        progress_done = 0

        def _tick_progress() -> None:
            nonlocal progress_done
            if not progress_notify:
                return
            with progress_lock:
                progress_done += 1
                done = progress_done
            progress_notify(done, total_batch_count)
        # ``max_workers``는 전체 한도가 아니라 모델당 한도, 공급자별
        # executor를 분리해 한 모델의 대기/제한이 다른 모델의 슬롯을 막지 않음
        worker_args_by_model: dict[str, list[tuple]] = defaultdict(list)
        for args in worker_args:
            worker_args_by_model[args[0]].append(args)

        def _run_model_batches(model: str, model_args: list[tuple]) -> tuple[str, list[tuple], list[tuple[tuple, Exception]]]:
            completed: list[tuple] = []
            failed: list[tuple[tuple, Exception]] = []
            with ThreadPoolExecutor(max_workers=min(max_workers, len(model_args))) as executor:
                future_map = {executor.submit(_batch_worker, args): args for args in model_args}
                for future in as_completed(future_map):
                    args = future_map[future]
                    try:
                        completed.append(future.result())
                    except Exception as exc:
                        failed.append((args, exc))
                    _tick_progress()
            return model, completed, failed

        with ThreadPoolExecutor(max_workers=max(1, len(worker_args_by_model))) as model_executor:
            model_futures = {
                model_executor.submit(_run_model_batches, model, model_args): model
                for model, model_args in worker_args_by_model.items()
            }
            for model_future in as_completed(model_futures):
                model, completed, failed = model_future.result()
                for result in completed:
                    _append_batch_result(model_results, result)
                for args, exc in failed:
                    print(f"  ✗ [{model}] batch 작업 실패: {exc}", flush=True)
                    _append_batch_error(model_results, args, exc)
                    failed_batch_args.append(args)

        if failed_batch_args:
            wait_sec = _env_float("ISSUE_TYPE_CLASSIFIER_RETRY_WAIT_SEC", 0.0, min_value=0.0)
            if wait_sec:
                print(f"\n  ── 실패 batch {wait_sec:g}초 후 1회 재시도 ──", flush=True)
                time.sleep(wait_sec)
            else:
                print("\n  ── 실패 batch 1회 재시도 ──", flush=True)
            for args in failed_batch_args:
                model = args[0]
                try:
                    result = _batch_worker(args)
                    _append_batch_result(model_results, result)
                    print(f"  ✓ [{model}] 실패 batch {args[2]}/{args[3]} 재시도 성공", flush=True)
                except Exception as exc:
                    print(f"  ✗ [{model}] 실패 batch {args[2]}/{args[3]} 재시도 실패: {exc}", flush=True)
                    _append_batch_error(model_results, args, exc)

        for model, result in model_results.items():
            batches = sorted(result.pop("_batch_results", []), key=lambda item: item[0])
            result["classifications"] = [
                row
                for _batch_index, rows in batches
                for row in rows
            ]
            usages = result.pop("token_usage_by_batch", [])
            result["token_usage"] = _aggregate_token_usage(usages)
            batch_errors = result.get("batch_errors", [])
            if batch_errors:
                successful_batches = {batch_index for batch_index, _rows in batches}
                unresolved_errors = [
                    error for error in batch_errors
                    if int(error.get("batch_index", 0) or 0) not in successful_batches
                ]
                result["batch_errors"] = unresolved_errors
                if unresolved_errors:
                    result["status"] = "failed"
                    result["error"] = "; ".join(
                        f"batch {row.get('batch_index')}: {row.get('error')}"
                        for row in unresolved_errors
                    )
            if not result.get("batch_errors"):
                result.pop("batch_errors", None)
            print(f"  ✓ [{model}] 모델 작업 완료: {len(result['classifications'])}건", flush=True)

    model_weights = _parse_model_weights(model_weights_spec, models, model_results)
    verdicts_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in model_results.values():
        for row in result.get("classifications", []) or []:
            verdicts_by_id[str(row.get("id", ""))].append(row)
    if not dry_run:
        _validate_classification_completeness(
            models=models,
            model_results=model_results,
            refs=refs,
            verdicts_by_id=verdicts_by_id,
        )
    records = [
        _classification_record(
            ref,
            verdicts_by_id.get(ref["id"], []),
            model_weights=model_weights,
            low_margin_threshold=resolved_low_margin_threshold,
            expected_model_count=len(models),
        )
        for ref in refs
    ]

    type_counts = Counter(record.get("final_issue_type") or "unclassified" for record in records)
    failed_models = [model for model, row in model_results.items() if row.get("status") == "failed"]
    model_type_breakdown = _model_breakdown(model_results)
    low_margin_count = sum(1 for record in records if record.get("low_margin"))
    composite_count = sum(1 for record in records if record.get("final_issue_type") == COMPOSITE_ISSUE_TYPE)
    routed_to_composite_count = sum(1 for record in records if record.get("routed_to_composite"))
    routing_reason_breakdown = Counter(
        reason
        for record in records
        for reason in (record.get("routing_reasons") or [])
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "issue_type_classifier",
        "generated_at": _now_iso(),
        "input_path": str(input_path),
        "merged_clean_path": str(merged_clean_path or ""),
        "current_date": current_date,
        "issue_list_keys": list_keys,
        "models": models,
        "model_weights": model_weights,
        "low_margin_threshold": resolved_low_margin_threshold,
        "dry_run": dry_run,
        "categories": {
            issue_type: _issue_type_label(issue_type)
            for issue_type in ALL_ISSUE_TYPES
        },
        "summary": {
            "input_issue_count": len(refs),
            "model_count": len(models),
            "failed_model_count": len(failed_models),
            "failed_models": failed_models,
            "breakdown_by_type": dict(type_counts),
            "low_margin_count": low_margin_count,
            "composite_count": composite_count,
            "routed_to_composite_count": routed_to_composite_count,
            "routing_reason_breakdown": dict(routing_reason_breakdown),
            "model_breakdown_by_type": model_type_breakdown,
        },
        "model_results": model_results,
        "classifications": records,
        "grouped_results": _group_results(records),
    }


# 입력 파일명에서 _issue_judge 접미어를 떼고 _issue_types.json 출력 경로 생성
def _default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith("_issue_judge"):
        stem = stem[: -len("_issue_judge")]
    return input_path.with_name(f"{stem}_issue_types.json")


# 출력 파일명에서 _issue_types 접미어를 떼고 _classified_issues.json 경로 생성
def _default_next_input_path(output_path: Path) -> Path:
    stem = output_path.stem
    if stem.endswith("_issue_types"):
        stem = stem[: -len("_issue_types")]
    return output_path.with_name(f"{stem}_classified_issues.json")


# CLI 인자 파서 구성
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="issue-judge가 선택한 issue를 composite routing을 포함한 4가지 issue 유형으로 분류",
    )
    parser.add_argument("input_json", help="verifier issue judge JSON 경로")
    parser.add_argument("-o", "--output", help="출력 JSON 경로")
    parser.add_argument(
        "--merged-clean",
        default=None,
        help="출력 메타데이터 호환성을 위해 유지하는 merged_clean JSON 경로",
    )
    parser.add_argument(
        "--models",
        default=",".join(_default_models()),
        help="콤마/공백 구분 모델 목록, 기본값은 issue_classify 스테이지 바인딩",
    )
    parser.add_argument(
        "--issue-list-keys",
        default=",".join(DEFAULT_LIST_KEYS),
        help="분류 대상 JSON list key(콤마/공백 구분), 기본값: issues",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(
            os.getenv(
                "VERIFIER_ISSUE_CLASSIFIER_BATCH_SIZE",
                os.getenv("ISSUE_TYPE_CLASSIFIER_BATCH_SIZE", "20"),
            )
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("ISSUE_TYPE_CLASSIFIER_MAX_TOKENS", "8192")))
    parser.add_argument("--max-workers", type=int, default=int(os.getenv("ISSUE_TYPE_CLASSIFIER_MAX_WORKERS", "20")))
    parser.add_argument("--current-date", default=os.getenv("ISSUE_TYPE_CLASSIFIER_CURRENT_DATE", "2026-05-12"))
    parser.add_argument(
        "--model-weights",
        default=None,
        help="사용 중단된 호환 옵션, 선택된 모델은 항상 동일 가중치를 받음",
    )
    parser.add_argument(
        "--low-margin-threshold",
        type=float,
        default=None,
        help=(
            "1위 점수와 2위 점수 차이가 이 값 미만이면 low_margin으로 표시 "
            f"(기본값: {DEFAULT_LOW_MARGIN_THRESHOLD}, 환경변수: ISSUE_TYPE_CLASSIFIER_LOW_MARGIN_THRESHOLD)"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="빠른 테스트용 issue 개수 제한(선택)")
    parser.add_argument("--dry-run", action="store_true", help="LLM 호출 없이 입출력 형식만 검증")
    parser.add_argument(
        "--next-input-output",
        default=None,
        help="후속 단계 입력용 축소 JSON 경로, 기본값은 classifier 출력과 같은 위치의 *_classified_issues.json",
    )
    parser.add_argument(
        "--no-next-input",
        action="store_true",
        help="후속 단계 입력용 축소 JSON을 작성하지 않음",
    )
    parser.add_argument(
        "--print-scores",
        action="store_true",
        help="분류 완료 후 issue별 모델 확률/앙상블 점수를 stdout에 출력",
    )
    parser.add_argument(
        "--scores-only",
        action="store_true",
        help="기존 *_issue_types.json을 읽어 점수 리포트만 출력 (LLM 호출 없음)",
    )
    return parser


# CLI 진입점, 분류 실행 후 결과/후속 입력 JSON 저장 및 요약 출력
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input_json)
    output_path = Path(args.output) if args.output else _default_output_path(input_path)
    merged_clean_path = Path(args.merged_clean) if args.merged_clean else _guess_merged_clean_path(input_path)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("입력 JSON 최상위 객체는 dict여야 합니다.")

    if args.scores_only:
        if payload.get("stage") != "issue_type_classifier":
            raise ValueError("--scores-only 입력은 issue_type_classifier 출력 JSON(*_issue_types.json)이어야 합니다.")
        _print_classification_score_report(payload.get("classifications", []) or [])
        return 0

    models = _split_csv(args.models)
    if not models:
        raise ValueError("issue_classify 단계에서 사용할 모델을 선택해야 합니다.")

    result = classify_issues(
        payload,
        input_path=input_path,
        merged_clean_path=merged_clean_path,
        models=models,
        list_keys=_split_csv(args.issue_list_keys) or list(DEFAULT_LIST_KEYS),
        batch_size=max(1, args.batch_size),
        current_date=args.current_date,
        max_tokens=max(256, args.max_tokens),
        max_workers=max(1, args.max_workers),
        limit=args.limit,
        dry_run=args.dry_run,
        model_weights_spec=args.model_weights,
        low_margin_threshold=args.low_margin_threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    next_input_path = None
    if not args.no_next_input:
        next_input_path = Path(args.next_input_output) if args.next_input_output else _default_next_input_path(output_path)
        next_input = build_next_stage_input(result, classification_path=output_path)
        next_input_path.parent.mkdir(parents=True, exist_ok=True)
        next_input_path.write_text(json.dumps(next_input, ensure_ascii=False, indent=2), encoding="utf-8")
        result.setdefault("artifacts", {})["classified_issues"] = str(next_input_path)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = result["summary"]
    print(f"입력 issue: {summary['input_issue_count']}건")
    print(f"모델: {', '.join(models)}")
    print(f"모델 가중치: {json.dumps(result.get('model_weights', {}), ensure_ascii=False)}")
    print(f"출력: {output_path}")
    if next_input_path:
        print(f"다음 단계 입력: {next_input_path}")
    if summary["failed_model_count"]:
        print(f"실패 모델: {', '.join(summary['failed_models'])}")
    print(f"유형별 분포: {json.dumps(summary['breakdown_by_type'], ensure_ascii=False)}")
    print(f"low_margin: {summary.get('low_margin_count', 0)}건")
    print(f"composite: {summary.get('composite_count', 0)}건")
    if summary.get("routing_reason_breakdown"):
        print(f"composite 사유: {json.dumps(summary['routing_reason_breakdown'], ensure_ascii=False)}")
    print("모델별 판정 갯수:")
    for model, breakdown in (summary.get("model_breakdown_by_type") or {}).items():
        counts = breakdown.get("breakdown_by_type", {})
        print(
            f"  - {model} ({breakdown.get('status')} / {breakdown.get('resolved_model')}): "
            f"{json.dumps(counts, ensure_ascii=False)}"
        )
    if args.print_scores:
        _print_classification_score_report(result.get("classifications", []) or [])
    return 0 if not summary["failed_model_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
