"""Web grounding for classified issue verifier results.

This stage runs after ``classified_issue_verifier`` and checks externally
verifiable factual_error and temporal_error issues with Gemini Search.
It only grounds surfaced verifier candidates and can lower refuted candidates
below the rejected threshold.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from google.genai import types

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import get_gemini_client_sequence
from utils import api_call_with_retry, is_retryable_api_error


GROUNDABLE_CATEGORIES = {"factual_error", "temporal_error"}
GROUNDING_STATUSES = {
    "supports_issue",
    "refutes_issue",
    "insufficient_evidence",
    "grounding_unavailable",
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


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _strip_json_fence(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in {float("inf"), float("-inf")}:
        return default
    return number


def _clamp01(value: Any, default: float = 0.0) -> float:
    return round(max(0.0, min(1.0, _safe_float(value, default))), 6)


def _rejected_threshold() -> float:
    return _safe_float(os.getenv("CLASSIFIED_ISSUE_VERIFIER_REJECTED_THRESHOLD"), 0.20)


def _confirmed_threshold() -> float:
    return _safe_float(os.getenv("CLASSIFIED_ISSUE_VERIFIER_CONFIRMED_THRESHOLD"), 0.80)


def _status_from_score(score: float) -> str:
    if score >= _confirmed_threshold():
        return "confirmed"
    if score <= _rejected_threshold():
        return "rejected"
    return "professor_check"


def _usage_from_gemini(resp: Any, model: str) -> dict[str, Any]:
    usage = getattr(resp, "usage_metadata", None)
    return {
        "provider": "google",
        "model": model,
        "input_tokens": _safe_int(
            getattr(usage, "prompt_token_count", None) or getattr(usage, "promptTokenCount", None)
        ),
        "output_tokens": _safe_int(
            getattr(usage, "candidates_token_count", None) or getattr(usage, "candidatesTokenCount", None)
        ),
        "reasoning_tokens": _safe_int(
            getattr(usage, "thoughts_token_count", None) or getattr(usage, "thoughtsTokenCount", None)
        ),
        "tool_input_tokens": _safe_int(
            getattr(usage, "tool_use_prompt_token_count", None)
            or getattr(usage, "toolUsePromptTokenCount", None)
        ),
        "cached_input_tokens": _safe_int(
            getattr(usage, "cached_content_token_count", None)
            or getattr(usage, "cachedContentTokenCount", None)
        ),
        "cache_creation_input_tokens": 0,
        "total_tokens": _safe_int(
            getattr(usage, "total_token_count", None) or getattr(usage, "totalTokenCount", None)
        ),
    }


def _empty_token_usage() -> dict[str, int]:
    return {field: 0 for field in TOKEN_USAGE_FIELDS}


def _merge_token_usage(total: dict[str, int], usage: dict[str, Any]) -> dict[str, int]:
    for field in TOKEN_USAGE_FIELDS:
        total[field] = int(total.get(field, 0) or 0) + _safe_int(usage.get(field))
    return total


def _slide_text(issue: dict[str, Any]) -> str:
    judge_context = issue.get("judge_context") if isinstance(issue.get("judge_context"), dict) else {}
    slide = judge_context.get("slide") if isinstance(judge_context.get("slide"), dict) else {}
    return str(slide.get("slide_text") or slide.get("t1") or "")[:1200]


def _context_text(issue: dict[str, Any]) -> str:
    judge_context = issue.get("judge_context") if isinstance(issue.get("judge_context"), dict) else {}
    bundle = judge_context.get("context_bundle") if isinstance(judge_context.get("context_bundle"), dict) else {}
    rows = []
    for key in ("target_contexts", "neighbor_contexts"):
        values = bundle.get(key) if isinstance(bundle.get(key), list) else []
        for row in values:
            if not isinstance(row, dict):
                continue
            cid = row.get("context_id", "")
            text = str(row.get("text", "") or "").strip()
            if text:
                rows.append(f"{cid}: {text}")
    return "\n".join(dict.fromkeys(rows))[:2500]


def _model_reason_summary(issue: dict[str, Any]) -> str:
    rows = []
    for verdict in issue.get("model_judgments", []) or []:
        if not isinstance(verdict, dict):
            continue
        model = verdict.get("model", "")
        score = verdict.get("final_model_score", 0.0)
        reason = str(verdict.get("reason", "") or "").strip()
        rows.append(f"- {model} final={score}: {reason}")
    return "\n".join(rows)[:2500]


def _build_grounding_prompt(issue: dict[str, Any], current_date: str) -> str:
    return f"""You are the web grounding stage for a lecture issue verifier.
Current date: {current_date}

Your task is to check whether external web evidence supports or refutes the issue.
Only judge externally verifiable facts. Do not overrule lecture-context judgments unless web evidence clearly supports/refutes the factual claim.

Issue category: {issue.get("category", "")}
Resolved claim: {issue.get("resolved_claim", "")}
Original claim_text: {issue.get("claim_text", "")}
Verifier score percent: {issue.get("final_severity_percent", "")}

Lecture context:
{_context_text(issue) or "(none)"}

Slide text:
{_slide_text(issue) or "(none)"}

Verifier model reasons:
{_model_reason_summary(issue) or "(none)"}

Decision rules:
- supports_issue: reliable web evidence indicates the resolved claim is false, outdated, or misleading in the way the issue suggests.
- refutes_issue: reliable web evidence indicates the resolved claim is true or acceptable in the relevant region/time/context, so the issue should likely be lowered or rejected.
- insufficient_evidence: web evidence is weak, mixed, irrelevant, or the issue is mainly conceptual/contextual rather than externally searchable.
- grounding_unavailable: search/tool failed.

Prefer official or primary sources. For brand/product specs, use the relevant region/market when available.
If the lecture claim is a concrete example value whose truth depends on brand, product, region, policy, current price, capacity, weight, version, support status, or statistics, do not rely on generic common knowledge; use web evidence.
If the web evidence only addresses a different unit, region, product, or time period, mark insufficient_evidence.
Do not mark a claim false only because a source does not explicitly state it. Use claim_false only when reliable sources directly contradict the claim.
If the claim can only be supported or rejected by indirect arithmetic, unofficial blogs, screenshots, or mixed sources, use uncertain unless the official source clearly supplies all required values.

Return exactly these five lines. Do not use JSON or markdown.
CLAIM_VERDICT=claim_true | claim_false | uncertain
STATUS=supports_issue | refutes_issue | insufficient_evidence | grounding_unavailable
REASON=one or two Korean sentences explaining the web-grounded judgment
SOURCES=URL1, URL2
SUMMARY=short Korean summary of the evidence

Consistency requirements:
- If claim_verdict is claim_false, issue_supported must be true and status must be supports_issue.
- If claim_verdict is claim_true, issue_supported must be false and status must be refutes_issue.
- If claim_verdict is uncertain, issue_supported must be null and status must be insufficient_evidence.
"""


def _normalize_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    return status if status in GROUNDING_STATUSES else "insufficient_evidence"


def _parse_response(text: str) -> dict[str, Any]:
    raw = _strip_json_fence(text)
    try:
        payload = json.loads(raw, strict=False)
    except Exception:
        payload = _parse_line_response(raw)
    if not isinstance(payload, dict):
        return {}
    claim_verdict = str(payload.get("claim_verdict", "") or "").strip().lower()
    issue_supported = payload.get("issue_supported")
    if isinstance(issue_supported, str):
        lowered = issue_supported.strip().lower()
        if lowered in {"true", "yes", "1"}:
            issue_supported = True
        elif lowered in {"false", "no", "0"}:
            issue_supported = False
        else:
            issue_supported = None
    reason = str(payload.get("reason", "") or "").strip()
    evidence_summary = str(payload.get("evidence_summary", "") or "").strip()
    uncertainty_text = f"{reason}\n{evidence_summary}".lower()
    has_uncertainty_marker = any(
        marker in uncertainty_text
        for marker in (
            "추정",
            "명시되어 있지",
            "찾을 수 없",
            "확인되지",
            "불확실",
            "mixed",
            "uncertain",
            "not explicitly",
            "not directly",
        )
    )
    if claim_verdict == "claim_false" and has_uncertainty_marker:
        claim_verdict = "uncertain"

    if claim_verdict == "claim_false":
        status = "supports_issue"
        issue_supported = True
    elif claim_verdict == "claim_true":
        status = "refutes_issue"
        issue_supported = False
    elif claim_verdict == "uncertain":
        status = "insufficient_evidence"
        issue_supported = None
    else:
        status = _normalize_status(payload.get("status"))
    sources = payload.get("evidence_sources", [])
    if not isinstance(sources, list):
        sources = []
    sources = [
        str(source).strip()
        for source in sources
        if re.match(r"^https?://", str(source).strip(), flags=re.IGNORECASE)
    ]
    if status in {"supports_issue", "refutes_issue"} and not sources:
        status = "insufficient_evidence"
        claim_verdict = "uncertain"
        issue_supported = None
    return {
        "status": status,
        "claim_verdict": claim_verdict or "uncertain",
        "issue_supported": issue_supported if isinstance(issue_supported, bool) else None,
        "reason": reason,
        "evidence_sources": sources,
        "evidence_summary": evidence_summary,
    }


def _parse_line_response(text: str) -> dict[str, Any]:
    fields = {
        "claim_verdict": "",
        "status": "",
        "reason": "",
        "evidence_sources": [],
        "evidence_summary": "",
    }
    key_map = {
        "CLAIM_VERDICT": "claim_verdict",
        "STATUS": "status",
        "REASON": "reason",
        "SOURCES": "evidence_sources",
        "SUMMARY": "evidence_summary",
    }
    matches = list(re.finditer(r"(?m)^(CLAIM_VERDICT|STATUS|REASON|SOURCES|SUMMARY)\s*=\s*", text))
    if not matches:
        raise ValueError("grounding response is neither JSON nor key-value lines")
    for index, match in enumerate(matches):
        key = key_map[match.group(1)]
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[start:end].strip()
        if key == "evidence_sources":
            lowered = value.lower()
            if lowered in {"", "none", "n/a", "없음"}:
                fields[key] = []
            else:
                fields[key] = [
                    re.sub(r"\s+", "", part)
                    for part in value.split(",")
                    if re.sub(r"\s+", "", part)
                ]
        else:
            fields[key] = value
    return fields


def _call_grounding(issue: dict[str, Any], current_date: str, max_tokens: int) -> tuple[dict[str, Any], dict[str, int]]:
    model = os.getenv("CLASSIFIED_ISSUE_GROUNDING_MODEL", "gemini-2.5-flash").strip()
    prompt = _build_grounding_prompt(issue, current_date)
    contents = [types.Part.from_text(text=prompt)]
    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=max_tokens,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    last_exc: Exception | None = None
    for index, (client_name, client) in enumerate(get_gemini_client_sequence()):
        try:
            def call_api():
                return client.models.generate_content(model=model, contents=contents, config=config)

            resp = api_call_with_retry(call_api)
            payload = _parse_response(resp.text or "")
            payload["model"] = model
            payload["provider"] = "google"
            payload["client"] = client_name
            return payload, _usage_from_gemini(resp, model)
        except Exception as exc:
            last_exc = exc
            if is_retryable_api_error(exc) and index < len(get_gemini_client_sequence()) - 1:
                continue
            break
    return {
        "status": "grounding_unavailable",
        "reason": f"grounding 실패: {last_exc}",
        "evidence_sources": [],
        "evidence_summary": "",
        "model": model,
        "provider": "google",
    }, _empty_token_usage()


def _should_ground(issue: dict[str, Any], categories: set[str]) -> bool:
    if str(issue.get("category") or "").strip() not in categories:
        return False
    return _clamp01(issue.get("final_severity_score")) > _rejected_threshold()


def _apply_grounding_decision(issue: dict[str, Any], payload: dict[str, Any]) -> None:
    if payload.get("status") != "refutes_issue":
        return
    original_score = _clamp01(issue.get("final_severity_score"))
    rejected_score = max(0.0, _rejected_threshold() - 0.001)
    issue["pre_grounding_final_severity_score"] = original_score
    issue["pre_grounding_final_severity_percent"] = round(original_score * 100.0, 2)
    issue["pre_grounding_status"] = _status_from_score(original_score)
    issue["final_severity_score"] = min(original_score, rejected_score)
    issue["final_severity_percent"] = round(float(issue["final_severity_score"]) * 100.0, 2)
    issue["needs_manual_review"] = False
    issue["rejected_by_web_grounding"] = True


def _refresh_summary(verifier_result: dict[str, Any]) -> None:
    summary = verifier_result.get("summary")
    issues = verifier_result.get("all_issues", []) or []
    if not isinstance(summary, dict):
        return
    summary["high_severity_count"] = sum(
        1 for issue in issues if _clamp01((issue or {}).get("final_severity_score")) >= _confirmed_threshold()
    )
    summary["needs_manual_review_count"] = sum(
        1 for issue in issues if isinstance(issue, dict) and bool(issue.get("needs_manual_review"))
    )
    summary["web_grounding_rejected_count"] = sum(
        1 for issue in issues if isinstance(issue, dict) and bool(issue.get("rejected_by_web_grounding"))
    )


def ground_classified_issues(
    verifier_result: dict[str, Any],
    *,
    current_date: str,
    max_workers: int = 3,
    max_tokens: int = 2048,
    categories: set[str] | None = None,
) -> dict[str, Any]:
    categories = categories or set(GROUNDABLE_CATEGORIES)
    issues = verifier_result.get("all_issues", []) or []
    targets = [issue for issue in issues if isinstance(issue, dict) and _should_ground(issue, categories)]
    token_usage = _empty_token_usage()
    status_counts: Counter[str] = Counter()

    if not targets:
        verifier_result["grounding"] = {
            "enabled": True,
            "grounded_issue_count": 0,
            "categories": sorted(categories),
            "status_counts": {},
            "token_usage": token_usage,
        }
        return verifier_result

    print(f"  classified issue grounding 시작: {len(targets)}건 ({', '.join(sorted(categories))})", flush=True)

    by_id: dict[str, dict[str, Any]] = {}

    def worker(issue: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, int]]:
        payload, usage = _call_grounding(issue, current_date, max_tokens)
        return str(issue.get("id") or issue.get("issue_id") or ""), payload, usage

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {executor.submit(worker, issue): issue for issue in targets}
        for future in as_completed(futures):
            issue = futures[future]
            try:
                issue_id, payload, usage = future.result()
            except Exception as exc:
                issue_id = str(issue.get("id") or issue.get("issue_id") or "")
                payload = {
                    "status": "grounding_unavailable",
                    "reason": f"grounding 실패: {exc}",
                    "evidence_sources": [],
                    "evidence_summary": "",
                }
                usage = _empty_token_usage()
            by_id[issue_id] = payload
            _merge_token_usage(token_usage, usage)
            status_counts[_normalize_status(payload.get("status"))] += 1
            print(f"    grounding {issue_id}: {payload.get('status')}", flush=True)

    for issue in issues:
        if not isinstance(issue, dict):
            continue
        issue_id = str(issue.get("id") or issue.get("issue_id") or "")
        payload = by_id.get(issue_id)
        if payload:
            issue["web_grounding"] = payload
            _apply_grounding_decision(issue, payload)
        else:
            if str(issue.get("category") or "").strip() not in categories:
                reason = "factual_error/temporal_error가 아니어서 web grounding을 실행하지 않음"
            else:
                reason = "verifier 최종 상태가 rejected라서 web grounding을 실행하지 않음"
            issue["web_grounding"] = {
                "status": "not_applicable",
                "reason": reason,
                "evidence_sources": [],
                "evidence_summary": "",
            }

    verifier_result["grounding"] = {
        "enabled": True,
        "grounded_issue_count": len(targets),
        "categories": sorted(categories),
        "status_counts": dict(status_counts),
        "token_usage": token_usage,
    }
    verifier_result["summary"]["grounded_issue_count"] = len(targets)
    verifier_result["summary"]["grounding_status_counts"] = dict(status_counts)
    _refresh_summary(verifier_result)
    return verifier_result


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run web grounding over classified issue verifier output.")
    parser.add_argument("verifier_json")
    parser.add_argument("-o", "--output")
    parser.add_argument("--current-date", default=os.getenv("CLASSIFIED_ISSUE_GROUNDING_CURRENT_DATE", "2026-05-31"))
    parser.add_argument("--max-workers", type=int, default=int(os.getenv("CLASSIFIED_ISSUE_GROUNDING_MAX_WORKERS", "3")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("CLASSIFIED_ISSUE_GROUNDING_MAX_TOKENS", "2048")))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.verifier_json)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_grounded.json")
    result = ground_classified_issues(
        _load_json(input_path),
        current_date=args.current_date,
        max_workers=args.max_workers,
        max_tokens=max(256, args.max_tokens),
    )
    result["grounding"]["generated_at"] = _now_iso()
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"입력 issue: {len(result.get('all_issues', []) or [])}건")
    print(f"grounded: {result.get('grounding', {}).get('grounded_issue_count', 0)}건")
    print(f"status: {json.dumps(result.get('grounding', {}).get('status_counts', {}), ensure_ascii=False)}")
    print(f"출력: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
