"""Claim verification common helpers."""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from .runtime_llm import call_runtime_llm, configured_stage_models, resolve_runtime_binding
except ImportError:
    from runtime_llm import call_runtime_llm, configured_stage_models, resolve_runtime_binding


# ── LLM 추상화 레이어 ────────────────────────────────────────


def _resolve_stage_model(stage: str) -> str:
    # Detector workers explicitly set their concrete model because several
    # models execute the same stage in parallel.
    if stage == "judge":
        model = os.getenv("VERIFIER_CLAIM_JUDGE_MODEL", "").strip()
        if model:
            return model

    selected = configured_stage_models(stage)
    if selected:
        return selected[0]
    raise RuntimeError(f"{stage} 단계에 사용할 모델이 선택되지 않았습니다.")


TOKEN_USAGE_STAGES = ("extract", "judge", "slide_error", "slide_error_transcribe")
TOKEN_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "tool_input_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "total_tokens",
)


def _new_token_bucket() -> dict:
    return {field: 0 for field in TOKEN_USAGE_FIELDS}


def _empty_token_usage() -> dict:
    usage = {stage: _new_token_bucket() for stage in TOKEN_USAGE_STAGES}
    usage["total"] = _new_token_bucket()
    return usage


def _safe_int(value) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except Exception:
        return 0


def _merge_token_usage(*usages: dict) -> dict:
    merged = _empty_token_usage()
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        if usage.get("stage"):
            _add_call_usage(merged, usage)
            continue
        for stage in TOKEN_USAGE_STAGES:
            bucket = usage.get(stage)
            if not isinstance(bucket, dict):
                continue
            for field in TOKEN_USAGE_FIELDS:
                merged[stage][field] += _safe_int(bucket.get(field))

    merged["total"] = _new_token_bucket()
    for stage in TOKEN_USAGE_STAGES:
        for field in TOKEN_USAGE_FIELDS:
            merged["total"][field] += merged[stage][field]
    return merged


def _add_call_usage(token_usage: dict, call_usage: dict) -> dict:
    if not isinstance(token_usage, dict):
        token_usage = _empty_token_usage()
    if not isinstance(call_usage, dict):
        return token_usage

    stage = str(call_usage.get("stage", "") or "")
    if not stage:
        return token_usage
    if stage not in token_usage:
        token_usage[stage] = _new_token_bucket()

    for field in TOKEN_USAGE_FIELDS:
        value = _safe_int(call_usage.get(field))
        token_usage[stage][field] += value
        token_usage["total"][field] += value
    return token_usage


def _call_llm(
    prompt: str,
    system_prompt: str = None,
    max_tokens: int = 8192,
    temperature: float = None,
    image_bytes: bytes = None,
    image_bytes_list: list[bytes] = None,
    thinking_budget: int = 1024,
    thinking_level: str = None,
    response_format: dict = None,
    stage: str = "default",
) -> tuple[str, dict]:
    """Call the selected model through the single LiteLLM runtime gateway."""
    temp = temperature if temperature is not None else VERIFIER_TEMPERATURE
    model_spec = _resolve_stage_model(stage)
    runtime_binding = resolve_runtime_binding(stage, model_spec)
    if not runtime_binding:
        raise RuntimeError(
            f"{stage} 단계의 선택 모델을 런타임 바인딩으로 해석하지 못했습니다: {model_spec}"
        )

    runtime_result = call_runtime_llm(
        runtime_binding,
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temp,
        response_format=response_format,
        image_bytes=image_bytes,
        image_bytes_list=image_bytes_list,
        model_spec=model_spec,
        stage=stage,
    )
    if runtime_result is None:
        raise RuntimeError(
            f"{stage} 단계가 지원하지 않는 런타임 프로토콜입니다: {model_spec}"
        )
    text, usage = runtime_result
    usage.setdefault("endpoint_ref", runtime_binding.get("endpoint_ref", ""))
    return text, usage


# ── 설정 ──────────────────────────────────────────────────

BATCH_SIZE = int(os.getenv("VERIFIER_BATCH_SIZE", "15"))
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "")
VERIFIER_CLAIM_EXTRACT_MODEL = os.getenv("VERIFIER_CLAIM_EXTRACT_MODEL", "")
VERIFIER_CLAIM_JUDGE_MODEL = os.getenv("VERIFIER_CLAIM_JUDGE_MODEL", "")
VERIFIER_SLIDE_ERROR_MODEL = os.getenv("VERIFIER_SLIDE_ERROR_MODEL", "")
VERIFIER_SLIDE_ERROR_TRANSCRIBE_MODEL = os.getenv("VERIFIER_SLIDE_ERROR_TRANSCRIBE_MODEL", "")
VERIFIER_TEMPERATURE = float(os.getenv("VERIFIER_TEMPERATURE", "0.0"))
ISSUE_TYPE_LABELS = {
    "factual_error": "사실 오류",
    "temporal_error": "오래된 내용",
    "scope_overclaim": "과도한 일반화",
    "confusing_explanation": "혼동 가능 설명",
    "composite_issue": "복합 오류",
}
VERIFIER_PARSE_RETRIES = int(os.getenv("VERIFIER_PARSE_RETRIES", "2"))
VERIFIER_BATCH_RECOVERY_RETRIES = int(os.getenv("VERIFIER_BATCH_RECOVERY_RETRIES", "1"))
VERIFIER_REQUIRE_COMPLETE = os.getenv("VERIFIER_REQUIRE_COMPLETE", "1") != "0"


def normalize_issue_type(value: str) -> str:
    return str(value or "").strip().lower()


def issue_type_label(issue_type: str) -> str:
    return ISSUE_TYPE_LABELS.get(normalize_issue_type(issue_type), str(issue_type or "unknown"))


def _strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if "```json" in text:
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def build_verification_question(claim: dict) -> str:
    """후속 판정에서 살아남은 claim에만 짧은 검증 질문을 생성."""
    text = str(
        claim.get("resolved_claim")
        or claim.get("claim_text")
        or claim.get("problematic_content")
        or ""
    ).strip()
    if not text:
        return ""

    text = " ".join(text.split()).strip(" \t\r\n.。?？!！")
    if not text:
        return ""

    claim_type = str(claim.get("claim_type", "") or "").strip()
    if claim_type == "numeric":
        return f"'{text}'라는 수치나 기준이 정확한가?"
    if claim_type == "causal":
        return f"'{text}'라는 인과 또는 작동 방식 설명이 타당한가?"
    if claim_type == "relationship":
        return f"'{text}'라는 개념 간 관계 설명이 타당한가?"
    if claim_type == "currentness":
        return f"'{text}'라는 현행성 설명이 현재 기준으로 타당한가?"
    return f"'{text}'라는 설명이 타당한가?"


# ── 도메인 힌트 ──────────────────────────────────────────


def _normalize_sub_domain(value: str) -> str:
    return str(value or "").strip()


def _get_domain_hint(domain: str, sub_domain: str) -> dict:
    domain_value = str(domain or "").strip()
    sub_domain_value = _normalize_sub_domain(sub_domain)
    label_parts = [part for part in (domain_value, sub_domain_value) if part]
    return {"label": " > ".join(label_parts) if label_parts else "일반"}


def _resolve_domain_fields(merged: dict) -> tuple[str, str]:
    """merged.json에서 domain/sub_domain을 읽고 정규화."""
    domain = str(
        merged.get("domain")
        or merged.get("primary_domain")
        or ""
    ).strip()
    sub_domain = str(
        merged.get("sub_domain")
        or merged.get("subdomain")
        or merged.get("secondary_domain")
        or ""
    ).strip()
    sub_domain = _normalize_sub_domain(sub_domain)
    return domain, sub_domain


# ── 유틸리티 ──────────────────────────────────────────────

def _compact_text(value: str) -> str:
    return re.sub(r"\s+", "", (value or "")).lower()


def _issue_key(issue: dict) -> tuple:
    return (
        issue.get("type", ""),
        int(issue.get("slide_number", 0) or 0),
        int(round(float(issue.get("start_time", 0) or 0))),
        _compact_text(str(issue.get("problematic_content", "") or ""))[:60],
    )


def _dedupe_issues(issues: list[dict]) -> list[dict]:
    dedup = {}
    for issue in issues:
        key = _issue_key(issue)
        cur = dedup.get(key)
        if cur is None or float(issue.get("confidence", 0) or 0) > float(cur.get("confidence", 0) or 0):
            dedup[key] = issue
    return sorted(dedup.values(), key=lambda x: float(x.get("start_time", 0) or 0))


def _normalize_severity(issue: dict) -> None:
    severity = str(issue.get("severity", "")).lower()
    if severity not in {"critical", "major", "minor"}:
        severity = "major"
    if severity == "critical" and float(issue.get("confidence", 0) or 0) < 0.9:
        severity = "major"
    issue["severity"] = severity


def _is_asr_artifact(issue: dict, context_map: dict) -> bool:
    cid = issue.get("context_id", "")
    ref = context_map.get(cid)
    if not ref:
        return False
    orig = _compact_text(ref.get("text_original", ""))
    corr = _compact_text(ref.get("text_corrected", ""))
    if not orig or not corr or orig == corr:
        return False
    snippet = _compact_text(str(issue.get("problematic_content", "") or ""))
    if len(snippet) < 6:
        return False
    if snippet in corr:
        return False
    if snippet in orig:
        return True
    return False


def _make_result(
    issues: list[dict],
    api_calls: int,
    parse_failures: int = 0,
    failed_calls: int = 0,
    token_usage: Optional[dict] = None,
) -> dict:
    issues = sorted(issues, key=lambda x: float(x.get("start_time", 0) or 0))
    return {
        "overall_assessment": {
            "has_issues": len(issues) > 0,
            "total_issues": len(issues),
            "severity_breakdown": {
                "critical": sum(1 for i in issues if i.get("severity") == "critical"),
                "major": sum(1 for i in issues if i.get("severity") == "major"),
                "minor": sum(1 for i in issues if i.get("severity") == "minor"),
            },
        },
        "issues": issues,
        "api_calls": api_calls,
        "parse_failures": parse_failures,
        "failed_calls": failed_calls,
        "token_usage": _merge_token_usage(token_usage),
    }


def merge_multiple_runs(
    run_results: list[dict],
    num_runs: int,
    min_detection_rate: float = 0.5,
    time_window_sec: float = 30,
) -> dict:
    issues_by_key = {}
    for result in run_results:
        for issue in result.get("issues", []):
            # ② claim_id 기반 매칭: context_id + claim_text 우선, fallback으로 기존 fuzzy key
            claim_text = _compact_text(str(issue.get("claim_text", "") or ""))
            cid = issue.get("context_id", "")
            if cid and claim_text:
                key = (cid, claim_text[:80])
            else:
                start = float(issue.get("start_time", 0) or 0)
                bucket = round(start / time_window_sec) * time_window_sec
                key = (
                    issue.get("type", ""),
                    int(issue.get("slide_number", 0) or 0),
                    bucket,
                    _compact_text(str(issue.get("problematic_content", "") or ""))[:60],
                )
            if key not in issues_by_key:
                issues_by_key[key] = {
                    "issue": issue,
                    "count": 1,
                    "conf_sum": float(issue.get("confidence", 0) or 0),
                }
            else:
                issues_by_key[key]["count"] += 1
                issues_by_key[key]["conf_sum"] += float(issue.get("confidence", 0) or 0)
                if float(issue.get("confidence", 0) or 0) > float(issues_by_key[key]["issue"].get("confidence", 0) or 0):
                    issues_by_key[key]["issue"] = issue

    all_issues, filtered = [], []
    for data in issues_by_key.values():
        issue = data["issue"].copy()
        issue["detection_count"] = data["count"]
        issue["detection_rate"] = data["count"] / num_runs
        issue["avg_confidence"] = data["conf_sum"] / data["count"]
        all_issues.append(issue)
        if issue["detection_rate"] >= min_detection_rate or (data["count"] >= 2 and issue["avg_confidence"] >= 0.9):
            filtered.append(issue)

    filtered.sort(key=lambda x: float(x.get("start_time", 0) or 0))
    dropped = len(all_issues) - len(filtered)
    return {
        "overall_assessment": {
            "has_issues": len(filtered) > 0,
            "total_issues": len(filtered),
            "severity_breakdown": {
                s: sum(1 for i in filtered if i.get("severity") == s)
                for s in ("critical", "major", "minor")
            },
        },
        "issues": filtered,
        "summary": f"{num_runs}회 판정, 합의 기준 {min_detection_rate:.0%} (시간 창 {time_window_sec:.0f}초). 총 {len(all_issues)}개 후보 중 {len(filtered)}개 확정 ({dropped}개 제외).",
        "verification_method": "claim_consensus_v8",
        "consensus_threshold": min_detection_rate,
        "api_calls": sum(r.get("api_calls", 0) for r in run_results),
        "parse_failures": sum(r.get("parse_failures", 0) for r in run_results),
        "failed_calls": sum(r.get("failed_calls", 0) for r in run_results),
        "token_usage": _merge_token_usage(*(r.get("token_usage") for r in run_results)),
    }


# ── context 수집 + 슬라이드 맥락 ────────────────────────────

def _collect_contexts(slides: list[dict]) -> list[dict]:
    contexts = []
    has_contexts = any(slide.get("contexts") for slide in slides)
    if has_contexts:
        for slide in slides:
            slide_no = int(slide.get("slide_number") or slide.get("slide_index") or 0)
            for fallback_context_index, ctx in enumerate(slide.get("contexts", []) or [], start=1):
                text = str(ctx.get("text", "") or "").strip()
                if not text:
                    continue
                context_id = str(ctx.get("context_id", "") or "").strip()
                if not context_id:
                    raw_context_index = ctx.get("context_index")
                    if raw_context_index is not None:
                        context_index = int(raw_context_index) + 1
                    else:
                        context_index = fallback_context_index
                    context_id = f"S{slide_no:03d}-C{context_index:03d}"
                contexts.append({
                    "context_id": context_id,
                    "slide_number": slide_no,
                    "scene_index": ctx.get("scene_index"),
                    "context_index": ctx.get("context_index"),
                    "start_time": float(ctx.get("start_time", ctx.get("start", 0)) or 0),
                    "end_time": float(ctx.get("end_time", ctx.get("end", ctx.get("start", 0))) or 0),
                    "text": text,
                    "text_corrected": text,
                    "text_original": "",
                    "text_corrected_candidate": "",
                    "correction_status": "context",
                    "correction_risk": "",
                    "correction_reason": "",
                    "source_segment_indices": ctx.get("source_segment_indices", []),
                })
        contexts.sort(key=lambda u: u["start_time"])
        return contexts

    for slide in slides:
        slide_no = int(slide.get("slide_number") or slide.get("slide_index") or 0)
        for seg in slide.get("transcript_segments", []) or []:
            corr = str(seg.get("text", "") or "").strip()
            orig = str(seg.get("text_original", "") or "").strip()
            candidate = str(seg.get("text_corrected_candidate", "") or "").strip()
            text = corr or orig
            if not text:
                continue
            contexts.append({
                "slide_number": slide_no,
                "start_time": float(seg.get("start", 0) or 0),
                "text": text,
                "text_corrected": corr,
                "text_original": orig,
                "text_corrected_candidate": candidate,
                "correction_status": str(seg.get("correction_status", "") or "").strip(),
                "correction_risk": str(seg.get("correction_risk", "") or "").strip(),
                "correction_reason": str(seg.get("correction_reason", "") or "").strip(),
            })
    contexts.sort(key=lambda u: u["start_time"])
    for i, u in enumerate(contexts, start=1):
        slide_no = int(u.get("slide_number", 0) or 0)
        u["context_id"] = f"S{slide_no:03d}-SEG{i:04d}"
    return contexts


def _build_slide_context_map(slides: list[dict]) -> dict:
    """slides 데이터에서 slide_number → {title, time_range, slide_text} 매핑."""
    ctx = {}
    for slide in slides:
        sn = int(slide.get("slide_number") or slide.get("slide_index") or 0)
        if sn <= 0:
            continue
        ctx[sn] = {
            "title": str(slide.get("title", "") or ""),
            "time_range": str(slide.get("time_range", "") or ""),
            "slide_text": str(slide.get("slide_text") or slide.get("text") or "").strip(),
        }
    return ctx


def _format_context_for_prompt(u: dict) -> str:
    cid = u["context_id"]
    ts = f"{u['start_time']:.1f}s"
    corr = str(u.get("text_corrected", "") or "").strip()
    orig = str(u.get("text_original", "") or "").strip()

    if str(u.get("correction_status", "") or "").strip() == "candidate_only":
        return f"{cid} | {ts} | {orig or u.get('text', '')}"

    if corr and orig and corr != orig:
        return f"{cid} | {ts} | 교정: {corr} | 원문: {orig}"

    return f"{cid} | {ts} | {u['text']}"
