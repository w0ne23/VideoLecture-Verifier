"""Web evidence helpers for classified lecture issues.

The primary pipeline retrieves compact, source-verified evidence *before* the final
verifier.  The verifier receives source passages, not a precomputed web verdict or
score.  Legacy post-verifier grounding helpers remain for artifact compatibility.
"""

from __future__ import annotations

import argparse
import copy
from difflib import SequenceMatcher
from html.parser import HTMLParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from google.genai import types

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import get_gemini_client_sequence
from utils import api_call_with_retry, is_retryable_api_error

try:
    from .issue_type_classifier import _call_llm, _resolve_model_spec, _split_csv
except ImportError:
    from verifier.issue_type_classifier import _call_llm, _resolve_model_spec, _split_csv


GROUNDABLE_CATEGORIES = {"factual_error", "temporal_error"}
WEB_QUERY_PLAN_BASIS_CODES = {
    "external_numeric_fact",
    "external_historical_fact",
    "current_status",
    "named_entity_fact",
    "context_unresolved",
    "lecture_internal",
    "interpretive_claim",
    "query_planner_unavailable",
}
GROUNDING_STATUSES = {
    "supports_issue",
    "refutes_issue",
    "insufficient_evidence",
    "grounding_unavailable",
    "not_applicable",
}
SOURCE_PRIORITY_ORDER = {
    "official_docs": 1,
    "standards": 2,
    "government": 2,
    "academic": 3,
    "educational": 4,
    "encyclopedia": 5,
}
SOURCE_PRIORITY_LABELS = {
    1: "official_docs",
    2: "standards_or_government",
    3: "academic",
    4: "educational",
    5: "encyclopedia",
}
EXCLUDED_SOURCE_LEVELS = {"tutorial", "blog", "forum", "weak_secondary"}
HARD_EXCLUDED_SOURCE_DOMAINS = {
    "medium.com": ("personal_blog", "개인 블로그 플랫폼"),
    "tistory.com": ("personal_blog", "개인 블로그 플랫폼"),
    "velog.io": ("personal_blog", "개인 블로그 플랫폼"),
    "blog.naver.com": ("personal_blog", "개인 블로그 플랫폼"),
    "blogspot.com": ("personal_blog", "개인 블로그 플랫폼"),
    "wordpress.com": ("personal_blog", "개인 블로그 플랫폼"),
    "substack.com": ("personal_blog", "개인 뉴스레터·블로그 플랫폼"),
    "brunch.co.kr": ("personal_blog", "개인 콘텐츠 플랫폼"),
    "reddit.com": ("forum", "사용자 포럼"),
    "quora.com": ("forum", "사용자 Q&A"),
    "stackoverflow.com": ("forum", "사용자 Q&A"),
    "stackexchange.com": ("forum", "사용자 Q&A"),
    "namu.wiki": ("user_wiki", "사용자 편집 위키"),
    "fandom.com": ("user_wiki", "사용자 편집 위키"),
    "tutorialspoint.com": ("tutorial", "일반 튜토리얼 사이트"),
    "w3schools.com": ("tutorial", "일반 튜토리얼 사이트"),
    "geeksforgeeks.org": ("tutorial", "일반 튜토리얼 사이트"),
    "freecodecamp.org": ("tutorial", "일반 튜토리얼 사이트"),
    "javatpoint.com": ("tutorial", "일반 튜토리얼 사이트"),
    "programiz.com": ("tutorial", "일반 튜토리얼 사이트"),
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


def _supports_issue_delta() -> float:
    return _safe_float(os.getenv("CLASSIFIED_ISSUE_GROUNDING_SUPPORTS_DELTA"), 0.10)


def _refutes_issue_delta() -> float:
    return _safe_float(os.getenv("CLASSIFIED_ISSUE_GROUNDING_REFUTES_DELTA"), -0.10)


def _status_from_score(score: float) -> str:
    if score >= _confirmed_threshold():
        return "confirmed"
    if score <= _rejected_threshold():
        return "rejected"
    return "professor_check"


def _grounding_model_specs() -> list[str]:
    try:
        from .runtime_llm import configured_stage_models
    except ImportError:
        from runtime_llm import configured_stage_models
    return configured_stage_models("grounding")


def _pre_verifier_evidence_model() -> str:
    configured = _grounding_model_specs()
    return configured[0] if configured else ""


def _pre_verifier_evidence_max_tool_calls() -> int:
    try:
        return max(1, int(os.getenv("CLASSIFIED_ISSUE_EVIDENCE_MAX_TOOL_CALLS", "1")))
    except ValueError:
        return 1


def _pre_verifier_evidence_search_context_size() -> str:
    configured = os.getenv("CLASSIFIED_ISSUE_EVIDENCE_SEARCH_CONTEXT_SIZE", "medium").strip().lower()
    return configured if configured in {"low", "medium", "high"} else "medium"


def _pre_verifier_query_plan_max_tokens() -> int:
    try:
        return max(
            600,
            int(
                os.getenv(
                    "CLASSIFIED_ISSUE_EVIDENCE_QUERY_PLAN_MAX_TOKENS",
                    "1200",
                )
            ),
        )
    except ValueError:
        return 1200


def _pre_verifier_evidence_max_sources() -> int:
    try:
        return max(1, int(os.getenv("CLASSIFIED_ISSUE_EVIDENCE_MAX_SOURCES", "2")))
    except ValueError:
        return 2


def _pre_verifier_evidence_verify_max_sources() -> int:
    try:
        return max(1, int(os.getenv("CLASSIFIED_ISSUE_EVIDENCE_VERIFY_MAX_SOURCES", "3")))
    except ValueError:
        return 3


def _pre_verifier_evidence_max_fetch_attempts() -> int:
    try:
        return max(
            _pre_verifier_evidence_verify_max_sources(),
            int(
                os.getenv(
                    "CLASSIFIED_ISSUE_EVIDENCE_MAX_FETCH_ATTEMPTS",
                    "10",
                )
            ),
        )
    except ValueError:
        return 10


def _pre_verifier_evidence_semantic_model() -> str:
    return _pre_verifier_evidence_model()


def _pre_verifier_evidence_semantic_max_tokens() -> int:
    try:
        return max(128, int(os.getenv("CLASSIFIED_ISSUE_EVIDENCE_SEMANTIC_MAX_TOKENS", "800")))
    except ValueError:
        return 800


def _pre_verifier_evidence_semantic_min_confidence() -> float:
    return _clamp01(
        os.getenv("CLASSIFIED_ISSUE_EVIDENCE_SEMANTIC_MIN_CONFIDENCE", "0.75"),
        0.75,
    )


def _pre_verifier_document_relevance_max_tokens() -> int:
    try:
        return max(
            128,
            int(os.getenv("CLASSIFIED_ISSUE_EVIDENCE_DOCUMENT_RELEVANCE_MAX_TOKENS", "600")),
        )
    except ValueError:
        return 600


def _pre_verifier_document_relevance_min_confidence() -> float:
    return _clamp01(
        os.getenv("CLASSIFIED_ISSUE_EVIDENCE_DOCUMENT_RELEVANCE_MIN_CONFIDENCE", "0.75"),
        0.75,
    )


def _pre_verifier_source_trust_max_tokens() -> int:
    try:
        return max(
            128,
            int(os.getenv("CLASSIFIED_ISSUE_EVIDENCE_SOURCE_TRUST_MAX_TOKENS", "700")),
        )
    except ValueError:
        return 700


def _pre_verifier_source_trust_min_confidence() -> float:
    return _clamp01(
        os.getenv("CLASSIFIED_ISSUE_EVIDENCE_SOURCE_TRUST_MIN_CONFIDENCE", "0.75"),
        0.75,
    )


def _pre_verifier_evidence_passage_chars() -> int:
    try:
        return max(120, int(os.getenv("CLASSIFIED_ISSUE_EVIDENCE_PASSAGE_MAX_CHARS", "450")))
    except ValueError:
        return 450


def _max_sources_per_trial() -> int:
    try:
        return max(1, int(os.getenv("CLASSIFIED_ISSUE_GROUNDING_MAX_SOURCES", "4")))
    except ValueError:
        return 4


def _passage_extraction_enabled() -> bool:
    return os.getenv("CLASSIFIED_ISSUE_GROUNDING_PASSAGE_EXTRACTION_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _source_repair_enabled() -> bool:
    return os.getenv("CLASSIFIED_ISSUE_GROUNDING_SOURCE_REPAIR_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _max_source_repair_urls() -> int:
    try:
        return max(1, int(os.getenv("CLASSIFIED_ISSUE_GROUNDING_SOURCE_REPAIR_MAX_URLS", "3")))
    except ValueError:
        return 3


def _fetch_timeout_sec() -> float:
    return max(1.0, _safe_float(os.getenv("CLASSIFIED_ISSUE_GROUNDING_FETCH_TIMEOUT_SEC"), 8.0))


def _fetch_max_bytes() -> int:
    try:
        return max(32_768, int(os.getenv("CLASSIFIED_ISSUE_GROUNDING_FETCH_MAX_BYTES", "1200000")))
    except ValueError:
        return 1_200_000


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
Record only the most useful search queries you used or would use to find the evidence.
Use both Korean and English search queries when the lecture claim is Korean, but keep the list short.
Record only the most important bilingual match terms that connect the Korean lecture claim to likely English source wording.
For supports_issue/refutes_issue only, include short source passages. For insufficient_evidence/grounding_unavailable, do not include source passages or evidence summaries.

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

Prefer evidence in this order: official documentation, standards/government sources, academic sources, educational/reference materials, then Wikipedia. Do not use tutorials, blogs, forums, or Q&A sites for a supports/refutes decision.
For brand/product specs, use the relevant region/market when available.
If the lecture claim is a concrete example value whose truth depends on brand, product, region, policy, current price, capacity, weight, version, support status, or statistics, do not rely on generic common knowledge; use web evidence.
If the web evidence only addresses a different unit, region, product, or time period, mark insufficient_evidence.
Do not mark a claim false only because a source does not explicitly state it. Use claim_false only when reliable sources directly contradict the claim.
If the claim can only be supported or rejected by indirect arithmetic, unofficial blogs, screenshots, or mixed sources, use uncertain unless the official source clearly supplies all required values.

Return exactly these eight lines. Do not use JSON or markdown.
SEARCH_QUERIES=query1 | query2 | query3
MATCH_TERMS=Korean term | English term | synonym
CLAIM_VERDICT=claim_true | claim_false | uncertain
STATUS=supports_issue | refutes_issue | insufficient_evidence | grounding_unavailable
REASON=one short Korean sentence explaining the web-grounded judgment
SOURCES=URL1, URL2, URL3
EVIDENCE_PASSAGES=[{{"url":"URL1","quote_or_paragraph":"source passage","key_sentence":"most important sentence","stance":"supports_issue | refutes_issue | unclear","why_relevant":"why this passage matters"}}]
SUMMARY=short Korean summary of the evidence

Consistency requirements:
- If claim_verdict is claim_false, issue_supported must be true and status must be supports_issue.
- If claim_verdict is claim_true, issue_supported must be false and status must be refutes_issue.
- If claim_verdict is uncertain, issue_supported must be null and status must be insufficient_evidence.
- If status is insufficient_evidence or grounding_unavailable, return SEARCH_QUERIES and MATCH_TERMS as empty, SOURCES as empty, EVIDENCE_PASSAGES as [], and SUMMARY as an empty string.
"""


def _build_grounding_json_prompt(issue: dict[str, Any], current_date: str) -> str:
    prompt = _build_grounding_prompt(issue, current_date)
    json_contract = """Return JSON only:
{
  "search_queries": ["query1", "query2", "query3"],
  "match_terms": ["Korean term", "English term", "synonym"],
  "claim_verdict": "claim_true | claim_false | uncertain",
  "status": "supports_issue | refutes_issue | insufficient_evidence | grounding_unavailable",
  "issue_supported": true,
  "reason": "one short Korean sentence explaining the web-grounded judgment",
  "evidence_sources": ["URL1", "URL2", "URL3"],
  "evidence_passages": [
    {
      "url": "URL1",
      "quote_or_paragraph": "the relevant source sentence or paragraph",
      "key_sentence": "the most important sentence",
      "stance": "supports_issue | refutes_issue | unclear",
      "why_relevant": "why this passage supports or refutes the issue"
    }
  ],
  "evidence_summary": "short Korean summary of the evidence"
}
For insufficient_evidence or grounding_unavailable, keep only status fields: use empty search_queries, match_terms, evidence_sources, evidence_passages, and evidence_summary.
"""
    return re.sub(
        r"Return exactly these eight lines\..*?(?=Consistency requirements:)",
        json_contract + "\n",
        prompt,
        flags=re.DOTALL,
    )


def _normalize_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    return status if status in GROUNDING_STATUSES else "insufficient_evidence"


def _normalize_evidence_passages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(_strip_json_fence(value), strict=False)
        except Exception:
            value = []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("source") or item.get("source_url") or "").strip()
        quote = str(
            item.get("quote_or_paragraph")
            or item.get("passage")
            or item.get("paragraph")
            or item.get("quote")
            or item.get("text")
            or ""
        ).strip()
        key_sentence = str(item.get("key_sentence") or item.get("sentence") or "").strip()
        stance = _normalize_status(item.get("stance") or item.get("status") or item.get("supports"))
        rows.append({
            "id": str(item.get("id") or f"E{index}"),
            "source_id": str(item.get("source_id") or "").strip(),
            "url": url,
            "quote_or_paragraph": quote[:1800],
            "key_sentence": key_sentence[:800],
            "stance": stance if stance in {"supports_issue", "refutes_issue", "insufficient_evidence"} else "insufficient_evidence",
            "why_relevant": str(item.get("why_relevant") or item.get("reason") or "").strip()[:800],
        })
    return rows


def _parse_response(text: str, *, require_sources: bool = True) -> dict[str, Any]:
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
    sources = payload.get("evidence_sources", payload.get("sources", []))
    if not isinstance(sources, list):
        sources = [part for part in re.split(r"[,|\n]+", str(sources or "")) if part.strip()]
    sources = [
        str(source).strip()
        for source in sources
        if re.match(r"^https?://", str(source).strip(), flags=re.IGNORECASE)
    ]
    match_terms = payload.get("match_terms", payload.get("terms", []))
    if not isinstance(match_terms, list):
        match_terms = [part for part in re.split(r"[,|\n]+", str(match_terms or "")) if part.strip()]
    match_terms = [str(term).strip() for term in match_terms if str(term).strip()]
    search_queries = payload.get("search_queries", payload.get("queries", []))
    if not isinstance(search_queries, list):
        search_queries = [part for part in re.split(r"[|\n]+", str(search_queries or "")) if part.strip()]
    search_queries = [str(query).strip() for query in search_queries if str(query).strip()]
    verification_target = str(payload.get("verification_target") or "").strip()
    suspected_error = str(payload.get("suspected_error") or "").strip()
    query_language = str(payload.get("query_language") or "").strip()
    query_language_reason = str(payload.get("query_language_reason") or "").strip()
    if require_sources and status in {"supports_issue", "refutes_issue"} and not sources:
        status = "insufficient_evidence"
        claim_verdict = "uncertain"
        issue_supported = None
    evidence_passages = _normalize_evidence_passages(
        payload.get("evidence_passages")
        or payload.get("passages")
        or payload.get("evidence_quotes")
        or payload.get("quotes")
        or []
    )
    return {
        "status": status,
        "claim_verdict": claim_verdict or "uncertain",
        "issue_supported": issue_supported if isinstance(issue_supported, bool) else None,
        "reason": reason,
        "evidence_sources": sources,
        "evidence_passages": evidence_passages,
        "evidence_summary": evidence_summary,
        "search_queries": search_queries,
        "match_terms": match_terms[:40],
        "verification_target": verification_target[:300],
        "suspected_error": suspected_error[:500],
        "query_language": query_language[:80],
        "query_language_reason": query_language_reason[:300],
    }


def _parse_line_response(text: str) -> dict[str, Any]:
    fields = {
        "search_queries": [],
        "match_terms": [],
        "claim_verdict": "",
        "status": "",
        "reason": "",
        "evidence_sources": [],
        "evidence_passages": [],
        "evidence_summary": "",
    }
    key_map = {
        "SEARCH_QUERIES": "search_queries",
        "MATCH_TERMS": "match_terms",
        "CLAIM_VERDICT": "claim_verdict",
        "STATUS": "status",
        "REASON": "reason",
        "SOURCES": "evidence_sources",
        "EVIDENCE_PASSAGES": "evidence_passages",
        "SUMMARY": "evidence_summary",
    }
    matches = list(re.finditer(r"(?m)^(SEARCH_QUERIES|MATCH_TERMS|CLAIM_VERDICT|STATUS|REASON|SOURCES|EVIDENCE_PASSAGES|SUMMARY)\s*=\s*", text))
    if not matches:
        raise ValueError("grounding response is neither JSON nor key-value lines")
    for index, match in enumerate(matches):
        key = key_map[match.group(1)]
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[start:end].strip()
        if key == "search_queries":
            fields[key] = [part.strip() for part in value.split("|") if part.strip()]
        elif key == "match_terms":
            fields[key] = [part.strip() for part in re.split(r"[|,]", value) if part.strip()][:40]
        elif key == "evidence_sources":
            lowered = value.lower()
            if lowered in {"", "none", "n/a", "없음"}:
                fields[key] = []
            else:
                fields[key] = [
                    re.sub(r"\s+", "", part)
                    for part in value.split(",")
                    if re.sub(r"\s+", "", part)
                ]
        elif key == "evidence_passages":
            fields[key] = _normalize_evidence_passages(value)
        else:
            fields[key] = value
    return fields


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_stack: list[str] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg", "canvas"}:
            self._skip_stack.append(tag.lower())
        if tag.lower() in {"p", "br", "li", "tr", "h1", "h2", "h3", "h4", "section", "article"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if self._skip_stack and self._skip_stack[-1] == lowered:
            self._skip_stack.pop()
        if lowered in {"p", "li", "tr", "h1", "h2", "h3", "h4", "section", "article"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        text = re.sub(r"\s+", " ", data or "").strip()
        if text:
            self.parts.append(text)


def _domain_from_url(url: str) -> str:
    return (urlparse(url).netloc or "").lower().removeprefix("www.")


def _domain_matches(domain: str, expected: str) -> bool:
    return bool(domain == expected or domain.endswith(f".{expected}"))


def _hard_source_exclusion(url: str) -> dict[str, str] | None:
    domain = _domain_from_url(url)
    if not domain:
        return {
            "url": str(url or ""),
            "domain": "",
            "category": "invalid_url",
            "reason": "유효한 웹 도메인이 없습니다.",
        }
    for excluded_domain, (category, reason) in HARD_EXCLUDED_SOURCE_DOMAINS.items():
        if _domain_matches(domain, excluded_domain):
            return {
                "url": str(url or ""),
                "domain": domain,
                "category": category,
                "reason": reason,
            }
    return None


def _prefilter_source_candidates(
    values: list[Any],
) -> tuple[list[str], list[dict[str, str]]]:
    accepted: list[str] = []
    excluded: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in values:
        url = str(value or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        exclusion = _hard_source_exclusion(url)
        if exclusion is not None:
            excluded.append(exclusion)
            continue
        if not re.match(r"^https?://", url, flags=re.IGNORECASE):
            excluded.append({
                "url": url,
                "domain": "",
                "category": "invalid_url",
                "reason": "HTTP(S) URL이 아닙니다.",
            })
            continue
        accepted.append(url)
    return accepted, excluded


def _source_trust(url: str) -> dict[str, Any]:
    domain = _domain_from_url(url)
    trust_level = "secondary"
    score = 0.55
    official_doc_domains = (
        "docs.",
        "developer.",
        "developers.",
        "learn.",
        "support.",
        "help.",
        "cloud.",
        "developer.mozilla.org",
        "docs.python.org",
        "kubernetes.io",
        "docs.docker.com",
        "docs.github.com",
        "docs.oracle.com",
        "docs.aws.amazon.com",
        "cloud.google.com",
        "docs.microsoft.com",
        "learn.microsoft.com",
    )
    standards_domains = (
        "w3.org",
        "ietf.org",
        "rfc-editor.org",
        "iso.org",
        "ecma-international.org",
        "ieee.org",
        "nist.gov",
    )
    intergovernmental_domains = (
        "unesco.org",
        "un.org",
        "who.int",
        "europa.eu",
        "oecd.org",
        "worldbank.org",
    )
    government_agency_domains = (
        "arko.or.kr",
    )
    academic_domains = (
        "arxiv.org",
        "ncbi.nlm.nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
        "jstor.org",
        "doi.org",
        "sciencedirect.com",
        "springer.com",
        "cambridge.org",
        "academic.oup.com",
    )
    museum_domains = (
        "uffizi.it",
        "britishmuseum.org",
        "metmuseum.org",
        "moma.org",
        "tate.org.uk",
        "getty.edu",
        "acmi.net.au",
        "vam.ac.uk",
    )
    educational_domains = (
        "britannica.com",
        "khanacademy.org",
        "openstax.org",
        "opentextbooks.org",
        "pressbooks.pub",
        "books.google.com",
        "openlibrary.org",
        "interaction-design.org",
        "treccani.it",
        "encyclopedia.com",
        "pbs.org",
    )
    hard_exclusion = _hard_source_exclusion(url)
    if hard_exclusion and hard_exclusion.get("category") == "tutorial":
        trust_level, score = "tutorial", 0.20
    elif hard_exclusion and hard_exclusion.get("category") in {"forum", "user_wiki"}:
        trust_level, score = "forum", 0.20
    elif hard_exclusion and hard_exclusion.get("category") == "personal_blog":
        trust_level, score = "blog", 0.20
    elif "wikipedia.org" in domain:
        trust_level, score = "encyclopedia", 0.70
    elif any(token in domain for token in standards_domains):
        trust_level, score = "standards", 0.95
    elif any(domain == token or domain.endswith(f".{token}") for token in intergovernmental_domains):
        trust_level, score = "government", 0.95
    elif (
        domain.endswith((".gov", ".go.kr", ".gov.kr"))
        or re.search(r"(?:^|\.)gov\.[a-z]{2,}$", domain)
        or any(domain == token or domain.endswith(f".{token}") for token in government_agency_domains)
    ):
        trust_level, score = "government", 0.95
    elif any(domain == token or domain.endswith(f".{token}") for token in museum_domains):
        trust_level, score = "official_docs", 0.90
    elif any(domain == token or domain.endswith(f".{token}") for token in academic_domains):
        trust_level, score = "academic", 0.85
    elif domain.endswith((".edu", ".ac.kr")):
        trust_level, score = "academic", 0.85
    elif any(token in domain for token in official_doc_domains):
        trust_level, score = "official_docs", 0.90
    elif any(token in domain for token in educational_domains):
        trust_level, score = "educational", 0.75
    elif any(token in domain for token in ("news", "reuters.", "apnews.", "bbc.", "nytimes.", "wsj.")):
        trust_level, score = "news", 0.60
    priority = SOURCE_PRIORITY_ORDER.get(trust_level)
    return {
        "domain": domain,
        "trust_level": trust_level,
        "trust_score": score,
        "source_priority": priority,
        "source_priority_label": SOURCE_PRIORITY_LABELS.get(priority, ""),
        "auto_decision_eligible": bool(priority) and trust_level not in EXCLUDED_SOURCE_LEVELS,
    }


def _decode_bytes(data: bytes, content_type: str) -> str:
    charset_match = re.search(r"charset=([^;\s]+)", content_type or "", flags=re.IGNORECASE)
    encodings = [charset_match.group(1)] if charset_match else []
    encodings.extend(["utf-8", "cp949", "latin-1"])
    for encoding in encodings:
        try:
            return data.decode(encoding, errors="replace")
        except LookupError:
            continue
    return data.decode("utf-8", errors="replace")


def _html_to_text(html: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(html or "")
    text = "\n".join(parser.parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fetch_url_text(url: str) -> dict[str, Any]:
    row = {
        "url": url,
        "original_url": url,
        "resolved_url": "",
        "domain": _domain_from_url(url),
        "trust_level": "unknown",
        "trust_score": 0.0,
        "source_priority": None,
        "source_priority_label": "",
        "auto_decision_eligible": False,
        "fetch_status": "unavailable",
        "content_type": "",
        "text_length": 0,
        "error": "",
        "text": "",
    }
    try:
        req = Request(
            url,
            headers={
                "User-Agent": os.getenv(
                    "CLASSIFIED_ISSUE_GROUNDING_USER_AGENT",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
                "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            },
        )
        with urlopen(req, timeout=_fetch_timeout_sec()) as resp:
            resolved_url = str(resp.geturl() or url).strip()
            row["resolved_url"] = resolved_url
            row["url"] = resolved_url
            resolved_domain = _domain_from_url(resolved_url)
            if (
                resolved_domain == "vertexaisearch.cloud.google.com"
                and urlparse(resolved_url).path.startswith(
                    "/grounding-api-redirect/"
                )
            ):
                row.update({
                    "domain": resolved_domain,
                    "fetch_status": "unresolved_redirect",
                    "error": "Google grounding redirect did not resolve to an original source URL",
                })
                return row
            exclusion = _hard_source_exclusion(resolved_url)
            if exclusion is not None:
                row.update({
                    "domain": str(exclusion.get("domain") or resolved_domain),
                    "fetch_status": "excluded_domain",
                    "error": str(
                        exclusion.get("reason")
                        or "신뢰 대상에서 제외된 도메인입니다."
                    ),
                    "source_exclusion": exclusion,
                })
                return row
            row.update(_source_trust(resolved_url))
            content_type = resp.headers.get("content-type", "")
            data = resp.read(_fetch_max_bytes())
    except HTTPError as exc:
        row.update({"fetch_status": "http_error", "error": f"HTTP {exc.code}: {exc.reason}"})
        return row
    except URLError as exc:
        row.update({"fetch_status": "url_error", "error": str(exc.reason)})
        return row
    except Exception as exc:
        row.update({"fetch_status": "error", "error": str(exc)})
        return row

    row["content_type"] = content_type
    if "pdf" in content_type.lower():
        row.update({"fetch_status": "unsupported_content_type", "error": "PDF extraction is not enabled"})
        return row
    raw_text = _decode_bytes(data, content_type)
    text = _html_to_text(raw_text) if "html" in content_type.lower() or "<html" in raw_text[:2000].lower() else raw_text
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s+", "\n", text).strip()
    row.update({"fetch_status": "ok", "text": text, "text_length": len(text)})
    return row


def _claim_terms(issue: dict[str, Any]) -> list[str]:
    text = " ".join(
        str(issue.get(key) or "")
        for key in ("resolved_claim", "claim_text", "category")
    )
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.+#/_-]{1,}|[가-힣]{2,}|\d+(?:\.\d+)?%?", text)
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "into", "about",
        "입니다", "있습니다", "한다", "되는", "대한", "때문", "그리고", "하지만",
    }
    seen: set[str] = set()
    terms: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in stopwords or lowered in seen:
            continue
        seen.add(lowered)
        terms.append(token)
    return terms[:24]


def _verification_terms(issue: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    query_terms: list[str] = []
    for query in payload.get("search_queries", []) or []:
        query_terms.extend(
            re.findall(r"[A-Za-z0-9][A-Za-z0-9.'’_-]{2,}|[가-힣]{2,}|\d+(?:\.\d+)?%?", str(query))
        )
    query_stopwords = {
        "site", "official", "source", "definition", "academic", "reference",
        "the", "and", "for", "with", "from", "that", "this",
    }
    query_terms = [
        term
        for term in query_terms
        if term.lower() not in query_stopwords and not term.lower().startswith("site:")
    ]
    for term in _claim_terms(issue) + [
        str(value).strip()
        for value in payload.get("match_terms", []) or []
        if str(value).strip()
    ] + query_terms:
        lowered = term.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        terms.append(term)
    return terms[:64]


def _split_passages(text: str, max_chars: int = 900) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{1,}", text or "") if part.strip()]
    passages: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if not current:
            current = paragraph
        elif len(current) + len(paragraph) + 1 <= max_chars:
            current = f"{current}\n{paragraph}"
        else:
            passages.append(current)
            current = paragraph
    if current:
        passages.append(current)
    return passages


def _match_passages(text: str, terms: list[str], *, limit: int = 3) -> list[dict[str, Any]]:
    if not text or not terms:
        return []
    lowered_terms = [term.lower() for term in terms]
    scored: list[tuple[float, str, list[str]]] = []
    for passage in _split_passages(text):
        lowered = passage.lower()
        matched = [term for term, lowered_term in zip(terms, lowered_terms) if lowered_term in lowered]
        numeric_matches = sum(1 for term in matched if re.search(r"\d", term))
        score = len(matched) + numeric_matches * 1.5
        if score > 0:
            scored.append((score, passage[:900], matched[:12]))
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    return [
        {
            "passage_id": f"P{index}",
            "match_score": round(score, 3),
            "matched_terms": matched,
            "text": passage,
        }
        for index, (score, passage, matched) in enumerate(scored[:limit], start=1)
    ]


def _normalize_match_text(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^\w가-힣.%+#/-]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _best_fuzzy_match(needle: str, haystack: str) -> tuple[str, float, str]:
    needle_norm = _normalize_match_text(needle)
    haystack_norm = _normalize_match_text(haystack)
    if not needle_norm or not haystack_norm:
        return "", 0.0, "not_found"
    if needle_norm in haystack_norm:
        return needle[:900], 1.0, "exact"

    best_text = ""
    best_score = 0.0
    for passage in _split_passages(haystack, max_chars=max(900, min(1800, len(needle) + 500))):
        passage_norm = _normalize_match_text(passage)
        score = SequenceMatcher(None, needle_norm[:1200], passage_norm[:1600]).ratio()
        if score > best_score:
            best_score = score
            best_text = passage[:900]
    status = "fuzzy" if best_score >= _safe_float(os.getenv("CLASSIFIED_ISSUE_GROUNDING_PASSAGE_MATCH_THRESHOLD"), 0.62) else "not_found"
    return best_text, round(best_score, 4), status


def _reported_passages_for_url(passages: list[dict[str, Any]], url: str) -> list[dict[str, Any]]:
    target_domain = _domain_from_url(url)
    rows = []
    for passage in passages:
        if not isinstance(passage, dict):
            continue
        passage_url = str(passage.get("url") or "").strip()
        if not passage_url:
            continue
        if passage_url == url or _domain_from_url(passage_url) == target_domain:
            rows.append(passage)
    return rows


def _verify_reported_passages(text: str, passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verified = []
    for passage in passages:
        quote = str(passage.get("quote_or_paragraph") or "").strip()
        key_sentence = str(passage.get("key_sentence") or "").strip()
        if not quote and not key_sentence:
            row = dict(passage)
            row.update({"match_status": "not_found", "match_score": 0.0, "matched_text": ""})
            verified.append(row)
            continue
        quote_match = _best_fuzzy_match(quote, text) if quote else ("", 0.0, "not_found")
        sentence_match = (
            _best_fuzzy_match(key_sentence, text)
            if key_sentence
            else ("", 0.0, "not_found")
        )
        matched_text, score, status = (
            sentence_match
            if _passage_match_usable(sentence_match[2], sentence_match[1])
            else quote_match
        )
        row = dict(passage)
        row.update({
            "match_status": status,
            "match_score": score,
            "matched_text": matched_text,
            "key_sentence_match_status": sentence_match[2],
            "key_sentence_match_score": sentence_match[1],
            "quote_match_status": quote_match[2],
            "quote_match_score": quote_match[1],
            "selection_method": "model_reported_passage",
        })
        verified.append(row)
    return verified


def _passage_match_usable(status: str, score: Any) -> bool:
    if status == "exact":
        return True
    return bool(
        status == "fuzzy"
        and _clamp01(score, 0.0)
        >= _clamp01(
            os.getenv("CLASSIFIED_ISSUE_EVIDENCE_PASSAGE_FUZZY_MIN_SCORE", "0.90"),
            0.90,
        )
    )


def _verify_source_url(url: str, reported_passages: list[dict[str, Any]], terms: list[str]) -> dict[str, Any]:
    row = _fetch_url_text(str(url))
    text = str(row.pop("text", "") or "")
    if text:
        row["_source_text"] = text
    source_reported = _reported_passages_for_url(reported_passages, str(url))
    row["model_reported_passages"] = source_reported
    row["verified_model_passages"] = _verify_reported_passages(text, source_reported) if text else [
        {**passage, "match_status": "unverified_fetch_failed", "match_score": 0.0, "matched_text": ""}
        for passage in source_reported
    ]
    matched_model_passages = [
        passage for passage in row["verified_model_passages"]
        if _passage_match_usable(
            str(passage.get("match_status") or ""),
            passage.get("match_score"),
        )
    ]
    fallback_passages = [] if matched_model_passages else _match_passages(text, terms)
    for passage in fallback_passages:
        passage["selection_method"] = "keyword_fallback"
        passage["match_status"] = "keyword_fallback"
    row["matched_passages"] = matched_model_passages or fallback_passages
    row["direct_match"] = bool(row["matched_passages"])
    row["priority_eligible"] = bool(row["direct_match"] and row.get("auto_decision_eligible"))
    return row


def _verify_payload_sources(
    issue: dict[str, Any],
    payload: dict[str, Any],
    *,
    max_sources: int | None = None,
) -> dict[str, Any]:
    sources = payload.get("evidence_sources") if isinstance(payload.get("evidence_sources"), list) else []
    reported_passages = payload.get("evidence_passages") if isinstance(payload.get("evidence_passages"), list) else []
    terms = _verification_terms(issue, payload)
    verified_sources = []
    source_limit = max_sources if max_sources is not None else _max_sources_per_trial()
    successful_fetch_count = 0
    fetch_attempt_limit = max(
        max(1, int(source_limit)),
        _pre_verifier_evidence_max_fetch_attempts(),
    )
    for url in sources[:fetch_attempt_limit]:
        row = _verify_source_url(str(url), reported_passages, terms)
        verified_sources.append(row)
        if row.get("fetch_status") == "ok" and row.get("_source_text"):
            successful_fetch_count += 1
        if successful_fetch_count >= max(1, int(source_limit)):
            break

    matched_sources = [row for row in verified_sources if row.get("direct_match")]
    if matched_sources:
        verification_status = "verified"
    elif verified_sources:
        verification_status = "no_direct_passage"
    else:
        verification_status = "no_sources"

    payload["verified_sources"] = verified_sources
    payload["source_fetch"] = {
        "target_success_count": max(1, int(source_limit)),
        "attempt_limit": fetch_attempt_limit,
        "attempt_count": len(verified_sources),
        "successful_fetch_count": successful_fetch_count,
        "refilled_count": max(
            0,
            len(verified_sources) - min(max(1, int(source_limit)), len(sources)),
        ),
    }
    payload["source_verification_status"] = verification_status
    payload["direct_evidence_count"] = sum(len(row.get("matched_passages") or []) for row in matched_sources)
    if payload.get("status") in {"supports_issue", "refutes_issue"} and verification_status != "verified":
        payload["pre_source_verification_status"] = payload.get("status")
        payload["status"] = "insufficient_evidence"
        payload["claim_verdict"] = "uncertain"
        payload["issue_supported"] = None
        payload["reason"] = (
            "모델이 웹 근거를 제시했지만 URL 본문에서 claim과 직접 연결되는 근거 문단을 확인하지 못했습니다."
        )
    return payload


def _refresh_source_verification_status(payload: dict[str, Any]) -> None:
    verified_sources = payload.get("verified_sources") if isinstance(payload.get("verified_sources"), list) else []
    matched_sources = [row for row in verified_sources if isinstance(row, dict) and row.get("direct_match")]
    if matched_sources:
        verification_status = "verified"
    elif verified_sources:
        verification_status = "no_direct_passage"
    else:
        verification_status = "no_sources"
    payload["source_verification_status"] = verification_status
    payload["direct_evidence_count"] = sum(len(row.get("matched_passages") or []) for row in matched_sources)


def _directional_status_before_repair(payload: dict[str, Any]) -> str:
    for key in ("status", "pre_source_verification_status", "pre_evidence_recheck_status"):
        status = str(payload.get(key) or "")
        if status in {"supports_issue", "refutes_issue"}:
            return status
    return ""


def _best_eligible_priority(payload: dict[str, Any]) -> int | None:
    priorities = [
        source.get("source_priority")
        for source in payload.get("verified_sources", []) or []
        if isinstance(source, dict)
        and source.get("direct_match")
        and source.get("auto_decision_eligible")
        and isinstance(source.get("source_priority"), int)
    ]
    return min(priorities) if priorities else None


def _restore_directional_status(payload: dict[str, Any], status: str) -> None:
    if status not in {"supports_issue", "refutes_issue"}:
        return
    payload["status"] = status
    if status == "supports_issue":
        payload["claim_verdict"] = "claim_false"
        payload["issue_supported"] = True
    else:
        payload["claim_verdict"] = "claim_true"
        payload["issue_supported"] = False


def _needs_source_repair(payload: dict[str, Any]) -> bool:
    best_priority = _best_eligible_priority(payload)
    if best_priority is not None and best_priority <= SOURCE_PRIORITY_ORDER["government"]:
        return False
    if _directional_status_before_repair(payload):
        return True
    for source in payload.get("verified_sources", []) or []:
        if not isinstance(source, dict):
            continue
        priority = source.get("source_priority")
        if isinstance(priority, int) and priority <= SOURCE_PRIORITY_ORDER["government"]:
            if source.get("fetch_status") != "ok" or not source.get("direct_match"):
                return True
    return payload.get("source_verification_status") in {"no_sources", "no_direct_passage"}


def _build_source_repair_prompt(issue: dict[str, Any], payload: dict[str, Any], current_date: str) -> str:
    existing_sources = []
    for source in payload.get("verified_sources", []) or []:
        if not isinstance(source, dict):
            continue
        existing_sources.append({
            "url": source.get("url", ""),
            "domain": source.get("domain", ""),
            "trust_level": source.get("trust_level", ""),
            "source_priority": source.get("source_priority"),
            "fetch_status": source.get("fetch_status", ""),
            "error": source.get("error", ""),
            "direct_match": source.get("direct_match", False),
            "auto_decision_eligible": source.get("auto_decision_eligible", False),
            "matched_passage_count": len(source.get("matched_passages") or []),
        })
    return f"""You are repairing web evidence sources for a lecture issue verifier.
Current date: {current_date}

The previous source set either failed to fetch, was too broad, or only produced low-priority evidence.
Find replacement URLs that are more specific and more authoritative.

Priority order:
1. Official product/vendor/API/reference documentation
2. Standards or government sources
3. Academic sources
4. Educational/reference materials
5. Wikipedia only as fallback

Do not return tutorials, blogs, forums, StackOverflow, or Q&A sites.
Prefer concrete reference/API pages over broad overview pages.
If the claim is about application file I/O, direct disk access, operating-system mediation, or system calls, prefer individual OS API/reference pages such as Microsoft Learn CreateFile, WriteFile, ReadFile, Windows file handles, or POSIX open/write references.
If a Microsoft Learn overview URL failed or was too broad, drill down to the individual function/reference pages.
Do not return a broad function index/listing page as the only official source; include the specific reference pages for the named API/function when possible.

Resolved claim: {issue.get("resolved_claim", "")}
Original claim_text: {issue.get("claim_text", "")}
Current model status before repair: {_directional_status_before_repair(payload) or payload.get("status", "")}
Existing search queries: {json.dumps(payload.get("search_queries", []), ensure_ascii=False)}
Existing sources and failures:
{json.dumps(existing_sources, ensure_ascii=False, indent=2)}

Return JSON only:
{{
  "search_queries": ["specific repair query 1", "specific repair query 2"],
  "match_terms": ["Korean term", "English term", "API/function name"],
  "evidence_sources": ["https://official-or-reference-url-1", "https://official-or-reference-url-2"],
  "evidence_passages": [
    {{
      "url": "https://source-url",
      "quote_or_paragraph": "known relevant sentence or paragraph if available",
      "key_sentence": "most important sentence if available",
      "stance": "supports_issue | refutes_issue | unclear",
      "why_relevant": "brief Korean explanation"
    }}
  ],
  "repair_reason": "brief Korean explanation of why these URLs are better"
}}
"""


def _call_source_repair_fallback(
    *,
    model_spec: str,
    issue: dict[str, Any],
    payload: dict[str, Any],
    current_date: str,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    if not _source_repair_enabled() or not _needs_source_repair(payload):
        return payload, _empty_token_usage()
    directional_status = _directional_status_before_repair(payload)
    prompt = _build_source_repair_prompt(issue, payload, current_date)
    text, usage, resolved = _call_llm(model_spec=model_spec, prompt=prompt, max_tokens=max(512, min(max_tokens, 1200)), stage="grounding")
    repair: dict[str, Any] = {
        "provider": resolved.get("provider", ""),
        "model": model_spec,
        "resolved_model": resolved.get("resolved_model", model_spec),
        "trigger_best_priority": _best_eligible_priority(payload),
        "added_source_count": 0,
        "direct_source_count": 0,
        "eligible_direct_source_count": 0,
    }
    try:
        repaired = _parse_response(text or "", require_sources=False)
    except Exception as exc:
        repair["parse_error"] = str(exc)
        payload["source_repair"] = repair
        return payload, usage

    existing_urls = {
        str(source.get("url") or "")
        for source in payload.get("verified_sources", []) or []
        if isinstance(source, dict)
    }
    payload_sources = payload.get("evidence_sources") if isinstance(payload.get("evidence_sources"), list) else []
    payload_queries = payload.get("search_queries") if isinstance(payload.get("search_queries"), list) else []
    payload_terms = payload.get("match_terms") if isinstance(payload.get("match_terms"), list) else []
    for query in repaired.get("search_queries", []) or []:
        if query not in payload_queries:
            payload_queries.append(query)
    for term in repaired.get("match_terms", []) or []:
        if term not in payload_terms:
            payload_terms.append(term)
    payload["search_queries"] = payload_queries
    payload["match_terms"] = payload_terms[:64]

    repair_sources = [
        str(url).strip()
        for url in repaired.get("evidence_sources", []) or []
        if re.match(r"^https?://", str(url).strip(), flags=re.IGNORECASE)
    ]
    repair_passages = repaired.get("evidence_passages") if isinstance(repaired.get("evidence_passages"), list) else []
    terms = _verification_terms(issue, payload)
    verified_sources = payload.get("verified_sources") if isinstance(payload.get("verified_sources"), list) else []
    for url in repair_sources:
        if url in existing_urls:
            continue
        if repair["added_source_count"] >= _max_source_repair_urls():
            break
        row = _verify_source_url(url, repair_passages, terms)
        row["source_added_by_repair"] = True
        verified_sources.append(row)
        existing_urls.add(url)
        if url not in payload_sources:
            payload_sources.append(url)
        repair["added_source_count"] += 1
    payload["verified_sources"] = verified_sources
    payload["evidence_sources"] = payload_sources
    _refresh_source_verification_status(payload)
    repair["direct_source_count"] = sum(
        1 for source in verified_sources if isinstance(source, dict) and source.get("direct_match")
    )
    repair["eligible_direct_source_count"] = sum(
        1 for source in verified_sources if isinstance(source, dict) and source.get("priority_eligible")
    )
    repair["best_priority_after_repair"] = _best_eligible_priority(payload)
    repair["repair_reason"] = repaired.get("repair_reason", "")
    payload["source_repair"] = repair
    if payload.get("source_verification_status") == "verified" and directional_status:
        _restore_directional_status(payload, directional_status)
    return payload, usage


def _source_text_sample(text: str, terms: list[str], max_chars: int = 5000) -> str:
    if not text:
        return ""
    passages = _split_passages(text, max_chars=900)
    lowered_terms = [term.lower() for term in terms if term]
    scored: list[tuple[int, int, str]] = []
    for index, passage in enumerate(passages):
        lowered = passage.lower()
        score = sum(1 for term in lowered_terms if term in lowered)
        scored.append((score, index, passage))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected: list[str] = []
    total = 0
    for _, _, passage in scored[:8]:
        if total + len(passage) + 2 > max_chars:
            break
        selected.append(passage)
        total += len(passage) + 2
    if selected:
        return "\n\n---\n\n".join(selected)
    return text[:max_chars]


def _build_document_relevance_prompt(
    issue: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[str, list[dict[str, str]]]:
    terms = _verification_terms(issue, payload)
    documents: list[dict[str, str]] = []
    for source in payload.get("verified_sources", []) or []:
        if not isinstance(source, dict):
            continue
        if source.get("fetch_status") != "ok":
            source["document_relevance"] = "unavailable"
            source["document_relevance_confidence"] = 0.0
            source["document_relevance_reason"] = "문서 본문을 가져오지 못했습니다."
            source["document_relevance_eligible"] = False
            continue
        text = str(source.get("_source_text") or "")
        sample = _source_text_sample(text, terms, max_chars=1800)
        if not sample:
            source["document_relevance"] = "irrelevant"
            source["document_relevance_confidence"] = 1.0
            source["document_relevance_reason"] = "판정할 수 있는 문서 본문이 없습니다."
            source["document_relevance_eligible"] = False
            continue
        document_id = f"D{len(documents) + 1}"
        source["_document_relevance_id"] = document_id
        documents.append({
            "document_id": document_id,
            "text_excerpt": sample,
        })
    prompt = f"""당신은 검색 결과 문서가 하나의 강의 claim을 사실 검증하기에 적합한지 판정합니다.
출처의 권위나 신뢰도, 문장의 정확한 인용 여부, claim의 참·거짓은 판단하지 마세요.
오직 제공된 문서 내용이 같은 대상과 검증에 필요한 관계·범위·시기·수치를 직접 다루는지만 판단하세요.

판정:
- direct: 같은 대상을 다루며 claim 전체를 판단하거나 명시된 핵심 주장 하나를 결정적으로 확인·반박할 수 있는 내용을 포함합니다.
- partial: 같은 대상이나 주제를 다루지만 필요한 관계·범위·시기·수치가 빠져 claim을 판단할 수 없습니다.
- irrelevant: 다른 대상이거나 단순한 키워드 중복으로 claim 검증에 사용할 수 없습니다.

복합 claim의 판정:
- claim에 명시된 필수 사실 구성요소 하나를 문서가 명시적으로 반박한다면, 나머지 구성요소를 모두 다루지 않아도
  claim의 거짓을 결정할 수 있으므로 direct입니다.
- 문서가 필수 구성요소 일부를 지지할 뿐이고 나머지 구성요소의 참·거짓을 결정할 수 없다면 partial입니다.
- 이 규칙은 최종 이슈 판정을 대신하는 것이 아니라, 다음 발췌 단계에 문서를 보낼 수 있는지를 판단하기 위한 것입니다.

resolved_claim: {issue.get("resolved_claim", "")}
claim_text: {issue.get("claim_text", "")}

문서 후보:
{json.dumps(documents, ensure_ascii=False)}

JSON만 반환하세요:
{{
  "assessments": [
    {{
      "document_id": "D1",
      "relevance": "direct | partial | irrelevant",
      "confidence": 0.0,
      "reason": "짧은 한국어 이유"
    }}
  ]
}}
"""
    return prompt, documents


def _call_document_relevance_assessment(
    *,
    model_spec: str,
    issue: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt, documents = _build_document_relevance_prompt(issue, payload)
    if not documents:
        payload["document_relevance"] = {
            "status": "not_run",
            "reason": "본문을 가져온 문서 후보가 없습니다.",
            "candidate_count": 0,
            "direct_count": 0,
        }
        return payload, _empty_token_usage()

    text, usage, resolved = _call_llm(
        model_spec=model_spec,
        prompt=prompt,
        max_tokens=_pre_verifier_document_relevance_max_tokens(),
        stage="grounding",
    )
    parse_error = ""
    assessments: dict[str, dict[str, Any]] = {}
    try:
        data = json.loads(_strip_json_fence(text or ""), strict=False)
        rows = data.get("assessments") if isinstance(data, dict) else []
        if not isinstance(rows, list):
            raise ValueError("document relevance assessments is not a list")
        valid_ids = {document["document_id"] for document in documents}
        for row in rows:
            if not isinstance(row, dict):
                continue
            document_id = str(row.get("document_id") or "").strip()
            if document_id not in valid_ids or document_id in assessments:
                continue
            relevance = str(row.get("relevance") or "").strip().lower()
            if relevance not in {"direct", "partial", "irrelevant"}:
                relevance = "irrelevant"
            assessments[document_id] = {
                "relevance": relevance,
                "confidence": round(_clamp01(row.get("confidence"), 0.0), 4),
                "reason": _trim_text(row.get("reason") or "", 220),
            }
    except Exception as exc:
        parse_error = str(exc)

    min_confidence = _pre_verifier_document_relevance_min_confidence()
    direct_count = 0
    partial_count = 0
    eligible_count = 0
    for source in payload.get("verified_sources", []) or []:
        if not isinstance(source, dict):
            continue
        document_id = str(source.pop("_document_relevance_id", "") or "")
        if not document_id:
            continue
        assessment = assessments.get(document_id)
        if assessment is None:
            source["document_relevance"] = "unknown"
            source["document_relevance_confidence"] = 0.0
            source["document_relevance_reason"] = (
                "문서 적합성 판정이 반환되지 않았습니다."
                if not parse_error
                else "문서 적합성 응답을 파싱하지 못했습니다."
            )
            source["document_relevance_eligible"] = False
            continue
        source["document_relevance"] = assessment["relevance"]
        source["document_relevance_confidence"] = assessment["confidence"]
        source["document_relevance_reason"] = assessment["reason"]
        confident = assessment["confidence"] >= min_confidence
        eligible = bool(
            assessment["relevance"] in {"direct", "partial"}
            and confident
        )
        source["document_relevance_eligible"] = eligible
        direct_count += int(assessment["relevance"] == "direct" and confident)
        partial_count += int(assessment["relevance"] == "partial" and confident)
        eligible_count += int(eligible)

    payload["document_relevance"] = {
        "status": "parse_failed" if parse_error else "ok",
        "model": model_spec,
        "resolved_model": resolved.get("resolved_model", model_spec),
        "parse_error": parse_error,
        "candidate_count": len(documents),
        "direct_count": direct_count,
        "partial_count": partial_count,
        "eligible_count": eligible_count,
        "min_confidence": min_confidence,
    }
    return payload, usage


def _build_passage_extraction_prompt(
    issue: dict[str, Any],
    payload: dict[str, Any],
    *,
    reselect_all: bool = False,
) -> str:
    terms = _verification_terms(issue, payload)
    sources = []
    for source in payload.get("verified_sources", []) or []:
        if not isinstance(source, dict):
            continue
        if not source.get("document_relevance_eligible", True):
            continue
        if not reselect_all and not source.get("auto_decision_eligible"):
            continue
        if not reselect_all and _has_verified_model_passage(source):
            continue
        if source.get("fetch_status") != "ok":
            continue
        text = str(source.get("_source_text") or "")
        if not text:
            continue
        source_id = f"P{len(sources) + 1}"
        source["_passage_extraction_id"] = source_id
        sources.append({
            "source_id": source_id,
            "url": source.get("url", ""),
            "domain": source.get("domain", ""),
            "relevance": source.get("document_relevance", "direct"),
            "relevance_reason": source.get("document_relevance_reason", ""),
            "text_excerpt": _source_text_sample(text, terms, max_chars=1800),
        })
    return f"""당신은 적합성 판정을 통과한 문서 본문에서 강의 claim 검증에 사용할 발췌문을 선택합니다.
제공된 본문 조각만 사용하고 웹 검색, 기억에 따른 보충, 문장 재작성이나 번역문 생성을 하지 마세요.

발췌 규칙:
1. 문서마다 가장 유용한 원문 문장 하나만 선택하세요. 필요한 경우 그 문장을 포함한 짧은 문단을 함께 반환할 수 있습니다.
2. 주체·대상·관계·범위·시기·수치·단위 중 claim 판정에 필요한 요소가 실제 문장에 명시돼야 합니다.
3. relevance=direct이면 claim 전체 또는 필수 구성요소 하나를 결정할 수 있는 문장을 선택하세요.
4. relevance=partial이면 문서가 실제로 다루는 구성요소만 보여주는 문장을 선택하고, 빠진 범위까지 확장하지 마세요.
5. 문서 조각에 적절한 문장이 없다면 해당 source_id를 결과에서 생략하세요.
6. key_sentence와 quote_or_paragraph는 제공된 본문에서 글자 그대로 복사하세요.

resolved_claim: {issue.get("resolved_claim", "")}
claim_text: {issue.get("claim_text", "")}

문서:
{json.dumps(sources, ensure_ascii=False, indent=2)}

JSON만 반환하세요:
{{
  "evidence_passages": [
    {{
      "source_id": "P1",
      "quote_or_paragraph": "본문에서 그대로 복사한 짧은 문단 또는 문장",
      "key_sentence": "본문에서 그대로 복사한 핵심 문장 하나"
    }}
  ]
}}
"""


def _has_verified_model_passage(source: dict[str, Any]) -> bool:
    return any(
        isinstance(passage, dict)
        and _passage_match_usable(
            str(passage.get("match_status") or ""),
            passage.get("match_score"),
        )
        for passage in source.get("verified_model_passages", []) or []
    )


def _call_passage_extraction_fallback(
    *,
    model_spec: str,
    issue: dict[str, Any],
    payload: dict[str, Any],
    max_tokens: int,
    reselect_all: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    if not _passage_extraction_enabled():
        return payload, _empty_token_usage()
    candidates = [
        source for source in payload.get("verified_sources", []) or []
        if isinstance(source, dict)
        and source.get("document_relevance_eligible", True)
        and (reselect_all or source.get("auto_decision_eligible"))
        and (reselect_all or not _has_verified_model_passage(source))
        and source.get("fetch_status") == "ok"
        and source.get("_source_text")
    ]
    if not candidates:
        return payload, _empty_token_usage()

    prompt = _build_passage_extraction_prompt(
        issue,
        payload,
        reselect_all=reselect_all,
    )
    text, usage, resolved = _call_llm(model_spec=model_spec, prompt=prompt, max_tokens=max(512, min(max_tokens, 1200)), stage="grounding")
    extraction: dict[str, Any] = {
        "provider": resolved.get("provider", ""),
        "model": model_spec,
        "resolved_model": resolved.get("resolved_model", model_spec),
        "candidate_source_count": len(candidates),
        "added_passage_count": 0,
        "reselect_all": reselect_all,
    }
    try:
        data = json.loads(_strip_json_fence(text or ""), strict=False)
        extracted_passages = _normalize_evidence_passages(data.get("evidence_passages", []))
    except Exception as exc:
        for source in candidates:
            source.pop("_passage_extraction_id", None)
        extraction["parse_error"] = str(exc)
        payload["passage_extraction"] = extraction
        return payload, usage

    added = 0
    selected_source_count = 0
    no_passage_source_count = 0
    for source in candidates:
        source_text = str(source.get("_source_text") or "")
        source_id = str(source.pop("_passage_extraction_id", "") or "")
        source_passages = [
            passage
            for passage in extracted_passages
            if str(passage.get("source_id") or "") == source_id
        ]
        if not source_passages:
            source_passages = _reported_passages_for_url(
                extracted_passages,
                str(source.get("url") or ""),
            )
        verified = _verify_reported_passages(source_text, source_passages)
        matched = [
            {**passage, "selection_method": "llm_passage_extraction"}
            for passage in verified
            if _passage_match_usable(
                str(passage.get("match_status") or ""),
                passage.get("match_score"),
            )
        ]
        if not matched:
            if reselect_all:
                source["stage3_verified_passages"] = verified
                source["matched_passages"] = []
                source["direct_match"] = False
                source["priority_eligible"] = False
                source["passage_extraction_status"] = "no_usable_passage"
            no_passage_source_count += 1
            continue
        source["stage3_verified_passages"] = verified
        source["matched_passages"] = (
            matched
            if reselect_all
            else matched + (source.get("matched_passages") or [])
        )
        source["direct_match"] = True
        source["priority_eligible"] = bool(source.get("auto_decision_eligible"))
        source["passage_extraction_status"] = "selected"
        added += len(matched)
        selected_source_count += 1
    extraction["added_passage_count"] = added
    extraction["extracted_passage_count"] = len(extracted_passages)
    extraction["selected_source_count"] = selected_source_count
    extraction["no_passage_source_count"] = no_passage_source_count
    payload["passage_extraction"] = extraction
    _refresh_source_verification_status(payload)
    return payload, usage


def _build_source_trust_prompt(
    issue: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    for source in payload.get("verified_sources", []) or []:
        if not isinstance(source, dict):
            continue
        if (
            source.get("fetch_status") != "ok"
            or not source.get("direct_match")
            or not source.get("_source_text")
        ):
            source["source_strength"] = "excluded"
            source["source_trust_eligible"] = False
            source["source_trust_reason"] = "확인된 발췌문이 없어 출처 신뢰도를 적용할 수 없습니다."
            source["priority_eligible"] = False
            continue
        passages = source.get("matched_passages") if isinstance(source.get("matched_passages"), list) else []
        key_sentence = next(
            (
                _evidence_passage_text(passage)
                for passage in passages
                if isinstance(passage, dict) and _evidence_passage_text(passage)
            ),
            "",
        )
        source_text = str(source.get("_source_text") or "")
        source_id = f"T{len(sources) + 1}"
        source["_source_trust_id"] = source_id
        sources.append({
            "source_id": source_id,
            "url": str(source.get("url") or ""),
            "domain": str(source.get("domain") or ""),
            "content_type": str(source.get("content_type") or ""),
            "document_relevance": str(source.get("document_relevance") or ""),
            "selected_passage": key_sentence,
            "page_identity_excerpt": _trim_text(source_text, 800),
        })
    prompt = f"""당신은 강의 claim의 웹 근거로 사용될 문서의 출처 신뢰도를 판정합니다.
claim의 참·거짓, 이슈 점수, 발췌문의 관계 방향은 판단하지 마세요.
URL 도메인의 겉모양만 보지 말고 발행 주체, 실제 문서 유형, 해당 claim 분야에 대한 권위와 편집·검토 수준을 판단하세요.
정부나 공공기관 도메인에 있다는 이유만으로 그 안의 학술논문·검색 색인·일반 안내문을 정부 원자료로 분류하지 마세요.
공식 사이트라도 그 기관의 소관 밖 사실을 설명하는 홍보·관광·일반 안내문은 공식 원자료가 아니라 2차 자료입니다.

source_class:
- primary_authority: 해당 사실을 직접 관리·발표하는 기관의 공식 기록, 제품 공식 문서, 법령·공식 통계·공식 문화재 기록
- standard: 공인 표준 또는 규격 문서
- scholarly: 관련 분야의 학술 논문이나 학술 출판물
- expert_reference: 전문가가 편집한 백과사전, 박물관·대학의 전문 해설, 교과서급 참고자료
- encyclopedia: 위키백과. 원문에서 claim과 직접 관련된 문장이 확인되면 단독 근거로 허용
- official_secondary: 공공·공식 기관이 제공하지만 해당 사실의 원자료는 아닌 안내·홍보·개요
- general_secondary: 언론·상업 출판·일반 참고 사이트의 2차 설명
- user_generated: 개인 블로그, 포럼, 나무위키 등 사용자 생성물
- promotional: 판매·홍보 목적이 강하고 근거 검토가 어려운 자료
- unknown: 발행 주체나 문서 성격을 확인할 수 없음

authority_for_claim:
- high: 해당 claim의 사실을 판정할 직접 권한이나 전문성이 큼
- medium: 유용한 2차 설명이지만 단독 확정 근거로는 제한됨
- low: 해당 claim을 판정할 전문성·책임·검토 수준이 부족함

resolved_claim: {issue.get("resolved_claim", "")}

문서:
{json.dumps(sources, ensure_ascii=False, indent=2)}

JSON만 반환하세요:
{{
  "assessments": [
    {{
      "source_id": "T1",
      "source_class": "primary_authority | standard | scholarly | expert_reference | encyclopedia | official_secondary | general_secondary | user_generated | promotional | unknown",
      "authority_for_claim": "high | medium | low",
      "confidence": 0.0,
      "reason": "짧은 한국어 이유"
    }}
  ]
}}
"""
    return prompt, sources


def _source_strength_from_assessment(
    source_class: str,
    authority_for_claim: str,
    confidence: float,
) -> str:
    if confidence < _pre_verifier_source_trust_min_confidence():
        return "excluded"
    if authority_for_claim == "low":
        return "excluded"
    if (
        source_class
        in {
            "primary_authority",
            "standard",
            "scholarly",
            "expert_reference",
            "encyclopedia",
        }
        and authority_for_claim in {"high", "medium"}
    ):
        return "strong"
    if (
        source_class in {
            "official_secondary",
            "general_secondary",
        }
        and authority_for_claim in {"high", "medium"}
    ):
        return "supporting"
    return "excluded"


def _apply_source_domain_policy(
    source: dict[str, Any],
    source_class: str,
    authority_for_claim: str,
    confidence: float,
    reason: str,
) -> tuple[str, str, float, str]:
    domain = str(
        source.get("domain")
        or _domain_from_url(source.get("url") or "")
    ).lower()
    if domain == "wikipedia.org" or domain.endswith(".wikipedia.org"):
        return (
            "encyclopedia",
            "medium",
            1.0,
            "위키백과는 정책상 원문 직접 근거가 확인되면 단독 근거로 허용합니다.",
        )
    return source_class, authority_for_claim, confidence, reason


def _call_source_trust_assessment(
    *,
    model_spec: str,
    issue: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt, sources = _build_source_trust_prompt(issue, payload)
    if not sources:
        payload["source_trust_assessment"] = {
            "status": "not_run",
            "reason": "정확한 발췌문이 확인된 문서가 없습니다.",
            "candidate_count": 0,
            "strong_count": 0,
            "supporting_count": 0,
            "excluded_count": 0,
        }
        return payload, _empty_token_usage()

    text, usage, resolved = _call_llm(
        model_spec=model_spec,
        prompt=prompt,
        max_tokens=_pre_verifier_source_trust_max_tokens(),
        stage="grounding",
    )
    valid_classes = {
        "primary_authority",
        "standard",
            "scholarly",
            "expert_reference",
            "encyclopedia",
            "official_secondary",
        "general_secondary",
        "user_generated",
        "promotional",
        "unknown",
    }
    assessments: dict[str, dict[str, Any]] = {}
    parse_error = ""
    try:
        data = json.loads(_strip_json_fence(text or ""), strict=False)
        rows = data.get("assessments") if isinstance(data, dict) else []
        if not isinstance(rows, list):
            raise ValueError("source trust assessments is not a list")
        valid_ids = {str(source.get("source_id") or "") for source in sources}
        for row in rows:
            if not isinstance(row, dict):
                continue
            source_id = str(row.get("source_id") or "").strip()
            if source_id not in valid_ids or source_id in assessments:
                continue
            source_class = str(row.get("source_class") or "").strip().lower()
            if source_class not in valid_classes:
                source_class = "unknown"
            authority = str(row.get("authority_for_claim") or "").strip().lower()
            if authority not in {"high", "medium", "low"}:
                authority = "low"
            confidence = round(_clamp01(row.get("confidence"), 0.0), 4)
            assessments[source_id] = {
                "source_class": source_class,
                "authority_for_claim": authority,
                "confidence": confidence,
                "reason": _trim_text(row.get("reason") or "", 220),
            }
    except Exception as exc:
        parse_error = str(exc)

    counts = Counter()
    for source in payload.get("verified_sources", []) or []:
        if not isinstance(source, dict):
            continue
        source_id = str(source.pop("_source_trust_id", "") or "")
        if not source_id:
            continue
        assessment = assessments.get(source_id)
        if assessment is None:
            source_class = "unknown"
            authority = "low"
            confidence = 0.0
            reason = (
                "출처 신뢰도 판정이 반환되지 않았습니다."
                if not parse_error
                else "출처 신뢰도 응답을 파싱하지 못했습니다."
            )
        else:
            source_class = assessment["source_class"]
            authority = assessment["authority_for_claim"]
            confidence = assessment["confidence"]
            reason = assessment["reason"]
        source_class, authority, confidence, reason = (
            _apply_source_domain_policy(
                source,
                source_class,
                authority,
                confidence,
                reason,
            )
        )
        strength = _source_strength_from_assessment(
            source_class,
            authority,
            confidence,
        )
        source["assessed_source_class"] = source_class
        source["authority_for_claim"] = authority
        source["source_trust_confidence"] = confidence
        source["source_trust_reason"] = reason
        source["source_strength"] = strength
        source["source_trust_eligible"] = strength in {"strong", "supporting"}
        source["priority_eligible"] = bool(
            source.get("direct_match") and source["source_trust_eligible"]
        )
        counts[strength] += 1

    payload["source_trust_assessment"] = {
        "status": "parse_failed" if parse_error else "ok",
        "model": model_spec,
        "resolved_model": resolved.get("resolved_model", model_spec),
        "parse_error": parse_error,
        "candidate_count": len(sources),
        "strong_count": counts.get("strong", 0),
        "supporting_count": counts.get("supporting", 0),
        "excluded_count": counts.get("excluded", 0),
        "min_confidence": _pre_verifier_source_trust_min_confidence(),
    }
    return payload, usage


def _strip_internal_source_text(payload: dict[str, Any]) -> None:
    for source in payload.get("verified_sources", []) or []:
        if isinstance(source, dict):
            source.pop("_source_text", None)


def _build_evidence_recheck_prompt(issue: dict[str, Any], payload: dict[str, Any], current_date: str) -> str:
    excerpts = []
    for source in payload.get("verified_sources", []) or []:
        if not isinstance(source, dict) or not source.get("direct_match"):
            continue
        if not source.get("auto_decision_eligible"):
            continue
        passages = source.get("matched_passages") if isinstance(source.get("matched_passages"), list) else []
        excerpts.append({
            "url": source.get("url", ""),
            "domain": source.get("domain", ""),
            "trust_level": source.get("trust_level", ""),
            "trust_score": source.get("trust_score", 0.0),
            "source_priority": source.get("source_priority"),
            "source_priority_label": source.get("source_priority_label", ""),
            "auto_decision_eligible": source.get("auto_decision_eligible", False),
            "passages": passages[:3],
        })
    return f"""You are rechecking a lecture issue using only fetched source excerpts.
Current date: {current_date}

Do not use web search, memory, or assumptions. Judge only from the excerpts.
If the excerpts do not directly prove the claim true or false in the relevant region/time/product context, return uncertain.
The excerpts have already been filtered to automatic-decision source tiers: official documentation, standards/government, academic, educational/reference, or Wikipedia.
If the lecture uses a named application/product merely as an example of a general software class, and the claim is about the general operating-system mechanism for that class, do not require product-specific vendor documentation. General OS/API/reference evidence may be sufficient.
Pay close attention to the actor/subject of the claim. If the claim says a resource/object itself performs an action, but the excerpts attribute that action to an operating system, manager, API, or other component, treat that as a contradiction rather than confirmation.

Issue category: {issue.get("category", "")}
Resolved claim: {issue.get("resolved_claim", "")}
Original claim_text: {issue.get("claim_text", "")}

Fetched source excerpts:
{json.dumps(excerpts, ensure_ascii=False, indent=2)}

Return JSON only:
{{
  "claim_verdict": "claim_true | claim_false | uncertain",
  "status": "supports_issue | refutes_issue | insufficient_evidence",
  "issue_supported": true,
  "reason": "one or two Korean sentences",
  "evidence_summary": "short Korean evidence summary"
}}
For insufficient_evidence, keep reason to one short Korean sentence and evidence_summary as an empty string.
"""


def _evidence_recheck_enabled() -> bool:
    return os.getenv("CLASSIFIED_ISSUE_GROUNDING_EVIDENCE_RECHECK_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _call_evidence_recheck(
    *,
    model_spec: str,
    issue: dict[str, Any],
    payload: dict[str, Any],
    current_date: str,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    if not _evidence_recheck_enabled() or payload.get("source_verification_status") != "verified":
        return {}, _empty_token_usage()
    prompt = _build_evidence_recheck_prompt(issue, payload, current_date)
    text, usage, resolved = _call_llm(model_spec=model_spec, prompt=prompt, max_tokens=max(512, min(max_tokens, 1200)), stage="grounding")
    try:
        recheck = _parse_response(text or "", require_sources=False)
    except Exception as exc:
        recheck = {
            "status": "insufficient_evidence",
            "claim_verdict": "uncertain",
            "issue_supported": None,
            "reason": f"발췌문 재검증 응답 파싱 실패: {exc}",
            "evidence_sources": [],
            "evidence_passages": [],
            "evidence_summary": "",
            "search_queries": [],
            "parse_error": str(exc),
        }
    recheck["provider"] = resolved.get("provider", "")
    recheck["model"] = model_spec
    recheck["resolved_model"] = resolved.get("resolved_model", model_spec)
    return recheck, usage


def _obj_get(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj.get(name)
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _gemini_web_chunk_url(chunk: Any) -> str:
    web = _obj_get(chunk, "web")
    uri = _obj_get(web, "uri", "url") if web is not None else None
    if not uri:
        uri = _obj_get(chunk, "uri", "url")
    uri = str(uri or "").strip()
    return uri if re.match(r"^https?://", uri, flags=re.IGNORECASE) else ""


def _gemini_web_chunk_title(chunk: Any) -> str:
    web = _obj_get(chunk, "web")
    title = _obj_get(web, "title") if web is not None else None
    if not title:
        title = _obj_get(chunk, "title")
    return str(title or "").strip()


def _extract_gemini_grounding(resp: Any) -> dict[str, Any]:
    """Recover Google Search grounding metadata even when resp.text is not parseable."""
    queries: list[str] = []
    sources: list[str] = []
    evidence_passages: list[dict[str, Any]] = []
    candidate_sources: list[str] = []

    candidates = _as_list(_obj_get(resp, "candidates"))
    for candidate in candidates:
        metadata = _obj_get(candidate, "grounding_metadata", "groundingMetadata")
        if metadata is None:
            continue
        for query in _as_list(_obj_get(metadata, "web_search_queries", "webSearchQueries")):
            query_text = str(query or "").strip()
            if query_text and query_text not in queries:
                queries.append(query_text)

        chunks = _as_list(_obj_get(metadata, "grounding_chunks", "groundingChunks"))
        chunk_urls: list[str] = []
        chunk_titles: list[str] = []
        for chunk in chunks:
            url = _gemini_web_chunk_url(chunk)
            title = _gemini_web_chunk_title(chunk)
            chunk_urls.append(url)
            chunk_titles.append(title)
            if url and url not in candidate_sources:
                candidate_sources.append(url)

        supports = _as_list(_obj_get(metadata, "grounding_supports", "groundingSupports"))
        for index, support in enumerate(supports, start=1):
            segment = _obj_get(support, "segment")
            text = str(_obj_get(segment, "text") or _obj_get(support, "text") or "").strip()
            indices = _as_list(
                _obj_get(
                    support,
                    "grounding_chunk_indices",
                    "groundingChunkIndices",
                    "grounding_chunks",
                    "groundingChunks",
                )
            )
            if not indices and len(chunk_urls) == 1:
                indices = [0]
            for raw_index in indices:
                try:
                    chunk_index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if chunk_index < 0 or chunk_index >= len(chunk_urls):
                    continue
                url = chunk_urls[chunk_index]
                if not url:
                    continue
                if url not in sources:
                    if len(sources) >= 3:
                        continue
                    sources.append(url)
                evidence_passages.append({
                    "id": f"G{index}:{chunk_index}",
                    "url": url,
                    "title": chunk_titles[chunk_index],
                    "quote_or_paragraph": text[:1800],
                    "key_sentence": text[:800],
                    "stance": "insufficient_evidence",
                    "why_relevant": (
                        f"Gemini grounding support linked this response segment to "
                        f"{chunk_titles[chunk_index] or url}."
                    ),
                })

    return {
        "search_queries": queries,
        "evidence_sources": sources,
        "evidence_passages": evidence_passages,
        "grounding_candidate_count": len(candidate_sources),
        "grounding_cited_source_count": len(sources),
    }


def _merge_gemini_metadata(payload: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    payload_queries = payload.get("search_queries") if isinstance(payload.get("search_queries"), list) else []
    for query in metadata.get("search_queries", []) or []:
        if query not in payload_queries:
            payload_queries.append(query)
    payload["search_queries"] = payload_queries

    payload_sources = payload.get("evidence_sources") if isinstance(payload.get("evidence_sources"), list) else []
    for source in metadata.get("evidence_sources", []) or []:
        if source not in payload_sources:
            payload_sources.append(source)
    payload["evidence_sources"] = payload_sources

    payload_passages = payload.get("evidence_passages") if isinstance(payload.get("evidence_passages"), list) else []
    seen = {
        (str(row.get("url") or ""), str(row.get("quote_or_paragraph") or row.get("key_sentence") or ""))
        for row in payload_passages
        if isinstance(row, dict)
    }
    for passage in metadata.get("evidence_passages", []) or []:
        key = (str(passage.get("url") or ""), str(passage.get("quote_or_paragraph") or passage.get("key_sentence") or ""))
        if key not in seen:
            payload_passages.append(passage)
            seen.add(key)
    payload["evidence_passages"] = payload_passages

    if metadata.get("evidence_sources"):
        payload["gemini_metadata_recovered"] = True
    return payload


def _build_gemini_pre_verifier_search_prompt(
    issue: dict[str, Any],
    current_date: str,
) -> str:
    questions = [
        str(value or "").strip()
        for value in issue.get("verification_questions", []) or []
        if str(value or "").strip()
    ]
    if not questions:
        questions = [str(issue.get("verification_question") or "").strip()]
    questions = list(dict.fromkeys(question for question in questions if question))
    return f"""당신은 동일한 claim을 표현한 한국어·영어 검증 질문으로 Google Search 관련 문서를 찾는 근거 수집기입니다.
현재 날짜: {current_date}

검증 질문(JSON 배열):
{json.dumps(questions, ensure_ascii=False)}

반드시 Google Search를 사용하세요.
검증 질문은 검색어 그 자체입니다. 위 배열의 한국어·영어 질문을 각각 글자와 의미를 바꾸지 않고
그대로 검색하세요. 질문을 짧게 줄이거나, 키워드로 분해하거나, 추가 번역하거나,
동의어·추정 정답·사건·인물·장소 또는 세 번째 검색어를 만들지 마세요.
이 generate_content 호출 안에서 두 검색어를 모두 사용하고 후속 API 재호출은 하지 마세요.

공식 기관, 정부·국제기구, 사료·학술 자료, 박물관·대학의 전문 자료,
공신력 있는 전문 참고자료만 최종 답변의 근거로 사용하세요.
위키백과는 단독 근거로도 사용할 수 있습니다.
나무위키·기타 사용자 위키, 개인 블로그, 관광·판매·홍보 사이트, 일반 Q&A,
다른 출처를 재인용한 글은 최종 답변의 근거로 인용하지 마세요.
검색 과정에서 본 모든 문서가 아니라 최종 판단에 실제로 사용한 문서만
답변에 연결하고, 최대 3개만 사용하세요.
검색 결과에서 검증 질문과 직접 관련되어 확인되는 내용만 한두 문장으로 작성하세요.
claim의 참·거짓, 이슈 여부, 문서 신뢰도, 지지·반박 관계를 판정하거나 이유를 만들지 마세요.
관련 내용을 확인할 수 없으면 다른 설명 없이 INSUFFICIENT_EVIDENCE만 작성하세요.
JSON이나 항목명을 사용하지 말고, 확인되는 내용 또는 INSUFFICIENT_EVIDENCE만 반환하세요.
"""


def _gemini_google_search_config(max_tokens: int) -> types.GenerateContentConfig:
    """Build a portable Gemini Google Search request configuration.

    Search-tool requests intentionally avoid structured-output and thinking
    controls because support for those fields differs across Gemini models and
    compatible endpoints.  The prompt and existing parser enforce the response
    contract instead.
    """
    del max_tokens
    return types.GenerateContentConfig(
        temperature=0.0,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )


def _call_gemini_pre_verifier_search(
    *,
    issue: dict[str, Any],
    current_date: str,
    model_spec: str = "gemini",
    max_tokens: int = 700,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one Gemini Google Search comparison and normalize its grounding metadata."""
    resolved = _resolve_model_spec(model_spec)
    if resolved.get("provider") != "gemini":
        raise ValueError("Gemini pre-verifier search requires a Gemini model")
    model = str(resolved.get("resolved_model") or model_spec)
    questions = [
        str(value or "").strip()
        for value in issue.get("verification_questions", []) or []
        if str(value or "").strip()
    ]
    if not questions:
        questions = [str(issue.get("verification_question") or "").strip()]
    questions = list(dict.fromkeys(question for question in questions if question))
    if not questions:
        raise ValueError("verification_questions are required")
    question = questions[0]

    prompt = _build_gemini_pre_verifier_search_prompt(issue, current_date)
    contents = [types.Part.from_text(text=prompt)]
    config = _gemini_google_search_config(max_tokens)
    last_exc: Exception | None = None
    client_sequence = get_gemini_client_sequence()
    for index, (client_name, client) in enumerate(client_sequence):
        try:
            def call_api():
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

            resp = api_call_with_retry(call_api)
            metadata = _extract_gemini_grounding(resp)
            raw_text = str(getattr(resp, "text", "") or "")
            sources = [
                str(url).strip()
                for url in metadata.get("evidence_sources", []) or []
                if str(url).strip()
            ]
            actual_queries = [
                re.sub(r"\s+", " ", str(value or "").strip())
                for value in metadata.get("search_queries", []) or []
                if str(value or "").strip()
            ]
            expected_queries = [
                re.sub(r"\s+", " ", value)
                for value in questions
            ]
            query_contract_followed = (
                len(actual_queries) == len(expected_queries)
                and set(actual_queries) == set(expected_queries)
            )
            confirmed_content = raw_text.strip()
            explicitly_insufficient = (
                confirmed_content.upper() == "INSUFFICIENT_EVIDENCE"
            )
            if not query_contract_followed:
                status = "query_contract_violation"
            else:
                status = (
                    "found"
                    if sources
                    and confirmed_content
                    and not explicitly_insufficient
                    else "insufficient_evidence"
                )
            result = {
                "candidate_id": str(issue.get("candidate_id") or ""),
                "verification_question": question,
                "verification_questions": questions,
                "status": status,
                "confirmed_content": (
                    _trim_text(confirmed_content, 1200)
                    if status == "found"
                    else ""
                ),
                "search_queries": actual_queries,
                "query_contract_followed": query_contract_followed,
                "evidence_sources": sources if query_contract_followed else [],
                "evidence_passages": (
                    list(metadata.get("evidence_passages", []) or [])
                    if query_contract_followed
                    else []
                ),
                "rejected_evidence_sources": (
                    [] if query_contract_followed else sources
                ),
                "grounding_candidate_count": int(
                    metadata.get("grounding_candidate_count", 0) or 0
                ),
                "grounding_cited_source_count": int(
                    metadata.get("grounding_cited_source_count", 0) or 0
                ),
                "provider": "gemini",
                "model": model,
                "client": client_name,
                "search_mode": "gemini_google_search_tool",
                "raw_response": _trim_text(raw_text, 3000),
            }
            if not query_contract_followed:
                result["error"] = (
                    "Gemini가 승인된 한국어·영어 질문만 그대로 검색하지 않았습니다: "
                    f"expected={expected_queries!r}, actual={actual_queries!r}"
                )
            usage = _usage_from_gemini(resp, model)
            usage["web_search_requests"] = 1 if (
                result["search_queries"] or sources
            ) else 0
            usage["web_search_queries"] = result["search_queries"]
            usage["web_search_sources"] = (
                result["evidence_sources"]
                if query_contract_followed
                else []
            )
            return result, usage
        except Exception as exc:
            last_exc = exc
            if is_retryable_api_error(exc) and index < len(client_sequence) - 1:
                continue
            break
    return {
        "candidate_id": str(issue.get("candidate_id") or ""),
        "verification_question": question,
        "verification_questions": questions,
        "status": "grounding_unavailable",
        "confirmed_content": "",
        "error": f"Gemini Google Search 호출 실패: {last_exc}",
        "search_queries": [],
        "evidence_sources": [],
        "evidence_passages": [],
        "provider": "gemini",
        "model": model,
        "search_mode": "gemini_google_search_tool",
    }, _empty_token_usage()


def _call_pre_verifier_web_search(
    *,
    issue: dict[str, Any],
    model_spec: str,
    current_date: str,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any], str, str, str]:
    """Dispatch native web search by provider and return one common contract."""
    resolved = _resolve_model_spec(model_spec)
    provider = str(resolved.get("provider") or "")
    resolved_model = str(resolved.get("resolved_model") or model_spec)

    if provider == "gemini":
        result, usage = _call_gemini_pre_verifier_search(
            issue=issue,
            current_date=current_date,
            model_spec=model_spec,
            max_tokens=max_tokens,
        )
        payload = {
            "match_terms": [],
            "evidence_sources": list(result.get("evidence_sources") or []),
            "evidence_passages": list(result.get("evidence_passages") or []),
            "search_confirmed_content": str(
                result.get("confirmed_content") or ""
            ),
        }
        raw_response = str(result.get("raw_response") or "")
        parse_error = str(result.get("error") or "")
        return payload, usage, resolved_model, parse_error, raw_response

    if provider != "openai":
        raise ValueError(
            "pre-verifier web evidence supports OpenAI or Gemini models"
        )

    prompt = _build_pre_verifier_search_prompt(issue, current_date)
    text, usage, resolved_call = _call_llm(
        model_spec=model_spec,
        prompt=prompt,
        max_tokens=max_tokens,
        stage="grounding",
        web_search=True,
        web_search_max_calls=_pre_verifier_evidence_max_tool_calls(),
        web_search_force=False,
        web_search_context_size=_pre_verifier_evidence_search_context_size(),
    )
    resolved_model = str(
        resolved_call.get("resolved_model")
        or resolved.get("resolved_model")
        or model_spec
    )
    parse_error = ""
    try:
        payload = _parse_response(text or "", require_sources=False)
    except Exception as exc:
        parse_error = str(exc)
        payload = {
            "match_terms": [],
            "evidence_sources": [],
            "evidence_passages": [],
        }
    if not isinstance(payload, dict):
        payload = {}
    return payload, usage, resolved_model, parse_error, text or ""


def _call_gemini_search_grounding(
    *,
    model_spec: str,
    issue: dict[str, Any],
    current_date: str,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    resolved = _resolve_model_spec(model_spec)
    model = resolved.get("resolved_model", model_spec)
    prompt = _build_grounding_prompt(issue, current_date)
    contents = [types.Part.from_text(text=prompt)]
    config = _gemini_google_search_config(max_tokens)
    last_exc: Exception | None = None
    for index, (client_name, client) in enumerate(get_gemini_client_sequence()):
        try:
            def call_api():
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

            resp = api_call_with_retry(call_api)
            metadata = _extract_gemini_grounding(resp)
            try:
                payload = _parse_response(resp.text or "", require_sources=False)
            except Exception as parse_exc:
                payload = {
                    "status": "insufficient_evidence",
                    "claim_verdict": "uncertain",
                    "issue_supported": None,
                    "reason": f"Gemini 응답 본문 파싱 실패, grounding metadata로 source를 복구했습니다: {parse_exc}",
                    "evidence_sources": [],
                    "evidence_passages": [],
                    "evidence_summary": "",
                    "search_queries": [],
                    "parse_error": str(parse_exc),
                }
            payload = _merge_gemini_metadata(payload, metadata)
            if payload.get("status") in {"supports_issue", "refutes_issue"} and not payload.get("evidence_sources"):
                payload["pre_source_required_status"] = payload.get("status")
                payload["status"] = "insufficient_evidence"
                payload["claim_verdict"] = "uncertain"
                payload["issue_supported"] = None
            payload["model"] = model
            payload["model_spec"] = model_spec
            payload["provider"] = "gemini"
            payload["client"] = client_name
            payload["search_mode"] = "gemini_google_search_tool"
            payload["gemini_metadata"] = {
                "search_query_count": len(metadata.get("search_queries", []) or []),
                "source_count": len(metadata.get("evidence_sources", []) or []),
                "support_passage_count": len(metadata.get("evidence_passages", []) or []),
            }
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
        "model_spec": model_spec,
        "provider": "gemini",
        "search_mode": "gemini_google_search_tool",
    }, _empty_token_usage()


def _call_text_grounding(
    *,
    model_spec: str,
    issue: dict[str, Any],
    current_date: str,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    prompt = _build_grounding_json_prompt(issue, current_date)
    text, usage, resolved = _call_llm(model_spec=model_spec, prompt=prompt, max_tokens=max_tokens, stage="grounding")
    payload = _parse_response(text or "")
    payload["model"] = model_spec
    payload["resolved_model"] = resolved.get("resolved_model", model_spec)
    payload["provider"] = resolved.get("provider", "")
    payload["search_mode"] = "model_reported_sources_no_native_search_tool"
    return payload, usage


def _pre_verifier_evidence_targets(payload: dict[str, Any]) -> list[dict[str, Any]]:
    issues_by_type = payload.get("issues_by_type") if isinstance(payload.get("issues_by_type"), dict) else {}
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(issue: dict[str, Any], category: str, candidate_id: str) -> None:
        if candidate_id in seen:
            return
        seen.add(candidate_id)
        classification_hints = [
            _trim_text(verdict.get("reason") or "", 320)
            for verdict in issue.get("model_classifications", []) or []
            if isinstance(verdict, dict)
            and str(verdict.get("reason") or "").strip()
        ][:3]
        targets.append({
            "candidate_id": candidate_id,
            "issue_id": str(issue.get("issue_id") or candidate_id),
            "claim_id": str(issue.get("claim_id") or ""),
            "category": category,
            "basis_code": str(issue.get("basis_code") or ""),
            "resolved_claim": str(issue.get("resolved_claim") or ""),
            "claim_text": str(issue.get("claim_text") or ""),
            "location": issue.get("location") if isinstance(issue.get("location"), dict) else {},
            "context": issue.get("context") if isinstance(issue.get("context"), dict) else {},
            "classification_hints": classification_hints,
        })

    for category in ("temporal_error", "factual_error"):
        for issue in issues_by_type.get(category) or []:
            if not isinstance(issue, dict):
                continue
            issue_id = str(issue.get("issue_id") or "")
            if issue_id:
                add(issue, category, issue_id)

    return targets


def _normalized_claim_retrieval_key(issue: dict[str, Any]) -> str:
    claim_id = str(issue.get("claim_id") or "").strip()
    claim = str(issue.get("resolved_claim") or issue.get("claim_text") or "").strip().lower()
    claim = re.sub(r"\s+", " ", claim)
    claim = re.sub(r"[^\w가-힣.%+#/\-]+", " ", claim)
    claim = re.sub(r"\s+", " ", claim).strip()
    return f"{claim_id}\x1f{claim}" if claim_id else claim


def _group_pre_verifier_targets(
    targets: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for target in targets:
        key = _normalized_claim_retrieval_key(target)
        if not key:
            key = f"candidate:{target.get('candidate_id', '')}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(target)
    return [groups[key] for key in order]


def _split_transcript_sentences(text: str) -> list[str]:
    return [
        match.group(0).strip()
        for match in re.finditer(
            r""".+?(?:[.!?。！？]+["'”’)\]]*(?=\s|$)|\n+|$)""",
            str(text or "").strip(),
            flags=re.DOTALL,
        )
        if match.group(0).strip()
    ]


def _normalized_sentence_match_text(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(text or "").lower())


def _best_claim_sentence_index(
    sentence_records: list[dict[str, str]],
    candidate_indices: list[int],
    *,
    claim_text: str,
    resolved_claim: str,
) -> int | None:
    if not candidate_indices:
        return None
    signals = [
        _normalized_sentence_match_text(value)
        for value in (claim_text, resolved_claim)
        if _normalized_sentence_match_text(value)
    ]
    if not signals:
        return candidate_indices[0]

    def score(index: int) -> tuple[float, float, int]:
        sentence = _normalized_sentence_match_text(sentence_records[index]["text"])
        ratios = [
            SequenceMatcher(None, sentence, signal).ratio()
            for signal in signals
        ]
        containment = max(
            (
                min(len(sentence), len(signal)) / max(1, max(len(sentence), len(signal)))
                if sentence in signal or signal in sentence
                else 0.0
            )
            for signal in signals
        )
        return containment, max(ratios, default=0.0), -index

    return max(candidate_indices, key=score)


def _attach_pre_verifier_transcript_context(
    targets: list[dict[str, Any]],
    merged_clean_path: str | Path | None,
) -> None:
    if not merged_clean_path:
        return
    path = Path(merged_clean_path)
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return

    contexts_by_id: dict[str, dict[str, Any]] = {}
    contexts_by_slide: dict[int, list[dict[str, Any]]] = {}
    slide_context_by_number: dict[int, dict[str, Any]] = {}
    ordered_contexts: list[dict[str, Any]] = []
    for slide in payload.get("slides") or []:
        if not isinstance(slide, dict):
            continue
        try:
            slide_number = int(slide.get("slide_number") or 0)
        except (TypeError, ValueError):
            continue
        if slide_number <= 0:
            continue
        slide_context_by_number[slide_number] = {
            "slide_number": slide_number,
            "title": _trim_text(slide.get("title") or "", 240),
            "text": _trim_text(
                slide.get("t1")
                or slide.get("slide_text")
                or "",
                2800,
            ),
        }
        rows: list[dict[str, Any]] = []
        for index, context in enumerate(slide.get("contexts") or []):
            if not isinstance(context, dict):
                continue
            row = dict(context)
            row.setdefault("slide_number", slide_number)
            row["_local_index"] = index
            context_id = str(row.get("context_id") or "").strip()
            if context_id:
                contexts_by_id[context_id] = row
            rows.append(row)
            ordered_contexts.append(row)
        contexts_by_slide[slide_number] = rows
    sentence_records: list[dict[str, str]] = []
    sentence_indices_by_context: dict[str, list[int]] = {}
    for context in ordered_contexts:
        context_id = str(context.get("context_id") or "").strip()
        for sentence in _split_transcript_sentences(context.get("text") or ""):
            sentence_indices_by_context.setdefault(context_id, []).append(
                len(sentence_records)
            )
            sentence_records.append({
                "context_id": context_id,
                "text": sentence,
            })

    for target in targets:
        context_meta = target.get("context") if isinstance(target.get("context"), dict) else {}
        context_ids = [
            str(value).strip()
            for value in (
                context_meta.get("context_ids")
                or [context_meta.get("context_id")]
            )
            if str(value or "").strip()
        ]
        centers = [contexts_by_id[context_id] for context_id in context_ids if context_id in contexts_by_id]
        location = target.get("location") if isinstance(target.get("location"), dict) else {}
        try:
            slide_number = int(location.get("slide_number") or 0)
        except (TypeError, ValueError):
            slide_number = 0
        if not slide_number and centers:
            try:
                slide_number = int(centers[0].get("slide_number") or 0)
            except (TypeError, ValueError):
                slide_number = 0
        slide_contexts = contexts_by_slide.get(slide_number) or []
        if not centers and slide_contexts:
            centers = slide_contexts[:1]
        center_context_ids = [
            str(context.get("context_id") or "").strip()
            for context in centers
            if str(context.get("context_id") or "").strip()
        ]
        center_sentence_indices = [
            index
            for context_id in center_context_ids
            for index in sentence_indices_by_context.get(context_id, [])
        ]
        target_sentence_index = _best_claim_sentence_index(
            sentence_records,
            center_sentence_indices,
            claim_text=str(target.get("claim_text") or ""),
            resolved_claim=str(target.get("resolved_claim") or ""),
        )
        selected: list[dict[str, str]] = []
        if target_sentence_index is not None:
            start = max(0, target_sentence_index - 1)
            end = min(len(sentence_records), target_sentence_index + 2)
            selected = sentence_records[start:end]
        lines = []
        selected_ids = []
        for sentence in selected:
            text = str(sentence.get("text") or "").strip()
            if not text:
                continue
            context_id = str(sentence.get("context_id") or "").strip()
            if context_id and context_id not in selected_ids:
                selected_ids.append(context_id)
            lines.append(f"{context_id}: {text}" if context_id else text)
        target["transcript_context_ids"] = selected_ids
        target["transcript_context"] = "\n".join(lines)
        target["slide_context"] = copy.deepcopy(
            slide_context_by_number.get(
                slide_number,
                {
                    "slide_number": slide_number or None,
                    "title": "",
                    "text": "",
                },
            )
        )
        target["reference_context_ids"] = []
        target["reference_context"] = ""


def _build_pre_verifier_query_plan_prompt(
    issues: list[dict[str, Any]],
    *,
    current_date: str,
) -> tuple[str, list[str]]:
    candidate_ids: list[str] = []
    cases: list[dict[str, Any]] = []
    shared_slide_context: dict[str, Any] = {}
    if issues and isinstance(issues[0].get("slide_context"), dict):
        shared_slide_context = copy.deepcopy(issues[0].get("slide_context") or {})
    for issue in issues:
        candidate_id = str(issue.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        candidate_ids.append(candidate_id)
        cases.append({
            "candidate_id": candidate_id,
            "category": str(issue.get("category") or ""),
            "resolved_claim": str(issue.get("resolved_claim") or ""),
            "claim_text": str(issue.get("claim_text") or ""),
            "transcript_context": str(issue.get("transcript_context") or ""),
        })

    prompt = f"""당신은 웹 검색 전에 강의 claim의 검증 질문을 만드는 라우터입니다.
현재 날짜: {current_date}

같은 슬라이드의 candidate들을 한 번에 처리하되, 각 candidate는 독립적으로 판단하세요.
resolved_claim과 claim_text가 검색의 유일한 중심입니다. 앞뒤 전사 한 문장과 슬라이드 텍스트는
생략된 주체·지시어·시기·범위를 해소하는 데만 사용하고, claim에 없는 사건·인물·장소·인과관계·정답을
새로 추가하지 마세요. 슬라이드와 전사에는 오류나 ASR 흔들림이 있을 수 있으므로 사실 근거로 쓰지 마세요.

주어·지시 대상 해소 우선 규칙:
- resolved_claim에 검증 대상의 주어가 없거나 "해당 화면", "이 작품", "그 기술"처럼 대상이 식별되지 않고,
  claim_text에도 이를 해소할 명시적 대상명이 없다면, 해당 candidate의 transcript_context와 공유
  slide_context를 함께 확인하여 무엇에 관한 claim인지 먼저 식별하세요.
- 주변 전사와 슬라이드에서 하나의 대상만 명확히 연결되면 그 고유명사·장소·작품명·제품명을
  verification_question에 반드시 명시하세요. "해당 화면" 같은 미해결 지시어를 검색 질문에 그대로
  남기지 마세요.
- 이때 검증의 중심은 계속 claim_text와 resolved_claim이 나타내는 동일한 술어·관계·수치·조건입니다.
  문맥에서 대상명만 해소하고, 주변 context의 다른 주장을 검색 대상으로 바꾸거나 새 claim을 만들지 마세요.
- 전사와 슬라이드는 검색 대상을 식별하기 위한 문맥일 뿐, claim이 참이라는 근거는 아닙니다.
- 문맥을 모두 확인해도 대상 후보가 둘 이상이거나 특정할 수 없을 때만 web_check=false와
  basis_code=context_unresolved를 사용하세요.

web_check=true 조건:
- factual_error 또는 temporal_error이며 외부 문서로 claim의 참·거짓을 직접 판정할 수 있다.
- 주체와 핵심 관계를 신뢰할 수 있게 식별할 수 있다.

web_check=false 조건:
- 앞뒤 문장과 슬라이드를 함께 봐도 주체나 지시 대상이 불명확하다.
- 강의 내부 계산, 슬라이드 배치, 설명 품질 또는 해석적 평가만으로 판정해야 한다.

verification_question 규칙:
- web_check=true이면 동일한 claim의 주체·관계·수치·단위·시기·전체/일부 범위를 보존한
  한국어 의문문과 영어 의문문을 각각 한 문장씩 작성한다.
- 두 질문은 언어만 다르고 검증하는 명제와 범위는 완전히 같아야 한다.
- claim이 맞는지 직접 묻는다. 추정 정답이나 반대 명제를 질문에 새로 넣지 않는다.
- site:, 도메인, "공식 자료", "official source", 검색 키워드 나열을 넣지 않는다.
- 고유명사는 한국어 질문에서는 강의 문맥의 표기를, 영어 질문에서는 공식 영문 표기를 우선한다.
- web_check=false이면 두 질문을 모두 빈 문자열로 둔다.

허용 basis_code:
- external_numeric_fact
- external_historical_fact
- current_status
- named_entity_fact
- context_unresolved
- lecture_internal
- interpretive_claim

공유 슬라이드 문맥:
{json.dumps(shared_slide_context, ensure_ascii=False)}

입력:
{json.dumps(cases, ensure_ascii=False)}

모든 candidate_id를 한 번씩 반환하세요. JSON만 반환하세요:
{{
  "plans": [
    {{
      "candidate_id": "I0001",
      "web_check": true,
      "basis_code": "external_historical_fact",
      "verification_question_ko": "한국어로 된 완전한 의문문 한 문장",
      "verification_question_en": "The same complete verification question in English?"
    }}
  ]
}}
"""
    return prompt, candidate_ids


def _normalize_pre_verifier_query_plans(
    text: str,
    *,
    valid_candidate_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], str]:
    try:
        payload = json.loads(_strip_json_fence(text or ""), strict=False)
    except Exception as exc:
        return {}, str(exc)
    rows = payload.get("plans") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return {}, "query plan response plans is not a list"
    plans: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id not in valid_candidate_ids or candidate_id in plans:
            continue
        web_check = row.get("web_check") is True
        basis_code = str(row.get("basis_code") or "").strip().lower()
        if basis_code not in WEB_QUERY_PLAN_BASIS_CODES:
            basis_code = (
                "external_historical_fact"
                if web_check
                else "context_unresolved"
            )
        question_ko = re.sub(
            r"\s+",
            " ",
            str(
                row.get("verification_question_ko")
                or row.get("verification_question")
                or ""
            ).strip(),
        )
        question_en = re.sub(
            r"\s+",
            " ",
            str(row.get("verification_question_en") or "").strip(),
        )
        if not question_ko or not question_en:
            web_check = False
            if basis_code not in {
                "context_unresolved",
                "lecture_internal",
                "interpretive_claim",
            }:
                basis_code = "context_unresolved"
        plans[candidate_id] = {
            "web_check": web_check,
            "basis_code": basis_code,
            "verification_question": _trim_text(question_ko, 500) if web_check else "",
            "verification_question_ko": _trim_text(question_ko, 500) if web_check else "",
            "verification_question_en": _trim_text(question_en, 500) if web_check else "",
            "verification_questions": (
                [
                    _trim_text(question_ko, 500),
                    _trim_text(question_en, 500),
                ]
                if web_check
                else []
            ),
        }
    return plans, ""


def _call_pre_verifier_query_plan(
    issues: list[dict[str, Any]],
    *,
    model_spec: str,
    current_date: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    prompt, candidate_ids = _build_pre_verifier_query_plan_prompt(
        issues,
        current_date=current_date,
    )
    if not candidate_ids:
        return {}, _empty_token_usage(), {
            "status": "not_run",
            "candidate_ids": [],
        }
    text, usage, resolved = _call_llm(
        model_spec=model_spec,
        prompt=prompt,
        max_tokens=_pre_verifier_query_plan_max_tokens(),
        stage="grounding",
    )
    plans, parse_error = _normalize_pre_verifier_query_plans(
        text,
        valid_candidate_ids=set(candidate_ids),
    )
    missing_candidate_ids = sorted(set(candidate_ids) - set(plans))
    for candidate_id in missing_candidate_ids:
        plans[candidate_id] = {
            "web_check": False,
            "basis_code": "query_planner_unavailable",
            "verification_question": "",
            "verification_question_ko": "",
            "verification_question_en": "",
            "verification_questions": [],
        }
    return plans, usage, {
        "status": "parse_failed" if parse_error else "ok",
        "model": model_spec,
        "resolved_model": str(resolved.get("resolved_model") or model_spec),
        "candidate_ids": candidate_ids,
        "returned_candidate_ids": sorted(
            set(candidate_ids) - set(missing_candidate_ids)
        ),
        "missing_candidate_ids": missing_candidate_ids,
        "prompt_chars": len(prompt),
        "parse_error": parse_error,
    }


def _build_pre_verifier_search_prompt(
    issue: dict[str, Any],
    current_date: str,
) -> str:
    question = str(issue.get("verification_question") or "").strip()
    return f"""당신은 이미 문맥 검토를 마친 검증 질문 하나의 웹 문서를 찾습니다.
현재 날짜: {current_date}

아래 verification_question은 검색해야 할 의미를 정하는 계약입니다. 실제 검색어는 핵심
주체·관계·수치·단위·시기·범위를 보존하면서 더 짧은 검색어로 재구성하세요. 권위 있는 원문 자료를
찾는 데 필요하면 다른 언어로 번역해도 됩니다. 줄일 수 있는 긴 질문을 문장 그대로 검색하지 마세요.
다만 주변 주제, 질문에 없는 사건·인물·장소 또는 추정 정답을 검색어에 새로 합치지 마세요.
웹 검색 도구는 정확히 한 번만 사용하고 후속 검색은 하지 마세요.
질문을 직접 판정하는 데 가장 적합한 후보 URL을 최대 3개 반환하세요.
이 단계에서는 문장 발췌, 출처 평가 또는 최종 참·거짓 판정을 하지 마세요.

verification_question: {question}

보조 claim 정보:
- resolved_claim: {issue.get("resolved_claim", "")}
- claim_text: {issue.get("claim_text", "")}
- 앞뒤 전사 문맥: {issue.get("transcript_context", "") or "(not available)"}
- 슬라이드 문맥: {json.dumps(issue.get("slide_context") or {}, ensure_ascii=False)}

verification_question에 주어가 없거나 "해당 화면", "이 작품", "그 기술"처럼 검색 대상을 식별할 수
없는 지시어만 남아 있으면 질문을 그대로 검색하지 마세요. 보조 claim 정보의 전사 문맥과 슬라이드
문맥을 함께 확인해 단일 선행 대상을 찾고, 그 대상의 구체적인 고유명사·장소·작품명·제품명을
실제 검색어에 포함하세요. 문맥으로도 대상을 하나로 확정할 수 없을 때만 후보 URL을 빈 배열로
반환하세요. 전사와 슬라이드의 내용 자체는 사실 근거로 간주하지 마세요.
검색 대상의 술어·관계·수치·조건은 verification_question과 claim_text/resolved_claim에서 유지하고,
문맥의 다른 주장을 새 검색 대상으로 대체하지 마세요.

JSON만 반환하세요:
{{
  "evidence_sources": ["https://source-1", "https://source-2", "https://source-3"]
}}
"""


def _build_pre_verifier_evidence_prompt(issue: dict[str, Any], current_date: str) -> str:
    return f"""당신은 틀릴 가능성이 있어 선별된 강의 claim 하나를 검증하는 웹 검색 실행기입니다.
현재 날짜: {current_date}

검색어 생성 규칙이 가장 높은 우선순위입니다.
1. resolved_claim은 claim_text에서 지시어와 생략된 대상을 최소한으로 해소한 한 문장입니다.
2. 모든 claim에서 resolved_claim, claim_text, 바로 앞뒤 한 문장, 해당 슬라이드의 제목과 텍스트를
   함께 읽으세요. 이를 통해 주체·지시 대상·고유명사·관계·범위와 해당 발화가 정의인지 예시인지
   구분한 뒤 검색 대상을 정하세요.
   resolved_claim에 주어가 없거나 미해결 지시어만 있고 claim_text에도 대상명이 없다면,
   transcript_context와 slide_context를 함께 확인해 단일 선행 대상을 먼저 복원하세요.
   단일 대상이 확인되면 그 고유명사·장소·작품명·제품명을 검색어에 반드시 포함하고, "해당 화면",
   "이 작품", "그 기술" 같은 표현만으로 검색하지 마세요.
   대상 복원 뒤에도 검증할 술어·관계·수치·조건은 원래 claim_text와 resolved_claim의 범위를 유지하세요.
3. 검색어의 중심은 반드시 대상 claim이어야 합니다. 전사·슬라이드 문맥은 claim에서 생략되거나
   축약된 대상을 복원하고 의미와 적용 범위를 한정하는 데 사용하되, 인접한 별도 주장이나 슬라이드의
   다른 사실을 대상 claim에 새로 합치지 마세요.
4. 전사 문맥에는 ASR 흔들림이 있을 수 있고 슬라이드 문맥에도 강의자의 오류가 그대로 적혀 있을 수
   있습니다. 두 문맥은 검색 대상의 식별과 슬라이드에 명시된 출처 URL 발견에만 사용하고, claim의
   참·거짓을 입증하는 근거로 사용하거나 문맥의 표현을 사실로 추가하지 마세요.
5. 분류 모델의 검색 가설은 앞 단계가 고유명사나 지시 대상을 어떻게 해석했는지 보여 주는 비검증
   힌트입니다. 여러 힌트와 슬라이드 문맥이 같은 대상을 가리킬 때 검색어의 고유명사로 사용할 수 있지만,
   이를 사실 근거나 정답으로 간주하지 말고 웹 문서로 반드시 확인하세요.
6. 문맥과 검색 가설을 함께 보아도 고유명사나 주체를 신뢰할 수 있게 특정하지 못하면 일반명사를
   임의로 번역하거나
   이름을 추측해 검색하지 마세요. verification_target에 식별 불가 대상을 적고 evidence_sources를
   빈 배열로 반환하세요.
7. 검색 전에 resolved_claim을 주체, 관계, 객체, 범위, 시기, 수치·단위 같은 검증 가능한 요소로 나누고,
   이슈 유형과 일반적으로 확립된 지식을 이용해 틀렸을 가능성이 가장 높은 요소 하나를 잠정적으로
   판정하세요. 이것은 검색 방향을 정하기 위한 가설이며 최종 이슈 판정이 아닙니다.
8. 의심되는 요소에 널리 알려진 올바른 대비 항목이 있다면, 원래 주장과 그 대비 항목 중 어느 쪽이
   맞는지를 직접 확인하는 질문을 만드세요. 예를 들어 전체인지 일부인지, 특정 시기인지 다른 시기인지,
   입력인지 출력인지, 제시된 수치인지 계산된 수치인지를 구분해 물으세요.
9. 대비 항목을 확신할 수 없다면 억지로 정답을 만들지 말고, 의심되는 요소의 정확한 값·범위·관계를
   묻는 질문을 만드세요. claim과 무관한 기관명·주변 사례·배경지식은 추가하지 마세요.
10. resolved_claim에 명시된 주체와 핵심 관계를 유지하되, 오류로 의심되는 범위·시기·수치·대상은
   검증 대상이므로 그대로 사실처럼 전제하지 마세요.
11. 검색 언어는 claim의 언어가 아니라 가장 신뢰할 만한 1차·공식 출처가 존재할 가능성이 높은 언어를
   선택하세요. 한국의 기관·제도·사건은 한국어를 우선하고, 해외 인물·기관·문화유산·국제기구·국제표준은
   해당 대상의 공식 명칭을 사용한 영어 또는 현지 공식 언어를 우선하세요.
12. 원문이 한국어여도 해외 대상의 공식 자료가 영어로 제공된다면 핵심 고유명사와 의심되는 오류 지점을
   자연스러운 영어 의문문으로 검색하세요. 한 질의에 한국어와 영어 번역을 불필요하게 중복하지 마세요.
13. claim의 어느 부분이 맞거나 틀린지 직접 확인할 수 있는 자연스러운 의문문 하나로 검색하세요.
   단순 명사 나열이나 키워드 조각으로 만들지 마세요.

위 규칙으로 검색 대상을 식별할 수 있을 때만 웹 검색 도구를 정확히 한 번 사용하세요.
검색 대상을 식별하지 못했다면 웹 검색 도구를 사용하지 마세요. 검색 후에는 후속 검색을 하지 마세요.
검색 결과에서 대상 claim을 직접 검증하는 데 가장 적합한 후보 URL을 최대 3개 반환하세요.
이 단계에서는 페이지 문장을 발췌하지 마세요. 검색 결과가 없으면 배열을 비워 두세요.
검색어는 API 도구 호출 기록에서 수집하므로 응답에 적지 마세요. 여기서는 검색 방향을 위한 잠정 오류
가설까지만 남기고, 문서 적합성, 정확한 문장 발췌, 출처 신뢰도, 최종 이슈 여부와 점수는 판단하지 마세요.

이슈 유형: {issue.get("category", "")}
근거 코드: {issue.get("basis_code", "")}
정리된 claim: {issue.get("resolved_claim", "")}
원래 claim_text: {issue.get("claim_text", "")}
앞뒤 한 문장 전사 문맥:
{issue.get("transcript_context", "") or "(not available)"}
해당 슬라이드 문맥(항상 함께 해석하되 사실 근거가 아님):
{json.dumps(issue.get("slide_context") or {}, ensure_ascii=False)}
분류 모델의 비검증 검색 가설(검색 대상 식별용이며 사실 근거가 아님):
{json.dumps(issue.get("classification_hints") or [], ensure_ascii=False)}

JSON만 반환하세요:
{{
  "verification_target": "우선 검증할 claim의 요소",
  "suspected_error": "그 요소가 어떻게 틀렸을 가능성이 있는지에 대한 짧은 잠정 가설",
  "query_language": "실제 검색에 선택한 언어",
  "query_language_reason": "그 언어가 공식·1차 출처 검색에 적합한 이유",
  "evidence_sources": ["https://source-1", "https://source-2", "https://source-3"]
}}
"""


def _slide_context_source_urls(issue: dict[str, Any]) -> list[str]:
    slide_context = (
        issue.get("slide_context")
        if isinstance(issue.get("slide_context"), dict)
        else {}
    )
    text = " ".join(
        str(slide_context.get(key) or "")
        for key in ("title", "text")
    )
    shortener_domains = {
        "url.kr",
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
    }
    urls: list[str] = []
    for raw_url in re.findall(r"https?://[^\s<>'\"`]+", text):
        url = raw_url.rstrip(".,;:!?)]}〉》")
        parsed = urlparse(url)
        domain = str(parsed.hostname or "").lower().removeprefix("www.")
        # Root homepages and short links are usually slide image/brand credits,
        # not direct documents for the claim. They must not displace a searched
        # evidence page from the three-source verification budget.
        if (
            not domain
            or domain in shortener_domains
            or str(parsed.path or "/") in {"", "/"}
        ):
            continue
        if url and url not in urls:
            urls.append(url)
    return urls[:3]


def _limit_pre_verifier_retrieval_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Bound model-authored retrieval metadata before URL verification."""
    payload.pop("search_queries", None)
    payload["verification_question"] = _trim_text(
        payload.get("verification_question") or "",
        500,
    )
    payload["verification_questions"] = [
        _trim_text(value, 500)
        for value in payload.get("verification_questions", []) or []
        if str(value or "").strip()
    ][:2]
    payload["web_check"] = payload.get("web_check") is True
    payload["web_check_basis_code"] = _trim_text(
        payload.get("web_check_basis_code") or "",
        80,
    )
    payload["verification_target"] = _trim_text(
        payload.get("verification_target") or "",
        300,
    )
    payload["suspected_error"] = _trim_text(
        payload.get("suspected_error") or "",
        500,
    )
    payload["query_language"] = _trim_text(
        payload.get("query_language") or "",
        80,
    )
    payload["query_language_reason"] = _trim_text(
        payload.get("query_language_reason") or "",
        300,
    )
    payload["match_terms"] = [
        str(value).strip()
        for value in (payload.get("match_terms") or [])
        if str(value or "").strip()
    ][:4]
    payload["evidence_sources"] = [
        str(value).strip()
        for value in (payload.get("evidence_sources") or [])
        if re.match(r"^https?://", str(value or "").strip(), flags=re.IGNORECASE)
    ][:3]
    passages = payload.get("evidence_passages")
    payload["evidence_passages"] = (
        [row for row in passages if isinstance(row, dict)][:3]
        if isinstance(passages, list)
        else []
    )
    return payload


def _pre_verifier_source_excerpts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    excerpts: list[dict[str, Any]] = []
    verified_sources = [
        source
        for source in payload.get("verified_sources", []) or []
        if isinstance(source, dict)
        and source.get("document_relevance_eligible", True)
        and source.get("fetch_status") == "ok"
        and source.get(
            "source_trust_eligible",
            source.get("priority_eligible", False),
        )
        and source.get("direct_match")
    ]
    verified_sources.sort(
        key=lambda source: (
            0 if source.get("source_strength") == "strong" else 1,
            int(source.get("source_priority") or 99),
            -float(
                source.get("source_trust_confidence")
                or source.get("trust_score")
                or 0.0
            ),
            str(source.get("url") or ""),
        )
    )
    for source in verified_sources[: _pre_verifier_evidence_verify_max_sources()]:
        passages = source.get("matched_passages") if isinstance(source.get("matched_passages"), list) else []
        selected = next(
            (
                passage
                for passage in passages
                if isinstance(passage, dict) and _evidence_passage_text(passage)
            ),
            None,
        )
        if selected is None:
            continue
        excerpts.append({
            "source_id": f"S{len(excerpts) + 1}",
            "url": str(source.get("url") or ""),
            "domain": str(source.get("domain") or ""),
            "document_relevance": str(source.get("document_relevance") or "direct"),
            "source_strength": str(source.get("source_strength") or "strong"),
            "assessed_source_class": str(
                source.get("assessed_source_class")
                or source.get("source_priority_label")
                or ""
            ),
            "source_priority": source.get("source_priority"),
            "source_priority_label": str(source.get("source_priority_label") or ""),
            "key_sentence": _evidence_passage_text(selected),
            "match_status": str(selected.get("match_status") or ""),
            "match_score": selected.get("match_score"),
        })
    return excerpts


def _build_pre_verifier_semantic_prompt(
    issue: dict[str, Any],
    excerpts: list[dict[str, Any]],
    current_date: str,
) -> str:
    compact_excerpts = [
        {
            "source_id": excerpt.get("source_id", ""),
            "key_sentence": excerpt.get("key_sentence", ""),
        }
        for excerpt in excerpts
    ]
    return f"""You check whether source sentences are semantically relevant to one lecture claim.
Current date: {current_date}

Use only the supplied source sentences. Do not use memory, web search, or the transcript as factual evidence.
Classify every source independently:
- supports_claim: the sentence directly entails the lecture claim in the same subject, relation, scope, region, and time.
- contradicts_claim: the sentence is directly incompatible with the lecture claim, or explicit source values make the
  claim false through one simple deterministic calculation using the current date.
- irrelevant: the sentence is generic, only topically related, uses a different subject/scope/time/region, or lacks
  information required to establish either direction.

Cross-language entity equivalence is allowed. Mere keyword overlap is never enough.
Do not infer absence from silence and do not fill missing facts from memory.
For a conjunction or list claim, supports_claim requires the source sentence to establish every required component.
Support for only some components is irrelevant because it cannot establish the complete claim. A direct contradiction
of any required component may still be contradicts_claim because it makes the conjunction false.
However, an authoritative definition, registry entry, catalogue record, genealogy, specification, or official property
boundary can contradict a claim that asserts a different definition, exclusive actor, necessary condition, lineage,
commission, or broader registered scope. In that case compare the affirmative relation stated by the source rather
than treating it as mere omission.

Resolved claim: {issue.get("resolved_claim", "")}
Original claim_text: {issue.get("claim_text", "")}

Verified source sentences:
{json.dumps(compact_excerpts, ensure_ascii=False)}

Return JSON only:
{{
  "assessments": [
    {{
      "source_id": "S1",
      "relation": "supports_claim | contradicts_claim | irrelevant",
      "confidence": 0.0,
      "reason": "짧은 한국어 근거"
    }}
  ]
}}
"""


def _normalize_semantic_assessments(
    text: str,
    excerpts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    try:
        payload = json.loads(_strip_json_fence(text or ""), strict=False)
    except Exception as exc:
        return [], str(exc)
    rows = payload.get("assessments") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return [], "semantic response assessments is not a list"
    valid_ids = {str(excerpt.get("source_id") or "") for excerpt in excerpts}
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source_id") or "").strip()
        if not source_id or source_id not in valid_ids or source_id in seen:
            continue
        relation = str(row.get("relation") or "").strip().lower()
        if relation not in {"supports_claim", "contradicts_claim", "irrelevant"}:
            relation = "irrelevant"
        seen.add(source_id)
        normalized.append({
            "source_id": source_id,
            "relation": relation,
            "confidence": round(_clamp01(row.get("confidence"), 0.0), 4),
            "reason": _trim_text(row.get("reason") or "", 220),
        })
    for source_id in sorted(valid_ids - seen):
        normalized.append({
            "source_id": source_id,
            "relation": "irrelevant",
            "confidence": 0.0,
            "reason": "의미 판정 결과가 반환되지 않았습니다.",
        })
    return normalized, ""


def _call_pre_verifier_semantic_assessment(
    issue: dict[str, Any],
    payload: dict[str, Any],
    *,
    current_date: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    excerpts = _pre_verifier_source_excerpts(payload)
    if not excerpts:
        return [], [], _empty_token_usage(), {
            "status": "not_run",
            "reason": "출처 본문에서 확인된 후보 문장이 없습니다.",
        }
    model_spec = _pre_verifier_evidence_semantic_model()
    prompt = _build_pre_verifier_semantic_prompt(issue, excerpts, current_date)
    text, usage, resolved = _call_llm(
        model_spec=model_spec,
        prompt=prompt,
        max_tokens=_pre_verifier_evidence_semantic_max_tokens(),
        stage="grounding",
    )
    assessments, parse_error = _normalize_semantic_assessments(text, excerpts)
    retry_count = 0
    if parse_error:
        retry_text, retry_usage, retry_resolved = _call_llm(
            model_spec=model_spec,
            prompt=prompt,
            max_tokens=_pre_verifier_evidence_semantic_max_tokens(),
            stage="grounding",
        )
        retry_assessments, retry_parse_error = _normalize_semantic_assessments(
            retry_text,
            excerpts,
        )
        combined_usage = _empty_token_usage()
        _merge_token_usage(combined_usage, usage)
        _merge_token_usage(combined_usage, retry_usage)
        usage = combined_usage
        retry_count = 1
        if not retry_parse_error:
            assessments = retry_assessments
            parse_error = ""
            resolved = retry_resolved
        else:
            parse_error = f"{parse_error}; retry: {retry_parse_error}"
    metadata = {
        "status": "parse_failed" if parse_error else "ok",
        "model": model_spec,
        "resolved_model": resolved.get("resolved_model", model_spec),
        "parse_error": parse_error,
        "retry_count": retry_count,
    }
    return excerpts, assessments, usage, metadata


def _evidence_passage_text(passage: dict[str, Any]) -> str:
    return _trim_text(
        passage.get("matched_text")
        or passage.get("key_sentence")
        or passage.get("quote_or_paragraph")
        or passage.get("text")
        or "",
        _pre_verifier_evidence_passage_chars(),
    )


def _compact_pre_verifier_evidence(
    issue: dict[str, Any],
    payload: dict[str, Any],
    *,
    model_spec: str,
    resolved_model: str,
    excerpts: list[dict[str, Any]] | None = None,
    semantic_assessments: list[dict[str, Any]] | None = None,
    semantic_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    excerpts = excerpts or []
    assessment_by_id = {
        str(row.get("source_id") or ""): row
        for row in semantic_assessments or []
        if isinstance(row, dict)
    }
    min_confidence = _pre_verifier_evidence_semantic_min_confidence()
    evidence_candidates: list[dict[str, Any]] = []
    for excerpt in excerpts:
        source_id = str(excerpt.get("source_id") or "")
        assessment = assessment_by_id.get(source_id, {})
        relation = str(assessment.get("relation") or "irrelevant")
        confidence = _clamp01(assessment.get("confidence"), 0.0)
        if relation == "irrelevant" or confidence < min_confidence:
            continue
        evidence_candidates.append({
            "source_id": source_id,
            "url": str(excerpt.get("url") or ""),
            "domain": str(excerpt.get("domain") or ""),
            "document_relevance": str(
                excerpt.get("document_relevance") or "direct"
            ),
            "source_strength": str(excerpt.get("source_strength") or "strong"),
            "assessed_source_class": str(
                excerpt.get("assessed_source_class") or ""
            ),
            "source_priority": excerpt.get("source_priority"),
            "source_priority_label": str(excerpt.get("source_priority_label") or ""),
            "key_sentence": str(excerpt.get("key_sentence") or ""),
            "relation_to_claim": relation,
            "relation_confidence": round(confidence, 4),
            "relation_reason": str(assessment.get("reason") or ""),
            "match_status": str(excerpt.get("match_status") or ""),
            "match_score": excerpt.get("match_score"),
        })
    evidence_candidates.sort(
        key=lambda row: (
            0 if row.get("source_strength") == "strong" else 1,
            int(row.get("source_priority") or 99),
            -float(row.get("relation_confidence") or 0.0),
            str(row.get("url") or ""),
        )
    )
    relevant_relations = {
        str(row.get("relation_to_claim") or "")
        for row in evidence_candidates
        if str(row.get("relation_to_claim") or "")
    }
    semantic_metadata = dict(semantic_metadata or {})
    if len(relevant_relations) > 1:
        semantic_metadata.update({
            "status": "mixed_relations",
            "mixed_relations": sorted(relevant_relations),
            "reason": (
                "근거들이 claim의 서로 다른 구성요소를 지지하거나 반박할 수 있어 "
                "각 근거의 관계를 개별 보존합니다."
            ),
        })
    evidence = []
    # Up to VERIFY_MAX_SOURCES documents may be inspected, but only the
    # highest-quality MAX_SOURCES evidence rows are handed to the verifier.
    for row in evidence_candidates[: _pre_verifier_evidence_max_sources()]:
        evidence.append({
            **row,
            "source_id": f"E{len(evidence) + 1}",
        })
    diagnostics = []
    excerpt_by_url = {
        str(excerpt.get("url") or ""): excerpt
        for excerpt in excerpts
    }
    for source in payload.get("verified_sources", []) or []:
        if not isinstance(source, dict):
            continue
        url = str(source.get("url") or "")
        excerpt = excerpt_by_url.get(url, {})
        assessment = assessment_by_id.get(str(excerpt.get("source_id") or ""), {})
        diagnostics.append({
            "url": url,
            "domain": str(source.get("domain") or ""),
            "fetch_status": str(source.get("fetch_status") or ""),
            "document_relevance": str(source.get("document_relevance") or ""),
            "document_relevance_confidence": source.get("document_relevance_confidence"),
            "document_relevance_reason": str(source.get("document_relevance_reason") or ""),
            "document_relevance_eligible": bool(source.get("document_relevance_eligible")),
            "passage_extraction_status": str(source.get("passage_extraction_status") or ""),
            "assessed_source_class": str(source.get("assessed_source_class") or ""),
            "authority_for_claim": str(source.get("authority_for_claim") or ""),
            "source_strength": str(source.get("source_strength") or ""),
            "source_trust_confidence": source.get("source_trust_confidence"),
            "source_trust_reason": str(source.get("source_trust_reason") or ""),
            "source_trust_eligible": bool(source.get("source_trust_eligible")),
            "source_priority": source.get("source_priority"),
            "priority_eligible": bool(source.get("priority_eligible")),
            "direct_match": bool(source.get("direct_match")),
            "key_sentence": str(excerpt.get("key_sentence") or ""),
            "match_status": str(excerpt.get("match_status") or ""),
            "match_score": excerpt.get("match_score"),
            "relation_to_claim": str(assessment.get("relation") or ""),
            "relation_confidence": assessment.get("confidence"),
            "relation_reason": str(assessment.get("reason") or ""),
            "error": _trim_text(source.get("error") or "", 220),
        })
    return {
        "candidate_id": issue.get("candidate_id", ""),
        "issue_id": issue.get("issue_id", ""),
        "claim_id": issue.get("claim_id", ""),
        "category": issue.get("category", ""),
        "basis_code": issue.get("basis_code", ""),
        "status": "verified" if evidence else "insufficient_evidence",
        "evidence": evidence,
        "model": model_spec,
        "resolved_model": resolved_model,
        "search_queries": payload.get("search_queries", []),
        "search_confirmed_content": str(
            payload.get("search_confirmed_content") or ""
        ),
        "verification_question": str(
            payload.get("verification_question") or ""
        ),
        "verification_questions": [
            str(value or "").strip()
            for value in payload.get("verification_questions", []) or []
            if str(value or "").strip()
        ],
        "web_check": payload.get("web_check") is True,
        "web_check_basis_code": str(
            payload.get("web_check_basis_code") or ""
        ),
        "match_terms": payload.get("match_terms", []),
        "verification_target": str(payload.get("verification_target") or ""),
        "suspected_error": str(payload.get("suspected_error") or ""),
        "query_language": str(payload.get("query_language") or ""),
        "query_language_reason": str(
            payload.get("query_language_reason") or ""
        ),
        "source_diagnostics": diagnostics,
        "retrieval_parse_error": str(payload.get("_retrieval_parse_error") or ""),
        "retrieval_response_preview": str(payload.get("_retrieval_response_preview") or ""),
        "source_prefilter": payload.get("source_prefilter", {}),
        "document_relevance": payload.get("document_relevance", {}),
        "passage_extraction": payload.get("passage_extraction", {}),
        "source_trust_assessment": payload.get("source_trust_assessment", {}),
        "semantic_assessment": semantic_metadata,
        "web_search_requests": int(payload.get("web_search_requests", 0) or 0),
        "transcript_context_ids": issue.get("transcript_context_ids", []),
        "transcript_context": issue.get("transcript_context", ""),
        "slide_context": issue.get("slide_context", {}),
        "reference_context_ids": issue.get("reference_context_ids", []),
        "reference_context": issue.get("reference_context", ""),
    }


def _process_pre_verifier_retrieval_payload(
    issue: dict[str, Any],
    payload: dict[str, Any],
    *,
    model_spec: str,
    resolved_model: str,
    max_tokens: int,
    retrieval_usage: dict[str, Any],
    actual_search_queries: list[str],
    web_search_request_count: int,
    parse_error: str = "",
    response_preview: str = "",
    run_individual_assessments: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, dict):
        payload = {}
    sources = payload.get("evidence_sources") if isinstance(payload.get("evidence_sources"), list) else []
    accepted_sources, excluded_sources = _prefilter_source_candidates(sources)
    source_limit = _pre_verifier_evidence_verify_max_sources()
    initially_accepted, _ = _prefilter_source_candidates(sources[:source_limit])
    selected_count = min(source_limit, len(accepted_sources))
    payload["evidence_sources"] = accepted_sources
    payload["source_prefilter"] = {
        "candidate_count": len(sources),
        "accepted_count": len(accepted_sources),
        "excluded_count": len(excluded_sources),
        "selected_count": selected_count,
        "refilled_count": max(
            0,
            selected_count - min(source_limit, len(initially_accepted)),
        ),
        "excluded_sources": excluded_sources,
    }
    payload["web_search_requests"] = int(web_search_request_count)
    if parse_error:
        payload["_retrieval_parse_error"] = parse_error
        payload["_retrieval_response_preview"] = _trim_text(response_preview, 500)
    if parse_error and not accepted_sources:
        return (
            {
                "issue": issue,
                "terminal_evidence": {
                    "candidate_id": issue.get("candidate_id", ""),
                    "issue_id": issue.get("issue_id", ""),
                    "claim_id": issue.get("claim_id", ""),
                    "category": issue.get("category", ""),
                    "basis_code": issue.get("basis_code", ""),
                    "status": "grounding_unavailable",
                    "evidence": [],
                    "model": model_spec,
                    "resolved_model": resolved_model,
                    "web_search_requests": int(web_search_request_count),
                    "search_queries": actual_search_queries,
                    "verification_question": str(
                        issue.get("verification_question") or ""
                    ),
                    "web_check": issue.get("web_check") is True,
                    "web_check_basis_code": str(
                        issue.get("web_check_basis_code") or ""
                    ),
                    "transcript_context_ids": issue.get("transcript_context_ids", []),
                    "transcript_context": issue.get("transcript_context", ""),
                    "slide_context": issue.get("slide_context", {}),
                    "error": parse_error,
                    "retrieval_response_preview": _trim_text(response_preview, 500),
                },
            },
            retrieval_usage,
        )
    payload = _limit_pre_verifier_retrieval_payload(payload)
    # The native search call already returned these URLs. Keep enough of that
    # same result set to replace pages that fail to fetch without searching
    # again.
    payload["evidence_sources"] = accepted_sources[
        : _pre_verifier_evidence_max_fetch_attempts()
    ]
    payload["source_prefilter"]["selected_count"] = len(
        payload.get("evidence_sources") or []
    )
    payload["search_queries"] = actual_search_queries
    payload = _verify_payload_sources(
        issue,
        payload,
        max_sources=_pre_verifier_evidence_verify_max_sources(),
    )
    source_fetch = (
        payload.get("source_fetch")
        if isinstance(payload.get("source_fetch"), dict)
        else {}
    )
    payload["source_prefilter"]["selected_count"] = min(
        source_limit,
        len(accepted_sources),
    )
    payload["source_prefilter"]["fetch_attempt_count"] = int(
        source_fetch.get("attempt_count", 0) or 0
    )
    payload["source_prefilter"]["successful_fetch_count"] = int(
        source_fetch.get("successful_fetch_count", 0) or 0
    )
    payload["source_prefilter"]["refilled_count"] = int(
        source_fetch.get("refilled_count", 0) or 0
    )
    if not run_individual_assessments:
        combined_usage = _empty_token_usage()
        _merge_token_usage(combined_usage, retrieval_usage)
        combined_usage.update({
            "web_search_requests": int(
                retrieval_usage.get("web_search_requests", 0) or 0
            ),
            "web_search_queries": retrieval_usage.get(
                "web_search_queries",
                [],
            ),
            "web_search_sources": retrieval_usage.get(
                "web_search_sources",
                [],
            ),
            "document_relevance_calls": 0,
            "source_trust_assessment_calls": 0,
            "passage_extraction_calls": 0,
            "semantic_assessment_calls": 0,
        })
        return (
            {
                "issue": issue,
                "payload": payload,
                "model_spec": model_spec,
                "resolved_model": resolved_model,
                "terminal_evidence": None,
            },
            combined_usage,
        )
    payload, document_relevance_usage = _call_document_relevance_assessment(
        model_spec=_pre_verifier_evidence_semantic_model(),
        issue=issue,
        payload=payload,
    )
    payload, extraction_usage = _call_passage_extraction_fallback(
        model_spec=_pre_verifier_evidence_semantic_model(),
        issue=issue,
        payload=payload,
        max_tokens=max(max_tokens, _pre_verifier_evidence_semantic_max_tokens()),
        reselect_all=True,
    )
    extraction_total_usage = _empty_token_usage()
    _merge_token_usage(extraction_total_usage, extraction_usage)
    extraction_meta = payload.get("passage_extraction") if isinstance(payload.get("passage_extraction"), dict) else {}
    if extraction_meta.get("parse_error"):
        payload, extraction_retry_usage = _call_passage_extraction_fallback(
            model_spec=_pre_verifier_evidence_semantic_model(),
            issue=issue,
            payload=payload,
            max_tokens=max(max_tokens, _pre_verifier_evidence_semantic_max_tokens()),
            reselect_all=True,
        )
        _merge_token_usage(extraction_total_usage, extraction_retry_usage)
        retry_meta = payload.get("passage_extraction") if isinstance(payload.get("passage_extraction"), dict) else {}
        retry_meta["retry_count"] = 1
        payload["passage_extraction"] = retry_meta
    payload, source_trust_usage = _call_source_trust_assessment(
        model_spec=_pre_verifier_evidence_semantic_model(),
        issue=issue,
        payload=payload,
    )
    combined_usage = _empty_token_usage()
    _merge_token_usage(combined_usage, retrieval_usage)
    _merge_token_usage(combined_usage, document_relevance_usage)
    _merge_token_usage(combined_usage, extraction_total_usage)
    _merge_token_usage(combined_usage, source_trust_usage)
    combined_usage.update({
        "web_search_requests": int(
            retrieval_usage.get("web_search_requests", 0) or 0
        ),
        "web_search_queries": retrieval_usage.get("web_search_queries", []),
        "web_search_sources": retrieval_usage.get("web_search_sources", []),
        "document_relevance_calls": (
            1
            if any(
                int(document_relevance_usage.get(field, 0) or 0)
                for field in TOKEN_USAGE_FIELDS
            )
            else 0
        ),
        "source_trust_assessment_calls": (
            1
            if any(
                int(source_trust_usage.get(field, 0) or 0)
                for field in TOKEN_USAGE_FIELDS
            )
            else 0
        ),
        "passage_extraction_calls": (
            1
            if any(int(extraction_total_usage.get(field, 0) or 0) for field in TOKEN_USAGE_FIELDS)
            else 0
        ),
        "semantic_assessment_calls": 0,
    })
    return (
        {
            "issue": issue,
            "payload": payload,
            "model_spec": model_spec,
            "resolved_model": resolved_model,
            "terminal_evidence": None,
        },
        combined_usage,
    )


def _retrieve_pre_verifier_evidence_material(
    issue: dict[str, Any],
    *,
    model_spec: str,
    current_date: str,
    max_tokens: int,
    run_individual_assessments: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = _resolve_model_spec(model_spec)
    if resolved.get("provider") not in {"openai", "gemini"}:
        raise ValueError(
            "pre-verifier web evidence supports OpenAI or Gemini models"
        )
    web_check = (
        issue.get("web_check") is True
        if "web_check" in issue
        else True
    )
    verification_question = str(
        issue.get("verification_question") or ""
    ).strip()
    if web_check and not verification_question and "web_check" not in issue:
        claim = str(
            issue.get("resolved_claim")
            or issue.get("claim_text")
            or ""
        ).strip()
        verification_question = (
            claim
            if claim.endswith(("?", "？"))
            else f"다음 강의 주장은 사실인가? {claim}"
        )
    if not web_check or not verification_question:
        basis_code = str(
            issue.get("web_check_basis_code")
            or "context_unresolved"
        )
        terminal_status = (
            "grounding_unavailable"
            if basis_code == "query_planner_unavailable"
            else "not_applicable"
        )
        return (
            {
                "issue": issue,
                "terminal_evidence": {
                    "candidate_id": issue.get("candidate_id", ""),
                    "issue_id": issue.get("issue_id", ""),
                    "claim_id": issue.get("claim_id", ""),
                    "category": issue.get("category", ""),
                    "basis_code": issue.get("basis_code", ""),
                    "status": terminal_status,
                    "evidence": [],
                    "model": model_spec,
                    "resolved_model": str(
                        resolved.get("resolved_model") or model_spec
                    ),
                    "web_search_requests": 0,
                    "search_queries": [],
                    "verification_question": verification_question,
                    "web_check": False,
                    "web_check_basis_code": basis_code,
                    "transcript_context_ids": issue.get(
                        "transcript_context_ids",
                        [],
                    ),
                    "transcript_context": issue.get(
                        "transcript_context",
                        "",
                    ),
                    "slide_context": issue.get("slide_context", {}),
                    "reason": (
                        "웹 검색 질문 계획 단계가 실패했습니다."
                        if terminal_status == "grounding_unavailable"
                        else "슬라이드·전사 문맥을 반영한 사전 라우터가 웹 검색 불필요 또는 대상 식별 불가로 판정했습니다."
                    ),
                },
            },
            _empty_token_usage(),
        )
    search_issue = dict(issue)
    search_issue["verification_question"] = verification_question
    payload, usage, resolved_model, parse_error, response_preview = (
        _call_pre_verifier_web_search(
            issue=search_issue,
            model_spec=model_spec,
            current_date=current_date,
            max_tokens=max_tokens,
        )
    )
    payload["verification_question"] = verification_question
    payload["web_check"] = True
    payload["web_check_basis_code"] = str(
        issue.get("web_check_basis_code") or ""
    )
    payload["verification_target"] = verification_question
    payload["suspected_error"] = ""
    payload["query_language"] = ""
    payload["query_language_reason"] = ""
    sources = payload.get("evidence_sources") if isinstance(payload.get("evidence_sources"), list) else []
    slide_source_urls = _slide_context_source_urls(issue)
    sources = slide_source_urls + [
        source
        for source in sources
        if source not in slide_source_urls
    ]
    for url in usage.get("web_search_sources", []) or []:
        url = str(url or "").strip()
        if url and url not in sources:
            sources.append(url)
    payload["evidence_sources"] = sources
    actual_search_queries = list(dict.fromkeys(
        str(query or "").strip()
        for query in (usage.get("web_search_queries", []) or [])
        if str(query or "").strip()
    ))
    return _process_pre_verifier_retrieval_payload(
        issue,
        payload,
        model_spec=model_spec,
        resolved_model=resolved_model,
        max_tokens=max_tokens,
        retrieval_usage=usage,
        actual_search_queries=actual_search_queries,
        web_search_request_count=int(usage.get("web_search_requests", 0) or 0),
        parse_error=parse_error,
        response_preview=response_preview,
        run_individual_assessments=run_individual_assessments,
    )


def _pre_verifier_batch_assessment_max_tokens() -> int:
    try:
        return max(
            1200,
            int(
                os.getenv(
                    "CLASSIFIED_ISSUE_EVIDENCE_BATCH_ASSESSMENT_MAX_TOKENS",
                    "6000",
                )
            ),
        )
    except ValueError:
        return 6000


def _pre_verifier_batch_max_candidates() -> int:
    try:
        return max(
            1,
            int(
                os.getenv(
                    "CLASSIFIED_ISSUE_EVIDENCE_BATCH_MAX_CANDIDATES",
                    "3",
                )
            ),
        )
    except ValueError:
        return 3


def _pre_verifier_batch_max_sources() -> int:
    try:
        return max(
            1,
            int(
                os.getenv(
                    "CLASSIFIED_ISSUE_EVIDENCE_BATCH_MAX_SOURCES",
                    "9",
                )
            ),
        )
    except ValueError:
        return 9


def _pre_verifier_batch_max_prompt_chars() -> int:
    try:
        return max(
            4_000,
            int(
                os.getenv(
                    "CLASSIFIED_ISSUE_EVIDENCE_BATCH_MAX_PROMPT_CHARS",
                    "24000",
                )
            ),
        )
    except ValueError:
        return 24_000


def _build_pre_verifier_batch_assessment_prompt(
    entries: list[dict[str, Any]],
    *,
    current_date: str,
) -> tuple[str, dict[str, tuple[str, dict[str, Any]]], list[str]]:
    cases: list[dict[str, Any]] = []
    source_lookup: dict[str, tuple[str, dict[str, Any]]] = {}
    candidate_ids: list[str] = []
    shared_slide_context = {}
    if entries:
        first_issue = (
            entries[0].get("issue")
            if isinstance(entries[0].get("issue"), dict)
            else {}
        )
        if isinstance(first_issue.get("slide_context"), dict):
            shared_slide_context = first_issue.get("slide_context") or {}
    for entry in entries:
        issue = entry.get("issue") if isinstance(entry.get("issue"), dict) else {}
        material = (
            entry.get("material")
            if isinstance(entry.get("material"), dict)
            else {}
        )
        payload = (
            material.get("payload")
            if isinstance(material.get("payload"), dict)
            else {}
        )
        candidate_id = str(issue.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        terms = _verification_terms(issue, payload)
        sources: list[dict[str, Any]] = []
        for source in payload.get("verified_sources", []) or []:
            if not isinstance(source, dict):
                continue
            if (
                source.get("fetch_status") != "ok"
                or not source.get("_source_text")
            ):
                continue
            source_id = f"{candidate_id}-S{len(sources) + 1}"
            source_lookup[source_id] = (candidate_id, source)
            source_text = str(source.get("_source_text") or "")
            sources.append({
                "source_id": source_id,
                "url": str(source.get("url") or ""),
                "domain": str(source.get("domain") or ""),
                "page_identity_excerpt": _trim_text(source_text, 600),
                "claim_focused_excerpt": _source_text_sample(
                    source_text,
                    terms,
                    max_chars=1800,
                ),
            })
        if not sources:
            continue
        candidate_ids.append(candidate_id)
        cases.append({
            "candidate_id": candidate_id,
            "slide_number": (
                issue.get("location", {}).get("slide_number")
                if isinstance(issue.get("location"), dict)
                else None
            ),
            "category": str(issue.get("category") or ""),
            "basis_code": str(issue.get("basis_code") or ""),
            "resolved_claim": str(issue.get("resolved_claim") or ""),
            "claim_text": str(issue.get("claim_text") or ""),
            "search_confirmed_content": _trim_text(
                payload.get("search_confirmed_content") or "",
                1200,
            ),
            "allowed_source_ids": [
                str(source.get("source_id") or "") for source in sources
            ],
            "sources": sources,
        })

    prompt = f"""당신은 여러 강의 claim과 각 claim에 허용된 웹 문서를 한 번에 검증합니다.
현재 날짜: {current_date}

각 candidate는 완전히 독립적으로 판정하세요. candidate가 allowed_source_ids에 명시하지 않은 문서는
절대 사용하지 마세요. 제공된 문서 조각 밖의 기억이나 추측을 사실 근거로 사용하지 마세요.
같은 배치의 candidate들은 동일한 슬라이드에 속합니다. shared_slide_context는 고유명사와 지시 대상을
해석하기 위한 문맥일 뿐 사실 근거가 아닙니다. 슬라이드 문구가 claim을 반복한다는 이유로 claim을
지지하거나 반박하지 말고, 판정에는 각 candidate에 허용된 웹 문서만 사용하세요.
search_confirmed_content는 검색 단계가 해당 URL들과 연결해 반환한 내용입니다. 문서에서 확인할 위치를
찾는 데 참고하되, 그 문장 자체를 독립 근거로 삼지 말고 반드시 sources의 실제 문서 조각과 대조하세요.

각 source에 대해 다음을 한 번에 판정합니다.
1. relevance
   - direct: 같은 대상을 다루며 claim 전체 또는 claim을 거짓으로 만드는 필수 구성요소 하나를 직접 확인·반박
   - partial: 같은 주제지만 필요한 범위·관계·시기·수치가 빠짐
   - irrelevant: 다른 대상이거나 키워드만 겹침
2. quote
   - direct 또는 partial인 경우 해당 판정에 사용한 원문 문장 하나를 문서 조각에서 글자 그대로 복사
   - partial 문장은 claim 전체의 증거가 아니라 그 문장이 실제로 다루는 구성요소의 참고 근거로만 사용
   - 적절한 문장이 없으면 빈 문자열
3. source_class
   - primary_authority: 해당 사실을 직접 관리·발표하는 공식 기록이나 공식 문서
   - standard: 공인 표준
   - scholarly: 학술 출판물
   - expert_reference: 박물관·대학·전문 백과 등 전문 참고자료
   - encyclopedia: 위키백과. 직접 관련 문장이 확인되면 단독 근거로 허용
   - official_secondary: 공식 기관의 2차 안내
   - general_secondary: 언론·상업 출판·일반 참고자료
   - user_generated | promotional | unknown
4. authority_for_claim
   - high | medium | low
5. relation
   - supports_claim: 같은 주체·관계·범위·시기에서 claim 전체를 직접 뒷받침
   - contradicts_claim: claim의 필수 구성요소 하나와 직접 충돌하여 claim을 거짓으로 만듦
   - irrelevant: 어느 방향도 직접 판정할 수 없음

복합 claim은 일부 구성요소만 지지하면 supports_claim이 아닙니다. 반대로 필수 구성요소 하나가 직접
반박되면 contradicts_claim이 될 수 있습니다. 문서에 말이 없다는 사실만으로 반박하지 마세요.

candidate별 claim_verdict:
- true: 신뢰 가능한 direct 근거가 claim 전체를 확립하고 직접 반박 근거가 없음
- false: 신뢰 가능한 direct 근거가 claim의 필수 구성요소를 직접 반박
- uncertain: 근거 부족, partial/irrelevant뿐임, 출처가 약함, 또는 지지·반박이 혼재

공유 슬라이드 문맥:
{json.dumps(shared_slide_context, ensure_ascii=False)}

입력:
{json.dumps(cases, ensure_ascii=False)}

모든 candidate_id와 모든 allowed_source_id에 대한 결과를 빠짐없이 반환하세요.
이유는 한 문장 이내로 짧게 작성하세요. JSON만 반환하세요:
{{
  "results": [
    {{
      "candidate_id": "I0001",
      "claim_verdict": "true | false | uncertain",
      "verdict_confidence": 0.0,
      "reason": "짧은 한국어 이유",
      "source_assessments": [
        {{
          "source_id": "I0001-S1",
          "relevance": "direct | partial | irrelevant",
          "relevance_confidence": 0.0,
          "quote": "문서 조각에서 그대로 복사한 문장 또는 빈 문자열",
          "source_class": "primary_authority | standard | scholarly | expert_reference | encyclopedia | official_secondary | general_secondary | user_generated | promotional | unknown",
          "authority_for_claim": "high | medium | low",
          "trust_confidence": 0.0,
          "relation": "supports_claim | contradicts_claim | irrelevant",
          "relation_confidence": 0.0,
          "reason": "짧은 한국어 이유"
        }}
      ]
    }}
  ]
}}
"""
    return prompt, source_lookup, candidate_ids


def _partition_pre_verifier_slide_entries(
    entries: list[dict[str, Any]],
    *,
    current_date: str,
) -> list[list[dict[str, Any]]]:
    """Split one slide's evidence by the actual assessment payload size."""
    max_candidates = _pre_verifier_batch_max_candidates()
    max_sources = _pre_verifier_batch_max_sources()
    max_prompt_chars = _pre_verifier_batch_max_prompt_chars()
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for entry in entries:
        trial = [*current, entry]
        prompt, source_lookup, candidate_ids = (
            _build_pre_verifier_batch_assessment_prompt(
                trial,
                current_date=current_date,
            )
        )
        exceeds_limit = bool(current) and (
            len(candidate_ids) > max_candidates
            or len(source_lookup) > max_sources
            or len(prompt) > max_prompt_chars
        )
        if exceeds_limit:
            batches.append(current)
            current = [entry]
        else:
            current = trial

    if current:
        batches.append(current)
    return batches


def _normalize_pre_verifier_batch_assessment(
    text: str,
    *,
    valid_candidate_ids: set[str],
    source_lookup: dict[str, tuple[str, dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], str]:
    try:
        payload = json.loads(_strip_json_fence(text or ""), strict=False)
    except Exception as exc:
        return {}, str(exc)
    rows = payload.get("results") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return {}, "batch assessment results is not a list"
    valid_source_classes = {
        "primary_authority",
        "standard",
        "scholarly",
        "expert_reference",
        "encyclopedia",
        "official_secondary",
        "general_secondary",
        "user_generated",
        "promotional",
        "unknown",
    }
    normalized: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        if (
            candidate_id not in valid_candidate_ids
            or candidate_id in normalized
        ):
            continue
        verdict = str(row.get("claim_verdict") or "").strip().lower()
        if verdict not in {"true", "false", "uncertain"}:
            verdict = "uncertain"
        source_rows = (
            row.get("source_assessments")
            if isinstance(row.get("source_assessments"), list)
            else []
        )
        assessments: dict[str, dict[str, Any]] = {}
        for source_row in source_rows:
            if not isinstance(source_row, dict):
                continue
            source_id = str(source_row.get("source_id") or "").strip()
            source_owner = source_lookup.get(source_id)
            if (
                source_owner is None
                or source_owner[0] != candidate_id
                or source_id in assessments
            ):
                continue
            relevance = str(source_row.get("relevance") or "").strip().lower()
            if relevance not in {"direct", "partial", "irrelevant"}:
                relevance = "irrelevant"
            source_class = str(
                source_row.get("source_class") or ""
            ).strip().lower()
            if source_class not in valid_source_classes:
                source_class = "unknown"
            authority = str(
                source_row.get("authority_for_claim") or ""
            ).strip().lower()
            if authority not in {"high", "medium", "low"}:
                authority = "low"
            relation = str(source_row.get("relation") or "").strip().lower()
            if relation not in {
                "supports_claim",
                "contradicts_claim",
                "irrelevant",
            }:
                relation = "irrelevant"
            assessments[source_id] = {
                "relevance": relevance,
                "relevance_confidence": round(
                    _clamp01(source_row.get("relevance_confidence"), 0.0),
                    4,
                ),
                "quote": _trim_text(source_row.get("quote") or "", 900),
                "source_class": source_class,
                "authority_for_claim": authority,
                "trust_confidence": round(
                    _clamp01(source_row.get("trust_confidence"), 0.0),
                    4,
                ),
                "relation": relation,
                "relation_confidence": round(
                    _clamp01(source_row.get("relation_confidence"), 0.0),
                    4,
                ),
                "reason": _trim_text(source_row.get("reason") or "", 220),
            }
        normalized[candidate_id] = {
            "claim_verdict": verdict,
            "verdict_confidence": round(
                _clamp01(row.get("verdict_confidence"), 0.0),
                4,
            ),
            "reason": _trim_text(row.get("reason") or "", 260),
            "source_assessments": assessments,
        }
    return normalized, ""


def _apply_pre_verifier_batch_assessment(
    entry: dict[str, Any],
    result: dict[str, Any],
    *,
    source_lookup: dict[str, tuple[str, dict[str, Any]]],
    model_spec: str,
    resolved_model: str,
) -> dict[str, Any]:
    issue = entry["issue"]
    material = entry["material"]
    payload = material["payload"]
    candidate_id = str(issue.get("candidate_id") or "")
    assessments = (
        result.get("source_assessments")
        if isinstance(result.get("source_assessments"), dict)
        else {}
    )
    excerpts: list[dict[str, Any]] = []
    partial_excerpts: list[dict[str, Any]] = []
    semantic_assessments: list[dict[str, Any]] = []
    relevance_counts: Counter[str] = Counter()
    strength_counts: Counter[str] = Counter()
    selected_passage_count = 0

    for source_id, (owner_id, source) in source_lookup.items():
        if owner_id != candidate_id:
            continue
        assessment = assessments.get(source_id)
        if not isinstance(assessment, dict):
            assessment = {
                "relevance": "irrelevant",
                "relevance_confidence": 0.0,
                "quote": "",
                "source_class": "unknown",
                "authority_for_claim": "low",
                "trust_confidence": 0.0,
                "relation": "irrelevant",
                "relation_confidence": 0.0,
                "reason": "배치 응답에 source 판정이 없습니다.",
            }
        relevance = str(assessment.get("relevance") or "irrelevant")
        relevance_confidence = _clamp01(
            assessment.get("relevance_confidence"),
            0.0,
        )
        relevance_eligible = bool(
            relevance == "direct"
            and relevance_confidence
            >= _pre_verifier_document_relevance_min_confidence()
        )
        source["document_relevance"] = relevance
        source["document_relevance_confidence"] = relevance_confidence
        source["document_relevance_reason"] = str(
            assessment.get("reason") or ""
        )
        source["document_relevance_eligible"] = relevance_eligible
        relevance_counts[relevance] += 1

        quote = str(assessment.get("quote") or "").strip()
        verified_passages = _verify_reported_passages(
            str(source.get("_source_text") or ""),
            [{
                "quote_or_paragraph": quote,
                "key_sentence": quote,
            }],
        ) if quote else []
        matched_passages = [
            {
                **passage,
                "selection_method": "batched_evidence_assessment",
            }
            for passage in verified_passages
            if _passage_match_usable(
                str(passage.get("match_status") or ""),
                passage.get("match_score"),
            )
        ]
        source["stage3_verified_passages"] = verified_passages
        source["matched_passages"] = matched_passages
        source["direct_match"] = bool(
            relevance_eligible and matched_passages
        )
        source["passage_extraction_status"] = (
            "selected" if source["direct_match"] else "no_usable_passage"
        )
        selected_passage_count += int(source["direct_match"])

        source_class = str(
            assessment.get("source_class") or "unknown"
        )
        authority = str(
            assessment.get("authority_for_claim") or "low"
        )
        trust_confidence = _clamp01(
            assessment.get("trust_confidence"),
            0.0,
        )
        source_class, authority, trust_confidence, trust_reason = (
            _apply_source_domain_policy(
                source,
                source_class,
                authority,
                trust_confidence,
                str(assessment.get("reason") or ""),
            )
        )
        strength = _source_strength_from_assessment(
            source_class,
            authority,
            trust_confidence,
        )
        source["assessed_source_class"] = source_class
        source["authority_for_claim"] = authority
        source["source_trust_confidence"] = trust_confidence
        source["source_trust_reason"] = trust_reason
        source["source_strength"] = strength
        source["source_trust_eligible"] = strength in {
            "strong",
            "supporting",
        }
        source["priority_eligible"] = bool(
            source["direct_match"] and source["source_trust_eligible"]
        )
        strength_counts[strength] += 1

        if (
            relevance == "partial"
            and matched_passages
            and source["source_trust_eligible"]
        ):
            selected = matched_passages[0]
            partial_excerpts.append({
                "source_id": source_id,
                "url": str(source.get("url") or ""),
                "domain": str(source.get("domain") or ""),
                "document_relevance": "partial",
                "source_strength": strength,
                "assessed_source_class": source_class,
                "source_priority": source.get("source_priority"),
                "source_priority_label": str(
                    source.get("source_priority_label") or ""
                ),
                "key_sentence": _evidence_passage_text(selected),
                "match_status": str(selected.get("match_status") or ""),
                "match_score": selected.get("match_score"),
            })

        relation = str(assessment.get("relation") or "irrelevant")
        relation_confidence = _clamp01(
            assessment.get("relation_confidence"),
            0.0,
        )
        if not (
            source["priority_eligible"]
            and relation != "irrelevant"
            and relation_confidence
            >= _pre_verifier_evidence_semantic_min_confidence()
        ):
            continue
        selected = matched_passages[0]
        excerpts.append({
            "source_id": source_id,
            "url": str(source.get("url") or ""),
            "domain": str(source.get("domain") or ""),
            "document_relevance": relevance,
            "source_strength": strength,
            "assessed_source_class": source_class,
            "source_priority": source.get("source_priority"),
            "source_priority_label": str(
                source.get("source_priority_label") or ""
            ),
            "key_sentence": _evidence_passage_text(selected),
            "match_status": str(selected.get("match_status") or ""),
            "match_score": selected.get("match_score"),
        })
        semantic_assessments.append({
            "source_id": source_id,
            "relation": relation,
            "confidence": relation_confidence,
            "reason": str(assessment.get("reason") or ""),
        })

    payload["document_relevance"] = {
        "status": "batched",
        "candidate_count": sum(relevance_counts.values()),
        "direct_count": relevance_counts.get("direct", 0),
        "partial_count": relevance_counts.get("partial", 0),
        "eligible_count": sum(
            1
            for source in payload.get("verified_sources", []) or []
            if isinstance(source, dict)
            and source.get("document_relevance_eligible")
        ),
    }
    payload["passage_extraction"] = {
        "status": "batched",
        "candidate_source_count": sum(relevance_counts.values()),
        "selected_source_count": selected_passage_count,
    }
    payload["source_trust_assessment"] = {
        "status": "batched",
        "candidate_count": sum(strength_counts.values()),
        "strong_count": strength_counts.get("strong", 0),
        "supporting_count": strength_counts.get("supporting", 0),
        "excluded_count": strength_counts.get("excluded", 0),
    }
    evidence = _compact_pre_verifier_evidence(
        issue,
        payload,
        model_spec=model_spec,
        resolved_model=resolved_model,
        excerpts=excerpts,
        semantic_assessments=semantic_assessments,
        semantic_metadata={
            "status": "batched",
            "model": model_spec,
            "resolved_model": resolved_model,
        },
    )
    if partial_excerpts:
        evidence["partial_evidence"] = partial_excerpts[:2]
    valid_relations = {
        str(row.get("relation_to_claim") or "")
        for row in evidence.get("evidence", []) or []
    }
    model_verdict = str(result.get("claim_verdict") or "uncertain")
    if not valid_relations:
        validated_verdict = "uncertain"
    elif len(valid_relations) > 1:
        validated_verdict = "uncertain"
    elif "contradicts_claim" in valid_relations:
        validated_verdict = "false"
    elif "supports_claim" in valid_relations:
        validated_verdict = "true"
    else:
        validated_verdict = "uncertain"
    if model_verdict != validated_verdict:
        validated_verdict = "uncertain"
    evidence["web_claim_verdict"] = validated_verdict
    evidence["web_verdict_confidence"] = (
        float(result.get("verdict_confidence") or 0.0)
        if validated_verdict != "uncertain"
        else 0.0
    )
    evidence["web_verdict_reason"] = str(result.get("reason") or "")
    evidence["batch_assessment"] = {
        "status": "ok",
        "candidate_id": candidate_id,
        "model_claim_verdict": model_verdict,
        "validated_claim_verdict": validated_verdict,
        "returned_source_count": len(assessments),
    }
    return evidence


def _call_pre_verifier_batch_assessment(
    entries: list[dict[str, Any]],
    *,
    current_date: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    prompt, source_lookup, candidate_ids = (
        _build_pre_verifier_batch_assessment_prompt(
            entries,
            current_date=current_date,
        )
    )
    if not candidate_ids:
        return {}, _empty_token_usage(), {
            "status": "not_run",
            "reason": "본문을 가져온 claim이 없습니다.",
            "candidate_ids": [],
        }
    model_spec = _pre_verifier_evidence_semantic_model()
    text, usage, resolved = _call_llm(
        model_spec=model_spec,
        prompt=prompt,
        max_tokens=_pre_verifier_batch_assessment_max_tokens(),
        stage="grounding",
    )
    results, parse_error = _normalize_pre_verifier_batch_assessment(
        text,
        valid_candidate_ids=set(candidate_ids),
        source_lookup=source_lookup,
    )
    evidence_by_candidate: dict[str, dict[str, Any]] = {}
    for entry in entries:
        issue = entry.get("issue") if isinstance(entry.get("issue"), dict) else {}
        candidate_id = str(issue.get("candidate_id") or "")
        result = results.get(candidate_id)
        if not isinstance(result, dict):
            continue
        evidence_by_candidate[candidate_id] = (
            _apply_pre_verifier_batch_assessment(
                entry,
                result,
                source_lookup=source_lookup,
                model_spec=model_spec,
                resolved_model=str(
                    resolved.get("resolved_model") or model_spec
                ),
            )
        )
    metadata = {
        "status": "parse_failed" if parse_error else "ok",
        "model": model_spec,
        "resolved_model": str(
            resolved.get("resolved_model") or model_spec
        ),
        "parse_error": parse_error,
        "candidate_ids": candidate_ids,
        "returned_candidate_ids": sorted(results),
        "missing_candidate_ids": sorted(set(candidate_ids) - set(results)),
        "source_count": len(source_lookup),
        "prompt_chars": len(prompt),
    }
    return evidence_by_candidate, usage, metadata


def _finalize_pre_verifier_evidence_material(
    issue: dict[str, Any],
    material: dict[str, Any],
    usage: dict[str, Any],
    *,
    model_spec: str,
    current_date: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    terminal = material.get("terminal_evidence")
    if isinstance(terminal, dict):
        return terminal, usage

    excerpts, assessments, assessment_usage, semantic_metadata = (
        _call_pre_verifier_semantic_assessment(
            issue,
            material["payload"],
            current_date=current_date,
        )
    )
    _merge_token_usage(usage, assessment_usage)
    usage["semantic_assessment_calls"] = 1 if excerpts else 0
    return (
        _compact_pre_verifier_evidence(
            issue,
            material["payload"],
            model_spec=model_spec,
            resolved_model=str(material.get("resolved_model") or model_spec),
            excerpts=excerpts,
            semantic_assessments=assessments,
            semantic_metadata=semantic_metadata,
        ),
        usage,
    )


def _retrieve_pre_verifier_evidence(
    issue: dict[str, Any],
    *,
    model_spec: str,
    current_date: str,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retrieve and semantically assess evidence for one unique claim."""
    material, usage = _retrieve_pre_verifier_evidence_material(
        issue,
        model_spec=model_spec,
        current_date=current_date,
        max_tokens=max_tokens,
    )
    return _finalize_pre_verifier_evidence_material(
        issue,
        material,
        usage,
        model_spec=model_spec,
        current_date=current_date,
    )


def _fan_out_shared_evidence(
    evidence: dict[str, Any],
    group: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    primary_id = str(group[0].get("candidate_id") or "") if group else ""
    rows: list[dict[str, Any]] = []
    for target in group:
        row = copy.deepcopy(evidence)
        row.update({
            "candidate_id": target.get("candidate_id", ""),
            "issue_id": target.get("issue_id", ""),
            "claim_id": target.get("claim_id", ""),
            "category": target.get("category", ""),
            "basis_code": target.get("basis_code", ""),
            "transcript_context_ids": target.get("transcript_context_ids", []),
            "transcript_context": target.get("transcript_context", ""),
            "slide_context": target.get("slide_context", {}),
            "reference_context_ids": target.get("reference_context_ids", []),
            "reference_context": target.get("reference_context", ""),
        })
        if str(target.get("candidate_id") or "") != primary_id:
            row["shared_retrieval_candidate_id"] = primary_id
        rows.append(row)
    return rows


def collect_pre_verifier_evidence(
    payload: dict[str, Any],
    *,
    input_path: str | Path,
    merged_clean_path: str | Path | None = None,
    current_date: str,
    max_workers: int = 20,
    max_tokens: int = 600,
) -> dict[str, Any]:
    """Retrieve compact native-search evidence for factual verifier candidates."""
    targets = _pre_verifier_evidence_targets(payload)
    _attach_pre_verifier_transcript_context(
        targets,
        merged_clean_path,
    )
    target_groups = _group_pre_verifier_targets(targets)
    retrieval_targets = [group[0] for group in target_groups]
    model_spec = _pre_verifier_evidence_model()
    if retrieval_targets and not model_spec:
        raise RuntimeError("웹 근거 수집에 사용할 모델을 grounding 단계에서 선택해야 합니다.")
    token_usage = _empty_token_usage()
    evidence_items: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    web_search_request_count = 0
    document_relevance_call_count = 0
    passage_extraction_call_count = 0
    source_trust_assessment_call_count = 0
    semantic_assessment_call_count = 0
    print(
        f"  pre-verifier web evidence 시작: targets={len(targets)}, "
        f"unique_searches={len(retrieval_targets)}, "
        f"model={model_spec}, workers={max_workers}",
        flush=True,
    )

    def worker(issue: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return _retrieve_pre_verifier_evidence(
            issue,
            model_spec=model_spec,
            current_date=current_date,
            max_tokens=max_tokens,
        )

    evidence_by_candidate: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(retrieval_targets)))) as executor:
        futures = {executor.submit(worker, issue): issue for issue in retrieval_targets}
        for future in as_completed(futures):
            issue = futures[future]
            candidate_id = str(issue.get("candidate_id") or "")
            try:
                evidence, usage = future.result()
                evidence_by_candidate[candidate_id] = evidence
                _merge_token_usage(token_usage, usage)
                web_search_request_count += int(usage.get("web_search_requests", 0) or 0)
                document_relevance_call_count += int(
                    usage.get("document_relevance_calls", 0) or 0
                )
                passage_extraction_call_count += int(
                    usage.get("passage_extraction_calls", 0) or 0
                )
                source_trust_assessment_call_count += int(
                    usage.get("source_trust_assessment_calls", 0) or 0
                )
                semantic_assessment_call_count += int(
                    usage.get("semantic_assessment_calls", 0) or 0
                )
                if evidence.get("status") == "grounding_unavailable":
                    errors.append({
                        "candidate_id": candidate_id,
                        "error": str(
                            evidence.get("error") or "grounding unavailable"
                        ),
                    })
            except Exception as exc:
                errors.append({
                    "candidate_id": candidate_id,
                    "error": str(exc),
                })
                evidence_by_candidate[candidate_id] = {
                    "candidate_id": issue.get("candidate_id", ""),
                    "issue_id": issue.get("issue_id", ""),
                    "claim_id": issue.get("claim_id", ""),
                    "category": issue.get("category", ""),
                    "basis_code": issue.get("basis_code", ""),
                    "status": "grounding_unavailable",
                    "evidence": [],
                    "model": model_spec,
                    "transcript_context_ids": issue.get("transcript_context_ids", []),
                    "transcript_context": issue.get("transcript_context", ""),
                    "error": str(exc),
                }

    for group in target_groups:
        candidate_id = str(group[0].get("candidate_id") or "")
        evidence = evidence_by_candidate.get(candidate_id)
        if not evidence:
            continue
        rows = _fan_out_shared_evidence(evidence, group)
        evidence_items.extend(rows)
        for row in rows:
            print(
                f"    evidence {row.get('candidate_id')}: {row.get('status')} "
                f"({len(row.get('evidence') or [])} sources)",
                flush=True,
            )

    token_usage["semantic_assessment_calls"] = semantic_assessment_call_count
    token_usage["web_search_requests"] = web_search_request_count
    token_usage["document_relevance_calls"] = document_relevance_call_count
    token_usage["passage_extraction_calls"] = passage_extraction_call_count
    token_usage["source_trust_assessment_calls"] = (
        source_trust_assessment_call_count
    )
    order = {str(issue.get("candidate_id") or ""): index for index, issue in enumerate(targets)}
    evidence_items.sort(key=lambda item: order.get(str(item.get("candidate_id") or ""), 10**9))
    status_counts = Counter(str(item.get("status") or "unknown") for item in evidence_items)
    relation_counts = Counter(
        str(evidence.get("relation_to_claim") or "unknown")
        for item in evidence_items
        for evidence in item.get("evidence", []) or []
        if isinstance(evidence, dict)
    )
    return {
        "schema_version": "classified_issue_evidence.v1",
        "stage": "pre_verifier_web_evidence",
        "source_input_path": str(input_path),
        "generated_at": _now_iso(),
        "current_date": current_date,
        "model": model_spec,
        "summary": {
            "target_count": len(targets),
            "unique_retrieval_count": len(retrieval_targets),
            "deduplicated_target_count": len(targets) - len(retrieval_targets),
            "verified_count": status_counts.get("verified", 0),
            "insufficient_evidence_count": status_counts.get("insufficient_evidence", 0),
            "grounding_unavailable_count": status_counts.get("grounding_unavailable", 0),
            "status_counts": dict(status_counts),
            "relation_counts": dict(relation_counts),
            "web_search_request_count": web_search_request_count,
            "document_relevance_assessment_count": sum(
                1
                for item in evidence_items
                if (item.get("document_relevance") or {}).get("status")
                in {"ok", "parse_failed"}
            ),
            "document_relevance_call_count": document_relevance_call_count,
            "source_trust_assessment_count": sum(
                1
                for item in evidence_items
                if (item.get("source_trust_assessment") or {}).get("status")
                in {"ok", "parse_failed"}
            ),
            "source_trust_assessment_call_count": (
                source_trust_assessment_call_count
            ),
            "semantic_assessment_count": sum(
                1
                for item in evidence_items
                if (item.get("semantic_assessment") or {}).get("status") in {"ok", "parse_failed"}
            ),
            "semantic_assessment_call_count": semantic_assessment_call_count,
            "retrieval_parse_recovered_count": sum(
                1
                for item in evidence_items
                if item.get("retrieval_parse_error") and item.get("status") != "grounding_unavailable"
            ),
        },
        "evidence_items": evidence_items,
        "errors": errors,
        "token_usage": token_usage,
    }


def collect_pre_verifier_evidence_batched(
    payload: dict[str, Any],
    *,
    input_path: str | Path,
    merged_clean_path: str | Path | None = None,
    current_date: str,
    max_workers: int = 20,
    max_tokens: int = 600,
    unique_claim_limit: int | None = None,
    progress_notify: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Retrieve claims independently, then assess fetched evidence per slide.

    Search remains claim-specific so the native tool records the actual query for
    each claim. Post-search relevance, quotation, trust, and claim-relation checks
    are grouped by slide. A missing/invalid slide-batch result falls back to the
    existing per-claim path for that claim only. Claims without fetched source
    text remain insufficient and continue without a redundant second search.
    """
    all_targets = _pre_verifier_evidence_targets(payload)
    _attach_pre_verifier_transcript_context(
        all_targets,
        merged_clean_path,
    )
    all_groups = _group_pre_verifier_targets(all_targets)
    limit = (
        max(1, int(unique_claim_limit))
        if unique_claim_limit is not None
        else None
    )
    target_groups = all_groups[:limit] if limit is not None else all_groups
    retrieval_targets = [group[0] for group in target_groups]
    selected_candidate_ids = {
        str(target.get("candidate_id") or "")
        for group in target_groups
        for target in group
    }
    targets = [
        target
        for target in all_targets
        if str(target.get("candidate_id") or "") in selected_candidate_ids
    ]
    model_spec = _pre_verifier_evidence_model()
    query_plan_model_spec = _pre_verifier_evidence_semantic_model()
    if retrieval_targets and (not model_spec or not query_plan_model_spec):
        raise RuntimeError("웹 근거 수집에 사용할 모델을 grounding 단계에서 선택해야 합니다.")
    stage_usage = {
        "query_planning": _empty_token_usage(),
        "web_search_and_fetch": _empty_token_usage(),
        "batch_assessment": _empty_token_usage(),
        "fallback": _empty_token_usage(),
    }
    errors: list[dict[str, str]] = []
    entries: list[dict[str, Any]] = []
    terminal_by_candidate: dict[str, dict[str, Any]] = {}
    retrieval_web_search_request_count = 0
    fallback_web_search_request_count = 0

    query_plan_batches: list[tuple[str, int, list[dict[str, Any]]]] = []
    query_targets_by_slide: dict[str, list[dict[str, Any]]] = {}
    for issue in retrieval_targets:
        location = (
            issue.get("location")
            if isinstance(issue.get("location"), dict)
            else {}
        )
        slide_number = location.get("slide_number")
        slide_key = (
            str(slide_number)
            if slide_number not in (None, "")
            else "unknown"
        )
        query_targets_by_slide.setdefault(slide_key, []).append(issue)
    for slide_key, slide_targets in query_targets_by_slide.items():
        slide_targets.sort(
            key=lambda issue: str(issue.get("candidate_id") or "")
        )
        chunk_size = _pre_verifier_batch_max_candidates()
        for start in range(0, len(slide_targets), chunk_size):
            query_plan_batches.append(
                (
                    slide_key,
                    (start // chunk_size) + 1,
                    slide_targets[start : start + chunk_size],
                )
            )

    query_plan_metadata: list[dict[str, Any]] = []

    def query_plan_worker(
        slide_key: str,
        batch_index: int,
        batch_targets: list[dict[str, Any]],
    ) -> tuple[
        str,
        int,
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, Any],
        dict[str, Any],
    ]:
        plans, usage, metadata = _call_pre_verifier_query_plan(
            batch_targets,
            model_spec=query_plan_model_spec,
            current_date=current_date,
        )
        return (
            slide_key,
            batch_index,
            batch_targets,
            plans,
            usage,
            metadata,
        )

    with ThreadPoolExecutor(
        max_workers=min(max_workers, max(1, len(query_plan_batches)))
    ) as executor:
        futures = {
            executor.submit(
                query_plan_worker,
                slide_key,
                batch_index,
                batch_targets,
            ): (slide_key, batch_index, batch_targets)
            for slide_key, batch_index, batch_targets in query_plan_batches
        }
        query_plan_done = 0
        query_plan_total = len(query_plan_batches)
        for future in as_completed(futures):
            slide_key, batch_index, batch_targets = futures[future]
            query_plan_done += 1
            if progress_notify:
                progress_notify(query_plan_done, query_plan_total)
            try:
                (
                    _,
                    _,
                    _,
                    plans,
                    usage,
                    metadata,
                ) = future.result()
                _merge_token_usage(stage_usage["query_planning"], usage)
                for issue in batch_targets:
                    candidate_id = str(issue.get("candidate_id") or "")
                    plan = plans.get(candidate_id) or {
                        "web_check": False,
                        "basis_code": "query_planner_unavailable",
                        "verification_question": "",
                    }
                    issue["web_check"] = plan.get("web_check") is True
                    issue["web_check_basis_code"] = str(
                        plan.get("basis_code") or ""
                    )
                    issue["verification_question"] = str(
                        plan.get("verification_question") or ""
                    )
                    issue["verification_question_ko"] = str(
                        plan.get("verification_question_ko") or ""
                    )
                    issue["verification_question_en"] = str(
                        plan.get("verification_question_en") or ""
                    )
                    issue["verification_questions"] = list(
                        plan.get("verification_questions") or []
                    )
                metadata = dict(metadata)
                metadata["slide_number"] = (
                    int(slide_key)
                    if slide_key.isdigit()
                    else slide_key
                )
                metadata["batch_index"] = batch_index
                query_plan_metadata.append(metadata)
            except Exception as exc:
                candidate_ids = [
                    str(issue.get("candidate_id") or "")
                    for issue in batch_targets
                ]
                for issue in batch_targets:
                    issue["web_check"] = False
                    issue[
                        "web_check_basis_code"
                    ] = "query_planner_unavailable"
                    issue["verification_question"] = ""
                    issue["verification_question_ko"] = ""
                    issue["verification_question_en"] = ""
                    issue["verification_questions"] = []
                errors.append({
                    "candidate_id": ",".join(candidate_ids),
                    "stage": "query_planning",
                    "error": str(exc),
                })
                query_plan_metadata.append({
                    "status": "failed",
                    "slide_number": (
                        int(slide_key)
                        if slide_key.isdigit()
                        else slide_key
                    ),
                    "batch_index": batch_index,
                    "candidate_ids": candidate_ids,
                    "returned_candidate_ids": [],
                    "missing_candidate_ids": candidate_ids,
                    "error": str(exc),
                })

    query_plan_metadata.sort(
        key=lambda row: (
            int(row.get("slide_number"))
            if str(row.get("slide_number") or "").isdigit()
            else 10**9,
            int(row.get("batch_index", 0) or 0),
        )
    )
    planned_search_count = sum(
        1 for issue in retrieval_targets if issue.get("web_check") is True
    )

    print(
        f"  batched pre-verifier web evidence 시작: targets={len(targets)}, "
        f"unique_searches={planned_search_count}/{len(retrieval_targets)}, "
        f"query_model={query_plan_model_spec}, search_model={model_spec}, "
        f"workers={max_workers}",
        flush=True,
    )

    def retrieval_worker(
        issue: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return _retrieve_pre_verifier_evidence_material(
            issue,
            model_spec=model_spec,
            current_date=current_date,
            max_tokens=max_tokens,
            run_individual_assessments=False,
        )

    with ThreadPoolExecutor(
        max_workers=min(max_workers, max(1, len(retrieval_targets)))
    ) as executor:
        futures = {
            executor.submit(retrieval_worker, issue): issue
            for issue in retrieval_targets
        }
        retrieval_done = 0
        retrieval_total = len(retrieval_targets)
        for future in as_completed(futures):
            issue = futures[future]
            retrieval_done += 1
            if progress_notify:
                progress_notify(retrieval_done, retrieval_total)
            candidate_id = str(issue.get("candidate_id") or "")
            try:
                material, usage = future.result()
                _merge_token_usage(
                    stage_usage["web_search_and_fetch"],
                    usage,
                )
                retrieval_web_search_request_count += int(
                    usage.get("web_search_requests", 0) or 0
                )
                terminal = material.get("terminal_evidence")
                if isinstance(terminal, dict):
                    terminal["search_result_urls"] = list(
                        usage.get("web_search_sources", []) or []
                    )
                    terminal["search_queries"] = list(
                        usage.get("web_search_queries", []) or []
                    )
                    terminal_by_candidate[candidate_id] = terminal
                    continue
                entries.append({
                    "issue": issue,
                    "material": material,
                    "retrieval_usage": usage,
                })
            except Exception as exc:
                errors.append({
                    "candidate_id": candidate_id,
                    "stage": "web_search_and_fetch",
                    "error": str(exc),
                })
                terminal_by_candidate[candidate_id] = {
                    "candidate_id": candidate_id,
                    "issue_id": issue.get("issue_id", ""),
                    "claim_id": issue.get("claim_id", ""),
                    "category": issue.get("category", ""),
                    "basis_code": issue.get("basis_code", ""),
                    "status": "grounding_unavailable",
                    "evidence": [],
                    "model": model_spec,
                    "search_queries": [],
                    "search_result_urls": [],
                    "error": str(exc),
                }

    entries_by_slide: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        issue = entry.get("issue") if isinstance(entry.get("issue"), dict) else {}
        location = (
            issue.get("location")
            if isinstance(issue.get("location"), dict)
            else {}
        )
        slide_number = location.get("slide_number")
        slide_key = str(slide_number) if slide_number not in (None, "") else "unknown"
        entries_by_slide.setdefault(slide_key, []).append(entry)
    for slide_entries in entries_by_slide.values():
        slide_entries.sort(
            key=lambda entry: str(
                (entry.get("issue") or {}).get("candidate_id") or ""
            )
        )

    assessment_batches: list[tuple[str, int, int, list[dict[str, Any]]]] = []
    for slide_key, slide_entries in entries_by_slide.items():
        slide_batches = _partition_pre_verifier_slide_entries(
            slide_entries,
            current_date=current_date,
        )
        for slide_batch_index, slide_batch_entries in enumerate(
            slide_batches,
            start=1,
        ):
            assessment_batches.append(
                (
                    slide_key,
                    slide_batch_index,
                    len(slide_batches),
                    slide_batch_entries,
                )
            )

    batch_evidence: dict[str, dict[str, Any]] = {}
    batch_metadata_rows: list[dict[str, Any]] = []
    batch_assessment_call_count = 0

    def batch_worker(
        slide_key: str,
        slide_batch_index: int,
        slide_batch_count: int,
        slide_entries: list[dict[str, Any]],
    ) -> tuple[
        str,
        int,
        int,
        dict[str, dict[str, Any]],
        dict[str, Any],
        dict[str, Any],
    ]:
        evidence, usage, metadata = _call_pre_verifier_batch_assessment(
            slide_entries,
            current_date=current_date,
        )
        return (
            slide_key,
            slide_batch_index,
            slide_batch_count,
            evidence,
            usage,
            metadata,
        )

    with ThreadPoolExecutor(
        max_workers=min(max_workers, max(1, len(assessment_batches)))
    ) as executor:
        futures = {
            executor.submit(
                batch_worker,
                slide_key,
                slide_batch_index,
                slide_batch_count,
                slide_entries,
            ): (
                slide_key,
                slide_batch_index,
                slide_batch_count,
                slide_entries,
            )
            for (
                slide_key,
                slide_batch_index,
                slide_batch_count,
                slide_entries,
            ) in assessment_batches
        }
        assessment_done = 0
        assessment_total = len(assessment_batches)
        for future in as_completed(futures):
            (
                slide_key,
                slide_batch_index,
                slide_batch_count,
                slide_entries,
            ) = futures[future]
            assessment_done += 1
            if progress_notify:
                progress_notify(assessment_done, assessment_total)
            try:
                (
                    _,
                    _,
                    _,
                    slide_evidence,
                    slide_usage,
                    slide_metadata,
                ) = future.result()
                batch_evidence.update(slide_evidence)
                _merge_token_usage(
                    stage_usage["batch_assessment"],
                    slide_usage,
                )
                if any(
                    int(slide_usage.get(field, 0) or 0)
                    for field in TOKEN_USAGE_FIELDS
                ):
                    batch_assessment_call_count += 1
                slide_metadata = dict(slide_metadata)
                slide_metadata["slide_number"] = (
                    int(slide_key)
                    if slide_key.isdigit()
                    else slide_key
                )
                slide_metadata["slide_batch_index"] = slide_batch_index
                slide_metadata["slide_batch_count"] = slide_batch_count
                batch_metadata_rows.append(slide_metadata)
            except Exception as exc:
                candidate_ids = [
                    str(entry.get("issue", {}).get("candidate_id") or "")
                    for entry in slide_entries
                    if isinstance(entry.get("issue"), dict)
                ]
                batch_metadata_rows.append({
                    "status": "failed",
                    "slide_number": (
                        int(slide_key)
                        if slide_key.isdigit()
                        else slide_key
                    ),
                    "slide_batch_index": slide_batch_index,
                    "slide_batch_count": slide_batch_count,
                    "candidate_ids": candidate_ids,
                    "returned_candidate_ids": [],
                    "missing_candidate_ids": candidate_ids,
                    "source_count": 0,
                    "prompt_chars": 0,
                    "error": str(exc),
                })
                errors.append({
                    "candidate_id": ",".join(candidate_ids),
                    "stage": "slide_batch_assessment",
                    "error": str(exc),
                })

    batch_metadata_rows.sort(
        key=lambda row: (
            int(row.get("slide_number"))
            if str(row.get("slide_number") or "").isdigit()
            else 10**9,
            int(row.get("slide_batch_index", 0) or 0),
        )
    )
    batch_candidate_ids = {
        str(candidate_id)
        for row in batch_metadata_rows
        for candidate_id in row.get("candidate_ids", []) or []
        if str(candidate_id or "")
    }
    batch_returned_candidate_ids = {
        str(candidate_id)
        for row in batch_metadata_rows
        for candidate_id in row.get("returned_candidate_ids", []) or []
        if str(candidate_id or "")
    }
    batch_metadata = {
        "status": (
            "ok"
            if all(row.get("status") in {"ok", "not_run"} for row in batch_metadata_rows)
            else "partial_failed"
        ),
        "batch_mode": "per_slide_dynamic",
        "batch_count": len(batch_metadata_rows),
        "assessment_call_count": batch_assessment_call_count,
        "max_candidates_per_batch": _pre_verifier_batch_max_candidates(),
        "max_sources_per_batch": _pre_verifier_batch_max_sources(),
        "max_prompt_chars_per_batch": (
            _pre_verifier_batch_max_prompt_chars()
        ),
        "candidate_ids": sorted(batch_candidate_ids),
        "returned_candidate_ids": sorted(batch_returned_candidate_ids),
        "missing_candidate_ids": sorted(
            batch_candidate_ids - batch_returned_candidate_ids
        ),
        "source_count": sum(
            int(row.get("source_count", 0) or 0)
            for row in batch_metadata_rows
        ),
        "prompt_chars": sum(
            int(row.get("prompt_chars", 0) or 0)
            for row in batch_metadata_rows
        ),
        "batches": batch_metadata_rows,
    }

    evidence_by_candidate = dict(terminal_by_candidate)
    fallback_candidate_ids: list[str] = []
    for entry in entries:
        issue = entry["issue"]
        candidate_id = str(issue.get("candidate_id") or "")
        retrieval_usage = entry["retrieval_usage"]
        evidence = batch_evidence.get(candidate_id)
        if evidence is None:
            material_payload = entry["material"].get("payload", {})
            has_fetched_source = any(
                isinstance(source, dict)
                and source.get("fetch_status") == "ok"
                and source.get("_source_text")
                for source in material_payload.get("verified_sources", []) or []
            )
            if not has_fetched_source:
                evidence = _compact_pre_verifier_evidence(
                    issue,
                    material_payload,
                    model_spec=model_spec,
                    resolved_model=str(
                        entry["material"].get("resolved_model") or model_spec
                    ),
                    excerpts=[],
                    semantic_assessments=[],
                    semantic_metadata={
                        "status": "not_run",
                        "reason": "본문을 확보한 웹 문서가 없어 배치 판정을 생략했습니다.",
                    },
                )
                evidence["batch_assessment"] = {
                    "status": "not_run_no_fetched_sources",
                    "reason": "웹 근거 없이 다음 verifier로 전달합니다.",
                }
            else:
                fallback_candidate_ids.append(candidate_id)
        if evidence is None:
            try:
                evidence, fallback_usage = _retrieve_pre_verifier_evidence(
                    issue,
                    model_spec=model_spec,
                    current_date=current_date,
                    max_tokens=max_tokens,
                )
                _merge_token_usage(
                    stage_usage["fallback"],
                    fallback_usage,
                )
                fallback_web_search_request_count += int(
                    fallback_usage.get("web_search_requests", 0) or 0
                )
                evidence["batch_assessment"] = {
                    "status": "fallback",
                    "reason": "배치 응답 누락 또는 파싱 실패로 기존 개별 경로를 사용했습니다.",
                }
                evidence["search_result_urls"] = list(
                    fallback_usage.get("web_search_sources", []) or []
                )
                if not evidence.get("search_queries"):
                    evidence["search_queries"] = list(
                        fallback_usage.get("web_search_queries", []) or []
                    )
            except Exception as exc:
                errors.append({
                    "candidate_id": candidate_id,
                    "stage": "fallback",
                    "error": str(exc),
                })
                evidence = {
                    "candidate_id": candidate_id,
                    "issue_id": issue.get("issue_id", ""),
                    "claim_id": issue.get("claim_id", ""),
                    "category": issue.get("category", ""),
                    "basis_code": issue.get("basis_code", ""),
                    "status": "grounding_unavailable",
                    "evidence": [],
                    "model": model_spec,
                    "error": str(exc),
                }
        if not evidence.get("search_result_urls"):
            evidence["search_result_urls"] = list(
                retrieval_usage.get("web_search_sources", []) or []
            )
        if not evidence.get("search_queries"):
            evidence["search_queries"] = list(
                retrieval_usage.get("web_search_queries", []) or []
            )
        if not evidence.get("fetched_candidate_urls"):
            material_payload = entry["material"].get("payload", {})
            evidence["fetched_candidate_urls"] = [
                str(source.get("url") or "")
                for source in material_payload.get("verified_sources", []) or []
                if isinstance(source, dict)
            ]
        evidence_by_candidate[candidate_id] = evidence

    evidence_items: list[dict[str, Any]] = []
    for group in target_groups:
        candidate_id = str(group[0].get("candidate_id") or "")
        evidence = evidence_by_candidate.get(candidate_id)
        if not isinstance(evidence, dict):
            continue
        evidence_items.extend(_fan_out_shared_evidence(evidence, group))

    order = {
        str(issue.get("candidate_id") or ""): index
        for index, issue in enumerate(targets)
    }
    evidence_items.sort(
        key=lambda item: order.get(
            str(item.get("candidate_id") or ""),
            10**9,
        )
    )
    total_usage = _empty_token_usage()
    for usage in stage_usage.values():
        _merge_token_usage(total_usage, usage)
    web_search_requests = (
        retrieval_web_search_request_count
        + fallback_web_search_request_count
    )
    status_counts = Counter(
        str(item.get("status") or "unknown")
        for item in evidence_items
    )
    relation_counts = Counter(
        str(evidence.get("relation_to_claim") or "unknown")
        for item in evidence_items
        for evidence in item.get("evidence", []) or []
        if isinstance(evidence, dict)
    )
    return {
        "schema_version": "classified_issue_evidence.v2",
        "stage": "pre_verifier_web_evidence",
        "source_input_path": str(input_path),
        "generated_at": _now_iso(),
        "current_date": current_date,
        "model": model_spec,
        "summary": {
            "available_target_count": len(all_targets),
            "available_unique_retrieval_count": len(all_groups),
            "target_count": len(targets),
            "unique_retrieval_count": len(retrieval_targets),
            "unique_claim_limit": limit,
            "query_plan_count": len(retrieval_targets),
            "query_plan_search_count": planned_search_count,
            "query_plan_skip_count": (
                len(retrieval_targets) - planned_search_count
            ),
            "query_plan_batch_count": len(query_plan_metadata),
            "verified_count": status_counts.get("verified", 0),
            "insufficient_evidence_count": status_counts.get(
                "insufficient_evidence",
                0,
            ),
            "grounding_unavailable_count": status_counts.get(
                "grounding_unavailable",
                0,
            ),
            "status_counts": dict(status_counts),
            "relation_counts": dict(relation_counts),
            "web_search_request_count": web_search_requests,
            "batch_assessment_call_count": batch_assessment_call_count,
            "slide_batch_count": len(batch_metadata_rows),
            "batch_candidate_count": len(
                batch_metadata.get("candidate_ids", []) or []
            ),
            "fallback_candidate_count": len(fallback_candidate_ids),
            "fallback_candidate_ids": fallback_candidate_ids,
        },
        "query_planning": {
            "batch_mode": "per_slide",
            "max_candidates_per_batch": (
                _pre_verifier_batch_max_candidates()
            ),
            "batches": query_plan_metadata,
        },
        "batch_assessment": batch_metadata,
        "evidence_items": evidence_items,
        "errors": errors,
        "token_usage_by_stage": stage_usage,
        "token_usage": total_usage,
    }


def _call_grounding_trial(
    *,
    model_spec: str,
    issue: dict[str, Any],
    current_date: str,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    resolved = _resolve_model_spec(model_spec)
    if resolved.get("provider") == "gemini":
        payload, usage = _call_gemini_search_grounding(
            model_spec=model_spec,
            issue=issue,
            current_date=current_date,
            max_tokens=max_tokens,
        )
    else:
        payload, usage = _call_text_grounding(
            model_spec=model_spec,
            issue=issue,
            current_date=current_date,
            max_tokens=max_tokens,
        )
    payload["model_spec"] = model_spec
    payload["provider"] = payload.get("provider") or resolved.get("provider", "")
    payload["resolved_model"] = payload.get("resolved_model") or resolved.get("resolved_model", model_spec)
    payload = _verify_payload_sources(issue, payload)
    payload, repair_usage = _call_source_repair_fallback(
        model_spec=model_spec,
        issue=issue,
        payload=payload,
        current_date=current_date,
        max_tokens=max_tokens,
    )
    _merge_token_usage(usage, repair_usage)
    payload, extraction_usage = _call_passage_extraction_fallback(
        model_spec=model_spec,
        issue=issue,
        payload=payload,
        max_tokens=max_tokens,
    )
    _merge_token_usage(usage, extraction_usage)
    recheck, recheck_usage = _call_evidence_recheck(
        model_spec=model_spec,
        issue=issue,
        payload=payload,
        current_date=current_date,
        max_tokens=max_tokens,
    )
    _merge_token_usage(usage, recheck_usage)
    if recheck:
        payload["evidence_recheck"] = recheck
        if not recheck.get("parse_error") and recheck.get("status") in {"supports_issue", "refutes_issue", "insufficient_evidence"}:
            payload["pre_evidence_recheck_status"] = payload.get("status")
            payload["status"] = recheck.get("status")
            payload["claim_verdict"] = recheck.get("claim_verdict", payload.get("claim_verdict", "uncertain"))
            payload["issue_supported"] = recheck.get("issue_supported")
            payload["reason"] = recheck.get("reason", payload.get("reason", ""))
            payload["evidence_summary"] = recheck.get("evidence_summary", payload.get("evidence_summary", ""))
    _strip_internal_source_text(payload)
    return payload, usage


def _eligible_sources_for_decision(trial: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in trial.get("verified_sources", []) or []:
        if not isinstance(source, dict):
            continue
        if not source.get("direct_match") or not source.get("auto_decision_eligible"):
            continue
        priority = source.get("source_priority")
        if not isinstance(priority, int):
            continue
        rows.append(source)
    return rows


def _trial_best_source_priority(trial: dict[str, Any]) -> int | None:
    priorities = [
        int(source["source_priority"])
        for source in _eligible_sources_for_decision(trial)
        if isinstance(source.get("source_priority"), int)
    ]
    return min(priorities) if priorities else None


def _trial_selected_sources(trial: dict[str, Any], priority: int) -> list[dict[str, Any]]:
    return [
        source for source in _eligible_sources_for_decision(trial)
        if source.get("source_priority") == priority
    ]


def _trim_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _selected_evidence_passages(selected_sources: list[dict[str, Any]], *, limit: int = 6) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source in selected_sources:
        if not isinstance(source, dict):
            continue
        url = str(source.get("url") or "").strip()
        passages = source.get("matched_passages") or source.get("verified_model_passages") or []
        if not isinstance(passages, list):
            continue
        for passage in passages:
            if not isinstance(passage, dict):
                continue
            sentence = (
                passage.get("key_sentence")
                or passage.get("quote_or_paragraph")
                or passage.get("matched_text")
                or ""
            )
            sentence = _trim_text(sentence, 500)
            if not sentence:
                continue
            key = (url, sentence)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "url": url,
                "key_sentence": sentence,
                "stance": _normalize_status(passage.get("stance") or passage.get("status") or ""),
                "why_relevant": _trim_text(passage.get("why_relevant") or "", 220),
                "match_status": passage.get("match_status", ""),
                "match_score": passage.get("match_score"),
            })
            if len(rows) >= limit:
                return rows
    return rows


def _compact_grounding_payload(
    *,
    status: str,
    claim_verdict: str,
    issue_supported: bool | None,
    reason: str,
    source_urls: list[str],
    selected_sources: list[dict[str, Any]],
    priority_trials: list[dict[str, Any]],
    status_counts: Counter[str],
    best_priority: int | None,
) -> dict[str, Any]:
    status = _normalize_status(status)
    base: dict[str, Any] = {"status": status}
    if status not in {"supports_issue", "refutes_issue"}:
        return base

    evidence_summary = " / ".join(
        str(trial.get("evidence_summary") or "").strip()
        for trial in priority_trials
        if str(trial.get("evidence_summary") or "").strip()
    )
    base.update({
        "reason": _trim_text(reason, 260),
        "evidence_sources": source_urls[:3],
        "evidence_passages": _selected_evidence_passages(selected_sources),
        "evidence_summary": _trim_text(evidence_summary, 800),
        "trial_status_counts": dict(status_counts),
        "selected_source_priority": best_priority,
        "selected_source_priority_label": SOURCE_PRIORITY_LABELS.get(best_priority, "") if best_priority else "",
        "selected_source_count": len(selected_sources),
        "source_verification_policy": "supports/refutes requires fetched URL text with directly matched claim passages",
    })
    return base


def _aggregate_grounding_trials(issue: dict[str, Any], trials: list[dict[str, Any]]) -> dict[str, Any]:
    verified_trials = [
        trial for trial in trials
        if trial.get("source_verification_status") == "verified"
        and trial.get("status") in {"supports_issue", "refutes_issue"}
        and _trial_best_source_priority(trial) is not None
    ]
    best_priority = min(
        (_trial_best_source_priority(trial) for trial in verified_trials),
        default=None,
    )
    priority_trials = [
        trial for trial in verified_trials
        if best_priority is not None and _trial_best_source_priority(trial) == best_priority
    ]
    support_trials = [trial for trial in priority_trials if trial.get("status") == "supports_issue"]
    refute_trials = [trial for trial in priority_trials if trial.get("status") == "refutes_issue"]
    source_urls: list[str] = []
    selected_sources: list[dict[str, Any]] = []
    if best_priority is not None:
        for trial in priority_trials:
            for source in _trial_selected_sources(trial, best_priority):
                selected_sources.append(source)
                url = str(source.get("url") or "")
                if url and url not in source_urls:
                    source_urls.append(url)

    if support_trials and refute_trials:
        status = "insufficient_evidence"
        claim_verdict = "uncertain"
        issue_supported = None
        reason = "모델별 웹 근거 재검증 결과가 충돌하여 보수적으로 근거 부족으로 처리했습니다."
    elif support_trials:
        status = "supports_issue"
        claim_verdict = "claim_false"
        issue_supported = True
        reason = support_trials[0].get("reason") or "검증된 웹 본문 근거가 issue를 지지합니다."
    elif refute_trials:
        status = "refutes_issue"
        claim_verdict = "claim_true"
        issue_supported = False
        reason = refute_trials[0].get("reason") or "검증된 웹 본문 근거가 issue를 반박합니다."
    else:
        status = "insufficient_evidence"
        claim_verdict = "uncertain"
        issue_supported = None
        reason = (
            "모델 응답의 URL을 직접 확인했지만, 자동판정에 허용된 출처 등급의 직접 본문 근거가 있는 "
            "supports/refutes 판단이 없습니다."
        )

    status_counts = Counter(_normalize_status(trial.get("status")) for trial in trials)
    return _compact_grounding_payload(
        status=status,
        claim_verdict=claim_verdict,
        issue_supported=issue_supported,
        reason=reason,
        source_urls=source_urls,
        selected_sources=selected_sources,
        priority_trials=priority_trials,
        status_counts=status_counts,
        best_priority=best_priority,
    )


def _call_grounding(issue: dict[str, Any], current_date: str, max_tokens: int) -> tuple[dict[str, Any], dict[str, int]]:
    models = _grounding_model_specs()
    token_usage = _empty_token_usage()
    trials: list[dict[str, Any]] = []
    for model_spec in models:
        try:
            payload, usage = _call_grounding_trial(
                model_spec=model_spec,
                issue=issue,
                current_date=current_date,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            payload = {
                "status": "grounding_unavailable",
                "claim_verdict": "uncertain",
                "issue_supported": None,
                "reason": f"grounding 실패: {exc}",
                "evidence_sources": [],
                "evidence_summary": "",
                "search_queries": [],
                "model_spec": model_spec,
                "model": model_spec,
                "provider": "",
                "search_mode": "unavailable",
                "source_verification_status": "no_sources",
                "verified_sources": [],
            }
            usage = _empty_token_usage()
        trials.append(payload)
        _merge_token_usage(token_usage, usage)
    return _aggregate_grounding_trials(issue, trials), token_usage


def _should_ground(issue: dict[str, Any], categories: set[str]) -> bool:
    if str(issue.get("category") or "").strip() not in categories:
        return False
    return True


def _clear_grounding_adjustment(issue: dict[str, Any]) -> None:
    for key in (
        "pre_grounding_final_severity_score",
        "pre_grounding_final_severity_percent",
        "pre_grounding_status",
        "web_grounding_adjustment",
        "rejected_by_web_grounding",
        "resurrected_by_web_grounding",
        "confirmed_by_web_grounding",
    ):
        issue.pop(key, None)


def _apply_grounding_decision(issue: dict[str, Any], payload: dict[str, Any]) -> None:
    _clear_grounding_adjustment(issue)
    if payload.get("status") not in {"refutes_issue", "supports_issue"}:
        return
    original_score = _clamp01(issue.get("final_severity_score"))
    status = str(payload.get("status") or "")
    delta = _supports_issue_delta() if status == "supports_issue" else _refutes_issue_delta()
    adjusted_score = _clamp01(original_score + delta)
    issue["pre_grounding_final_severity_score"] = original_score
    issue["pre_grounding_final_severity_percent"] = round(original_score * 100.0, 2)
    issue["pre_grounding_status"] = _status_from_score(original_score)
    issue["final_severity_score"] = adjusted_score
    issue["final_severity_percent"] = round(adjusted_score * 100.0, 2)
    issue["needs_manual_review"] = _status_from_score(adjusted_score) == "professor_check"
    issue["web_grounding_adjustment"] = {
        "mode": "soft_delta",
        "status": status,
        "delta": round(adjusted_score - original_score, 6),
        "configured_delta": delta,
        "pre_score": original_score,
        "post_score": adjusted_score,
        "pre_status": issue["pre_grounding_status"],
        "post_status": _status_from_score(adjusted_score),
    }


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
    adjustments = [
        issue.get("web_grounding_adjustment")
        for issue in issues
        if isinstance(issue, dict) and isinstance(issue.get("web_grounding_adjustment"), dict)
    ]
    summary["web_grounding_adjusted_count"] = len(adjustments)
    summary["web_grounding_supports_adjusted_count"] = sum(
        1 for adjustment in adjustments if adjustment.get("status") == "supports_issue"
    )
    summary["web_grounding_refutes_adjusted_count"] = sum(
        1 for adjustment in adjustments if adjustment.get("status") == "refutes_issue"
    )
    summary["web_grounding_total_score_delta"] = round(
        sum(_safe_float(adjustment.get("delta")) for adjustment in adjustments),
        6,
    )
    summary.pop("web_grounding_rejected_count", None)
    summary.pop("web_grounding_resurrected_count", None)


def ground_classified_issues(
    verifier_result: dict[str, Any],
    *,
    current_date: str,
    max_workers: int = 20,
    max_tokens: int = 2048,
    categories: set[str] | None = None,
) -> dict[str, Any]:
    categories = categories or set(GROUNDABLE_CATEGORIES)
    issues = verifier_result.get("all_issues", []) or []
    verifier_result.pop("issues_by_type", None)
    targets = [issue for issue in issues if isinstance(issue, dict) and _should_ground(issue, categories)]
    token_usage = _empty_token_usage()
    status_counts: Counter[str] = Counter()

    if not targets:
        verifier_result["grounding"] = {
            "enabled": True,
            "grounded_issue_count": 0,
            "categories": sorted(categories),
            "models": _grounding_model_specs(),
            "status_counts": {},
            "token_usage": token_usage,
        }
        return verifier_result

    models = _grounding_model_specs()
    print(
        f"  classified issue grounding 시작: {len(targets)}건 ({', '.join(sorted(categories))}), "
        f"workers_per_model={max_workers}",
        flush=True,
    )

    # 모델별로 독립된 pool을 사용한다. 따라서 여러 grounding 모델을 켜도
    # ``max_workers``가 전체 한도가 아니라 모델당 동시 요청 한도가 된다.
    trials_by_id: dict[str, list[dict[str, Any]]] = {}

    def worker(model_spec: str, issue: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, int]]:
        issue_id = str(issue.get("id") or issue.get("issue_id") or "")
        try:
            payload, usage = _call_grounding_trial(
                model_spec=model_spec,
                issue=issue,
                current_date=current_date,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            payload = {
                "status": "grounding_unavailable",
                "claim_verdict": "uncertain",
                "issue_supported": None,
                "reason": f"grounding 실패: {exc}",
                "evidence_sources": [],
                "evidence_summary": "",
                "search_queries": [],
                "model_spec": model_spec,
                "model": model_spec,
                "provider": "",
                "search_mode": "unavailable",
                "source_verification_status": "no_sources",
                "verified_sources": [],
            }
            usage = _empty_token_usage()
        return issue_id, payload, usage

    def run_model(model_spec: str) -> list[tuple[str, dict[str, Any], dict[str, int]]]:
        results: list[tuple[str, dict[str, Any], dict[str, int]]] = []
        with ThreadPoolExecutor(max_workers=min(max(1, max_workers), len(targets))) as executor:
            futures = [executor.submit(worker, model_spec, issue) for issue in targets]
            for future in as_completed(futures):
                results.append(future.result())
        return results

    with ThreadPoolExecutor(max_workers=max(1, len(models))) as model_executor:
        futures = [model_executor.submit(run_model, model_spec) for model_spec in models]
        for future in as_completed(futures):
            for issue_id, payload, usage in future.result():
                trials_by_id.setdefault(issue_id, []).append(payload)
                _merge_token_usage(token_usage, usage)

    by_id: dict[str, dict[str, Any]] = {}
    target_by_id = {str(issue.get("id") or issue.get("issue_id") or ""): issue for issue in targets}
    for issue_id, issue in target_by_id.items():
        payload = _aggregate_grounding_trials(issue, trials_by_id.get(issue_id, []))
        by_id[issue_id] = payload
        status_counts[_normalize_status(payload.get("status"))] += 1
        print(f"    grounding {issue_id}: {payload.get('status')}", flush=True)

    for issue in issues:
        if not isinstance(issue, dict):
            continue
        _clear_grounding_adjustment(issue)
        issue_id = str(issue.get("id") or issue.get("issue_id") or "")
        payload = by_id.get(issue_id)
        if payload:
            issue["web_grounding"] = payload
            _apply_grounding_decision(issue, payload)
        else:
            if str(issue.get("category") or "").strip() not in categories:
                reason = "factual_error/temporal_error가 아니어서 web grounding을 실행하지 않음"
            else:
                reason = "web grounding 대상 조건에 맞지 않아 실행하지 않음"
            issue["web_grounding"] = {"status": "not_applicable"}

    verifier_result["grounding"] = {
        "enabled": True,
        "grounded_issue_count": len(targets),
        "categories": sorted(categories),
        "models": models,
        "status_counts": dict(status_counts),
        "token_usage": token_usage,
        "target_policy": "all factual_error/temporal_error candidates, including final rejected candidates",
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
    parser.add_argument("--max-workers", type=int, default=int(os.getenv("CLASSIFIED_ISSUE_GROUNDING_MAX_WORKERS", "20")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("CLASSIFIED_ISSUE_GROUNDING_MAX_TOKENS", "2048")))
    parser.add_argument(
        "--models",
        default="",
        help="comma/space-separated grounding models. Defaults to CLASSIFIED_ISSUE_GROUNDING_MODELS or gpt.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.verifier_json)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_grounded.json")
    if args.models:
        os.environ["CLASSIFIED_ISSUE_GROUNDING_MODELS"] = args.models
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
