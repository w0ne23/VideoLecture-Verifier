"""Web grounding for classified issue verifier results.

This stage runs after ``classified_issue_verifier`` and checks externally
verifiable factual_error and temporal_error issues with Gemini Search.
It only grounds surfaced verifier candidates and can lower refuted candidates
below the rejected threshold.
"""

from __future__ import annotations

import argparse
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
from typing import Any
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
SOURCE_PRIORITY_POLICY = (
    "official_docs > standards/government > academic > educational > encyclopedia; "
    "tutorial/blog/forum sources are fetched and logged but excluded from automatic decisions"
)
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


def _resurrected_score() -> float:
    configured = os.getenv("CLASSIFIED_ISSUE_GROUNDING_RESURRECT_SCORE")
    if configured is not None and str(configured).strip():
        return _clamp01(configured, _rejected_threshold() + 0.001)
    return min(_confirmed_threshold() - 0.001, _rejected_threshold() + 0.001)


def _status_from_score(score: float) -> str:
    if score >= _confirmed_threshold():
        return "confirmed"
    if score <= _rejected_threshold():
        return "rejected"
    return "professor_check"


def _grounding_model_specs() -> list[str]:
    configured = (
        _split_csv(os.getenv("CLASSIFIED_ISSUE_GROUNDING_MODELS"))
        or _split_csv(os.getenv("VERIFIER_GROUNDING_MODELS"))
        or _split_csv(os.getenv("CLASSIFIED_ISSUE_GROUNDING_MODEL"))
        or _split_csv(os.getenv("VERIFIER_GROUNDING_MODEL"))
    )
    return configured or ["gemini", "gpt", "claude"]


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
Record the exact search queries you used or would use to find the evidence.
Use both Korean and English search queries when the lecture claim is Korean. Include at least one high-priority source query and one fallback query for Wikipedia or educational/reference material.
Record bilingual match terms that connect the Korean lecture claim to likely English source wording.
For each source URL, include the most relevant sentence or paragraph you relied on. Prefer short passages, but keep enough surrounding context to avoid misleading quote fragments.

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
SEARCH_QUERIES=query1 | query2
MATCH_TERMS=Korean term | English term | synonym
CLAIM_VERDICT=claim_true | claim_false | uncertain
STATUS=supports_issue | refutes_issue | insufficient_evidence | grounding_unavailable
REASON=one or two Korean sentences explaining the web-grounded judgment
SOURCES=URL1, URL2
EVIDENCE_PASSAGES=[{{"url":"URL1","quote_or_paragraph":"source passage","key_sentence":"most important sentence","stance":"supports_issue | refutes_issue | unclear","why_relevant":"why this passage matters"}}]
SUMMARY=short Korean summary of the evidence

Consistency requirements:
- If claim_verdict is claim_false, issue_supported must be true and status must be supports_issue.
- If claim_verdict is claim_true, issue_supported must be false and status must be refutes_issue.
- If claim_verdict is uncertain, issue_supported must be null and status must be insufficient_evidence.
"""


def _build_grounding_json_prompt(issue: dict[str, Any], current_date: str) -> str:
    prompt = _build_grounding_prompt(issue, current_date)
    json_contract = """Return JSON only:
{
  "search_queries": ["query1", "query2"],
  "match_terms": ["Korean term", "English term", "synonym"],
  "claim_verdict": "claim_true | claim_false | uncertain",
  "status": "supports_issue | refutes_issue | insufficient_evidence | grounding_unavailable",
  "issue_supported": true,
  "reason": "one or two Korean sentences explaining the web-grounded judgment",
  "evidence_sources": ["URL1", "URL2"],
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
    educational_domains = (
        "britannica.com",
        "khanacademy.org",
        "openstax.org",
        "opentextbooks.org",
        "pressbooks.pub",
    )
    tutorial_domains = (
        "tutorialspoint.",
        "w3schools.",
        "geeksforgeeks.",
        "freecodecamp.",
        "javatpoint.",
        "programiz.",
    )
    forum_domains = ("reddit.", "stackoverflow.", "stackexchange.")
    blog_domains = ("blog", "medium.", "tistory.", "velog.", "substack.")
    if any(token in domain for token in tutorial_domains):
        trust_level, score = "tutorial", 0.20
    elif any(token in domain for token in forum_domains):
        trust_level, score = "forum", 0.20
    elif any(token in domain for token in blog_domains):
        trust_level, score = "blog", 0.20
    elif "wikipedia.org" in domain:
        trust_level, score = "encyclopedia", 0.70
    elif any(token in domain for token in standards_domains):
        trust_level, score = "standards", 0.95
    elif domain.endswith((".gov", ".go.kr", ".gov.kr")):
        trust_level, score = "government", 0.95
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
    trust = _source_trust(url)
    row = {
        "url": url,
        **trust,
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
    for term in _claim_terms(issue) + [
        str(value).strip()
        for value in payload.get("match_terms", []) or []
        if str(value).strip()
    ]:
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
        candidate = quote or key_sentence
        if not candidate:
            row = dict(passage)
            row.update({"match_status": "not_found", "match_score": 0.0, "matched_text": ""})
            verified.append(row)
            continue
        matched_text, score, status = _best_fuzzy_match(candidate, text)
        if status == "not_found" and key_sentence and key_sentence != candidate:
            matched_text, score, status = _best_fuzzy_match(key_sentence, text)
        row = dict(passage)
        row.update({
            "match_status": status,
            "match_score": score,
            "matched_text": matched_text,
            "selection_method": "model_reported_passage",
        })
        verified.append(row)
    return verified


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
        if passage.get("match_status") in {"exact", "fuzzy"}
    ]
    fallback_passages = [] if matched_model_passages else _match_passages(text, terms)
    for passage in fallback_passages:
        passage["selection_method"] = "keyword_fallback"
        passage["match_status"] = "keyword_fallback"
    row["matched_passages"] = matched_model_passages or fallback_passages
    row["direct_match"] = bool(row["matched_passages"])
    row["priority_eligible"] = bool(row["direct_match"] and row.get("auto_decision_eligible"))
    return row


def _verify_payload_sources(issue: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    sources = payload.get("evidence_sources") if isinstance(payload.get("evidence_sources"), list) else []
    reported_passages = payload.get("evidence_passages") if isinstance(payload.get("evidence_passages"), list) else []
    terms = _verification_terms(issue, payload)
    verified_sources = []
    for url in sources[:_max_sources_per_trial()]:
        verified_sources.append(_verify_source_url(str(url), reported_passages, terms))

    matched_sources = [row for row in verified_sources if row.get("direct_match")]
    if matched_sources:
        verification_status = "verified"
    elif verified_sources:
        verification_status = "no_direct_passage"
    else:
        verification_status = "no_sources"

    payload["verified_sources"] = verified_sources
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
    text, usage, resolved = _call_llm(model_spec=model_spec, prompt=prompt, max_tokens=max(512, min(max_tokens, 1200)))
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


def _build_passage_extraction_prompt(issue: dict[str, Any], payload: dict[str, Any]) -> str:
    terms = _verification_terms(issue, payload)
    sources = []
    for source in payload.get("verified_sources", []) or []:
        if not isinstance(source, dict):
            continue
        if not source.get("auto_decision_eligible"):
            continue
        if _has_verified_model_passage(source):
            continue
        if source.get("fetch_status") != "ok":
            continue
        text = str(source.get("_source_text") or "")
        if not text:
            continue
        sources.append({
            "url": source.get("url", ""),
            "domain": source.get("domain", ""),
            "trust_level": source.get("trust_level", ""),
            "source_priority": source.get("source_priority"),
            "text_excerpt": _source_text_sample(text, terms),
        })
    return f"""You are extracting evidence passages from fetched web source text.
Use only the provided source excerpts. Do not browse, infer from memory, or invent quotes.
The lecture claim may be Korean while source text may be English; consider translation and synonyms.
Return only passages that directly help judge whether the claim is true or false. If none are directly relevant, return an empty list.

Resolved claim: {issue.get("resolved_claim", "")}
Original claim_text: {issue.get("claim_text", "")}
Match terms and synonyms: {json.dumps(terms, ensure_ascii=False)}

Sources:
{json.dumps(sources, ensure_ascii=False, indent=2)}

Return JSON only:
{{
  "evidence_passages": [
    {{
      "url": "source URL",
      "quote_or_paragraph": "exact sentence or short paragraph copied from the provided excerpt",
      "key_sentence": "the single most important sentence copied from the provided excerpt",
      "stance": "supports_issue | refutes_issue | unclear",
      "why_relevant": "brief Korean explanation"
    }}
  ]
}}
"""


def _has_verified_model_passage(source: dict[str, Any]) -> bool:
    return any(
        isinstance(passage, dict) and passage.get("match_status") in {"exact", "fuzzy"}
        for passage in source.get("verified_model_passages", []) or []
    )


def _call_passage_extraction_fallback(
    *,
    model_spec: str,
    issue: dict[str, Any],
    payload: dict[str, Any],
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    if not _passage_extraction_enabled():
        return payload, _empty_token_usage()
    candidates = [
        source for source in payload.get("verified_sources", []) or []
        if isinstance(source, dict)
        and source.get("auto_decision_eligible")
        and not _has_verified_model_passage(source)
        and source.get("fetch_status") == "ok"
        and source.get("_source_text")
    ]
    if not candidates:
        return payload, _empty_token_usage()

    prompt = _build_passage_extraction_prompt(issue, payload)
    text, usage, resolved = _call_llm(model_spec=model_spec, prompt=prompt, max_tokens=max(512, min(max_tokens, 1200)))
    extraction: dict[str, Any] = {
        "provider": resolved.get("provider", ""),
        "model": model_spec,
        "resolved_model": resolved.get("resolved_model", model_spec),
        "candidate_source_count": len(candidates),
        "added_passage_count": 0,
    }
    try:
        data = json.loads(_strip_json_fence(text or ""), strict=False)
        extracted_passages = _normalize_evidence_passages(data.get("evidence_passages", []))
    except Exception as exc:
        extraction["parse_error"] = str(exc)
        payload["passage_extraction"] = extraction
        return payload, usage

    added = 0
    for source in candidates:
        source_text = str(source.get("_source_text") or "")
        source_passages = _reported_passages_for_url(extracted_passages, str(source.get("url") or ""))
        verified = _verify_reported_passages(source_text, source_passages)
        matched = [
            {**passage, "selection_method": "llm_passage_extraction"}
            for passage in verified
            if passage.get("match_status") in {"exact", "fuzzy"}
        ]
        if not matched:
            continue
        source["verified_model_passages"] = (source.get("verified_model_passages") or []) + verified
        source["matched_passages"] = matched + (source.get("matched_passages") or [])
        source["direct_match"] = True
        source["priority_eligible"] = bool(source.get("auto_decision_eligible"))
        added += len(matched)
    extraction["added_passage_count"] = added
    extraction["extracted_passage_count"] = len(extracted_passages)
    payload["passage_extraction"] = extraction
    _refresh_source_verification_status(payload)
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
    text, usage, resolved = _call_llm(model_spec=model_spec, prompt=prompt, max_tokens=max(512, min(max_tokens, 1200)))
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
            if url and url not in sources:
                sources.append(url)

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
                evidence_passages.append({
                    "id": f"G{index}:{chunk_index}",
                    "url": url,
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
    cfg_kwargs = {
        "temperature": 0.0,
        "max_output_tokens": max_tokens,
        "response_mime_type": "application/json",
        "tools": [types.Tool(google_search=types.GoogleSearch())],
        "thinking_config": types.ThinkingConfig(thinking_budget=0),
    }
    last_exc: Exception | None = None
    for index, (client_name, client) in enumerate(get_gemini_client_sequence()):
        try:
            def call_api():
                try:
                    return client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(**cfg_kwargs),
                    )
                except Exception as exc:
                    message = str(exc).lower()
                    if "response_mime_type" not in message and "json" not in message:
                        raise
                    fallback_kwargs = dict(cfg_kwargs)
                    fallback_kwargs.pop("response_mime_type", None)
                    return client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(**fallback_kwargs),
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
    text, usage, resolved = _call_llm(model_spec=model_spec, prompt=prompt, max_tokens=max_tokens)
    payload = _parse_response(text or "")
    payload["model"] = model_spec
    payload["resolved_model"] = resolved.get("resolved_model", model_spec)
    payload["provider"] = resolved.get("provider", "")
    payload["search_mode"] = "model_reported_sources_no_native_search_tool"
    return payload, usage


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


def _source_priority_diagnostics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    excluded_counts = Counter()
    for trial in trials:
        for source in trial.get("verified_sources", []) or []:
            if not isinstance(source, dict):
                continue
            level = str(source.get("trust_level") or "unknown")
            if source.get("priority_eligible"):
                counts[level] += 1
            elif source.get("direct_match"):
                excluded_counts[level] += 1
    return {
        "eligible_direct_source_counts": dict(counts),
        "excluded_direct_source_counts": dict(excluded_counts),
    }


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
    priority_diagnostics = _source_priority_diagnostics(trials)
    return {
        "status": status,
        "claim_verdict": claim_verdict,
        "issue_supported": issue_supported,
        "reason": reason,
        "evidence_sources": source_urls,
        "evidence_summary": " / ".join(
            str(trial.get("evidence_summary") or "").strip()
            for trial in priority_trials
            if str(trial.get("evidence_summary") or "").strip()
        )[:1200],
        "search_queries": {
            str(trial.get("model_spec") or trial.get("model") or ""): trial.get("search_queries", [])
            for trial in trials
        },
        "trials": trials,
        "trial_status_counts": dict(status_counts),
        "selected_source_priority": best_priority,
        "selected_source_priority_label": SOURCE_PRIORITY_LABELS.get(best_priority, "") if best_priority else "",
        "selected_source_count": len(selected_sources),
        "source_priority_diagnostics": priority_diagnostics,
        "source_verification_policy": "supports/refutes requires fetched URL text with directly matched claim passages",
        "source_priority_policy": SOURCE_PRIORITY_POLICY,
        "excluded_source_levels": sorted(EXCLUDED_SOURCE_LEVELS),
    }


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


def _apply_grounding_decision(issue: dict[str, Any], payload: dict[str, Any]) -> None:
    if payload.get("status") not in {"refutes_issue", "supports_issue"}:
        return
    original_score = _clamp01(issue.get("final_severity_score"))
    issue["pre_grounding_final_severity_score"] = original_score
    issue["pre_grounding_final_severity_percent"] = round(original_score * 100.0, 2)
    issue["pre_grounding_status"] = _status_from_score(original_score)
    if payload.get("status") == "refutes_issue":
        rejected_score = max(0.0, _rejected_threshold() - 0.001)
        issue["final_severity_score"] = min(original_score, rejected_score)
        issue["final_severity_percent"] = round(float(issue["final_severity_score"]) * 100.0, 2)
        issue["needs_manual_review"] = False
        issue["rejected_by_web_grounding"] = True
        return

    if original_score <= _rejected_threshold():
        issue["final_severity_score"] = max(original_score, _resurrected_score())
        issue["final_severity_percent"] = round(float(issue["final_severity_score"]) * 100.0, 2)
        issue["needs_manual_review"] = _status_from_score(float(issue["final_severity_score"])) == "professor_check"
        issue["resurrected_by_web_grounding"] = True
    else:
        issue["confirmed_by_web_grounding"] = True


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
    summary["web_grounding_resurrected_count"] = sum(
        1 for issue in issues if isinstance(issue, dict) and bool(issue.get("resurrected_by_web_grounding"))
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
            "models": _grounding_model_specs(),
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
                reason = "web grounding 대상 조건에 맞지 않아 실행하지 않음"
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
        "models": _grounding_model_specs(),
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
    parser.add_argument("--max-workers", type=int, default=int(os.getenv("CLASSIFIED_ISSUE_GROUNDING_MAX_WORKERS", "3")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("CLASSIFIED_ISSUE_GROUNDING_MAX_TOKENS", "2048")))
    parser.add_argument(
        "--models",
        default="",
        help="comma/space-separated grounding models. Defaults to CLASSIFIED_ISSUE_GROUNDING_MODELS or gemini,gpt,claude.",
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
