"""Claim verification pipeline orchestration."""

from __future__ import annotations

import json
from datetime import datetime

from . import claim_common as cc


def prepare_verification(merged_path: str, current_date: str = None):
    """Load merged_clean.json and build shared classified verifier context."""
    if current_date is None:
        current_date = datetime.now().strftime("%Y-%m-%d")
    with open(merged_path, "r", encoding="utf-8") as f:
        merged = json.load(f)
    slides = merged.get("slides", [])
    domain, sub_domain = cc._resolve_domain_fields(merged)
    hint = cc._get_domain_hint(domain, sub_domain)
    contexts = cc._collect_contexts(slides)
    slide_ctx = cc._build_slide_context_map(slides)
    return {
        "merged": merged,
        "slides": slides,
        "domain": domain,
        "sub_domain": sub_domain,
        "hint": hint,
        "contexts": contexts,
        "slide_ctx": slide_ctx,
        "current_date": current_date,
    }


def extract_claims_only(
    contexts: list[dict], current_date: str, hint: dict,
    slide_ctx: dict, batch_size: int | None = None, max_workers: int | None = None,
) -> tuple[list[tuple], int, dict]:
    """Run claim extraction for the classified issue pipeline."""
    from pipeline.verifier.claim_extractor import extract_claims_only as _run
    return _run(contexts, current_date, hint, slide_ctx, batch_size=batch_size, max_workers=max_workers)


def judge_issue_candidates_only(
    all_claims_by_batch: list[tuple], current_date: str, hint: dict,
    slide_ctx: dict, *, min_confidence: float, log_prefix: str = "",
) -> tuple[list[dict], list[dict], int, dict]:
    """Run the classified pipeline's first issue candidate judge."""
    from pipeline.verifier.issue_detector import judge_issue_candidates_only as _run
    return _run(
        all_claims_by_batch,
        current_date,
        hint,
        slide_ctx,
        min_confidence=min_confidence,
        log_prefix=log_prefix,
    )
