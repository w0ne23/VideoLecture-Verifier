"""merged_clean.json 입력 기준 verifier 실행기."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable

from . import claim_common as cv
from .claim_pipeline import prepare_verification
from .verifier_utils import (
    _ROOT,
    _collect_env_vars,
    _empty_token_usage,
    _merge_token_usage,
    _setup_worker,
    _write_claims_jsonl,
)


STATUS_CONFIRMED = "confirmed"
STATUS_PROFESSOR_CHECK = "professor_check"
STATUS_REJECTED = "rejected"
_DOCKER_LOG_TEE_ENABLED = False
_LLM_LOG_RULE = "═" * 72
_LLM_LOG_SUBRULE = "─" * 72


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    if seconds < 60:
        return f"{seconds:.1f}초"
    minutes = int(seconds // 60)
    remainder = seconds - (minutes * 60)
    return f"{minutes}분 {remainder:.1f}초"


def _format_breakdown(values: dict | None) -> str:
    if not isinstance(values, dict) or not values:
        return "없음"
    return ", ".join(f"{key} {value}건" for key, value in values.items())


def _print_log_rows(rows: list[tuple[str, object]] | None) -> None:
    for label, value in rows or []:
        print(f"    • {label}: {value}", flush=True)


def _llm_pipeline_banner(
    *,
    merged_file: Path,
    output_dir: Path,
    duration: object,
    slide_count: int,
    context_count: int,
) -> None:
    print(f"\n{_LLM_LOG_RULE}", flush=True)
    print("  🤖 LLM 검증 파이프라인 시작", flush=True)
    print(_LLM_LOG_RULE, flush=True)
    _print_log_rows([
        ("입력", merged_file),
        ("출력 폴더", output_dir),
        ("강의 길이", duration or "알 수 없음"),
        ("슬라이드", f"{slide_count}개"),
        ("Context", f"{context_count}개"),
    ])


def _llm_stage_start(
    stage_code: str,
    title: str,
    rows: list[tuple[str, object]] | None = None,
) -> float:
    print(f"\n{_LLM_LOG_RULE}", flush=True)
    print(f"  {stage_code} {title}", flush=True)
    print(_LLM_LOG_RULE, flush=True)
    _print_log_rows(rows)
    return time.perf_counter()


def _llm_stage_done(
    stage_code: str,
    title: str,
    started_at: float,
    *,
    rows: list[tuple[str, object]] | None = None,
    files: list[Path | str] | None = None,
) -> float:
    elapsed = time.perf_counter() - started_at
    print(f"\n  ✅ {stage_code} {title} 완료", flush=True)
    if rows:
        print("  📊 결과", flush=True)
        _print_log_rows(rows)
    valid_files = [str(path) for path in (files or []) if path]
    if valid_files:
        print("  📁 생성 파일", flush=True)
        for path in valid_files:
            print(f"    • {path}", flush=True)
    print(f"  ⏱ 처리 시간: {_format_elapsed(elapsed)}", flush=True)
    print(_LLM_LOG_SUBRULE, flush=True)
    return elapsed


def _llm_stage_failed(stage_code: str, title: str, started_at: float, exc: Exception) -> None:
    elapsed = time.perf_counter() - started_at
    print(f"\n  ❌ {stage_code} {title} 실패", flush=True)
    _print_log_rows([
        ("오류", f"{type(exc).__name__}: {exc}"),
        ("실패까지 경과", _format_elapsed(elapsed)),
    ])
    print(_LLM_LOG_SUBRULE, flush=True)


def _llm_stage_skipped(stage_code: str, title: str, reason: str) -> None:
    print(f"\n{_LLM_LOG_RULE}", flush=True)
    print(f"  {stage_code} {title}", flush=True)
    print(_LLM_LOG_RULE, flush=True)
    print(f"  ⏭ 건너뜀: {reason}", flush=True)
    print(_LLM_LOG_SUBRULE, flush=True)


def _llm_pipeline_done(
    *,
    started_at: float,
    timings: dict[str, float],
    rows: list[tuple[str, object]],
    files: list[Path | str],
) -> float:
    elapsed = time.perf_counter() - started_at
    print(f"\n{_LLM_LOG_RULE}", flush=True)
    print("  ✅ LLM 검증 파이프라인 완료", flush=True)
    print(_LLM_LOG_RULE, flush=True)
    print("  📊 최종 결과", flush=True)
    _print_log_rows(rows)
    print("  ⏱ 단계별 처리 시간", flush=True)
    for label, seconds in timings.items():
        print(f"    ✓ {label}: {_format_elapsed(seconds)}", flush=True)
    print(f"    ✓ 전체: {_format_elapsed(elapsed)}", flush=True)
    print("  📁 생성 파일", flush=True)
    for path in files:
        if path:
            print(f"    • {path}", flush=True)
    print(_LLM_LOG_RULE, flush=True)
    return elapsed


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _env_float(name: str, default: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        value = float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _clamp01(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _json_file_exists(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _load_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


DEFAULT_ISSUE_JUDGE_MAX_WORKERS = _env_int("ISSUE_JUDGE_MAX_WORKERS", 20)
CLAIM_EXTRACT_BATCH_SIZE = _env_int(
    "VERIFIER_CLAIM_EXTRACT_BATCH_SIZE",
    _env_int("VERIFIER_BATCH_SIZE", 4),
)
ISSUE_DETECTOR_BATCH_SIZE = _env_int("VERIFIER_ISSUE_DETECTOR_BATCH_SIZE", 4)
ISSUE_TYPE_CLASSIFIER_BATCH_SIZE = _env_int("VERIFIER_ISSUE_CLASSIFIER_BATCH_SIZE", 20)
CLASSIFIED_ISSUE_VERIFIER_BATCH_SIZE = _env_int(
    "VERIFIER_CROSSCHECK_MAX_ISSUES_PER_BATCH",
    _env_int("CLASSIFIED_ISSUE_VERIFIER_BATCH_SIZE", 5),
)


class _DockerLogTee:
    def __init__(self, primary, docker_stream):
        self.primary = primary
        self.docker_stream = docker_stream
        self.encoding = getattr(primary, "encoding", None) or "utf-8"
        self.errors = getattr(primary, "errors", None) or "replace"

    def write(self, data):
        written = self.primary.write(data)
        self.primary.flush()
        self.docker_stream.write(data)
        self.docker_stream.flush()
        return written

    def flush(self):
        self.primary.flush()
        self.docker_stream.flush()

    def fileno(self):
        return self.primary.fileno()

    def isatty(self):
        return self.primary.isatty()

    def __getattr__(self, name):
        return getattr(self.primary, name)


def _same_stream_target(stream, target_path: str) -> bool:
    try:
        return os.fstat(stream.fileno()) == os.stat(target_path)
    except Exception:
        return False


def _enable_docker_log_tee() -> None:
    """
    Docker 백그라운드 verifier는 stdout이 파일로 리다이렉트된다.
    파일 로그는 유지하면서 Docker Desktop/backend logs에도 같은 내용을 흘려보낸다.
    """
    global _DOCKER_LOG_TEE_ENABLED
    if _DOCKER_LOG_TEE_ENABLED:
        return

    flag = os.getenv("VERIFIER_DOCKER_LOGS", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return

    target_path = os.getenv("VERIFIER_DOCKER_LOG_TARGET", "/proc/1/fd/1")
    if not Path(target_path).exists():
        return
    if not (Path("/.dockerenv").exists() or Path("/pipeline").exists()):
        return
    if _same_stream_target(sys.stdout, target_path) and _same_stream_target(sys.stderr, target_path):
        return

    try:
        docker_stream = open(target_path, "a", encoding="utf-8", errors="replace", buffering=1)
    except OSError:
        return

    if not _same_stream_target(sys.stdout, target_path):
        sys.stdout = _DockerLogTee(sys.stdout, docker_stream)
    if not _same_stream_target(sys.stderr, target_path):
        sys.stderr = _DockerLogTee(sys.stderr, docker_stream)
    _DOCKER_LOG_TEE_ENABLED = True


def _base_stem(merged_path: Path) -> str:
    return merged_path.stem.replace("_merged_clean", "").replace("_merged", "")


def _split_model_specs(value: str | None) -> list[str]:
    if not value:
        return []
    return [part for part in re.split(r"[\s,]+", str(value).strip()) if part]


def _default_issue_judge_models() -> list[str]:
    configured = (
        _split_model_specs(os.getenv("ISSUE_JUDGE_MODELS"))
        or _split_model_specs(os.getenv("VERIFIER_ISSUE_JUDGE_MODELS"))
    )
    return configured or ["gpt-5.4", "claude-sonnet-5", "grok-4.5"]


def _is_openai_model(model: str) -> bool:
    return str(model or "").lower().startswith(("gpt", "o1", "o3"))


def _is_xai_model(model: str) -> bool:
    return cv._is_xai_model(model)


def _is_deepseek_model(model: str) -> bool:
    return cv._is_deepseek_model(model)


def _is_anthropic_model(model: str) -> bool:
    lowered = str(model or "").lower()
    return lowered.startswith("claude") or "sonnet" in lowered or "opus" in lowered or "haiku" in lowered


def _issue_judge_min_confidence_for_model(
    model: str,
) -> float:
    """Return the first-pass issue detector threshold for a judge model."""
    def _bounded(value: float) -> float:
        return max(0.0, min(1.0, value))

    model_key = re.sub(r"[^0-9A-Za-z]+", "_", str(model or "").strip()).strip("_").upper()
    env_candidates = []
    if model_key:
        env_candidates.append(f"VERIFIER_ISSUE_JUDGE_MIN_CONFIDENCE_{model_key}")
    if _is_anthropic_model(model):
        env_candidates.extend(
            [
                "VERIFIER_ISSUE_JUDGE_MIN_CONFIDENCE_CLAUDE",
                "VERIFIER_ISSUE_JUDGE_MIN_CONFIDENCE_ANTHROPIC",
            ]
        )
        default = 0.60
    elif _is_openai_model(model):
        env_candidates.extend(
            [
                "VERIFIER_ISSUE_JUDGE_MIN_CONFIDENCE_GPT",
                "VERIFIER_ISSUE_JUDGE_MIN_CONFIDENCE_OPENAI",
            ]
        )
        default = 0.8
    else:
        default = 0.8

    for key in env_candidates:
        raw = os.getenv(key)
        if raw is None:
            continue
        try:
            return _bounded(float(str(raw).strip()))
        except ValueError:
            continue
    return _bounded(default)


def _issue_judge_single_model_keep_confidence() -> float:
    return _env_float("VERIFIER_ISSUE_JUDGE_SINGLE_MODEL_KEEP_CONFIDENCE", 0.85)


def _issue_judge_score_lookup(
    *,
    models: list[str],
    judge_results: dict[str, dict],
) -> dict[str, dict[str, float]]:
    scores_by_claim: dict[str, dict[str, float]] = {}
    for model in models:
        result = judge_results.get(model, {}) or {}
        if result.get("ok") is False:
            continue
        for score in result.get("claim_scores", []) or []:
            if not isinstance(score, dict):
                continue
            claim_id = str(score.get("claim_id", "") or "").strip()
            if not claim_id:
                continue
            scores_by_claim.setdefault(claim_id, {})[model] = _clamp01(score.get("confidence", 0))
    return scores_by_claim


def _issue_judge_consensus_decision(
    claim_id: str,
    *,
    issue_models: list[str],
    evaluated_models: list[str],
    scores_by_claim: dict[str, dict[str, float]],
    single_keep_confidence: float | None = None,
) -> dict:
    """Resolve detector votes before downstream verification.

    Two or more positive model votes always pass. A single-model candidate
    passes only when that model reaches the strong-keep threshold.
    """
    single_keep_confidence = (
        _issue_judge_single_model_keep_confidence()
        if single_keep_confidence is None
        else single_keep_confidence
    )
    claim_scores = scores_by_claim.get(claim_id, {}) or {}
    issue_model_count = len(issue_models)
    evaluated_count = len(evaluated_models)
    if evaluated_count == 0:
        return {"keep": False, "status": "all_models_failed"}
    if issue_model_count == 0:
        return {"keep": False, "status": "no_issue"}
    if issue_model_count >= 2:
        status = "all_models_agreed" if issue_model_count == evaluated_count else "partial_agreement"
        return {"keep": True, "status": status}

    positive_model = issue_models[0]
    positive_confidence = _clamp01(claim_scores.get(positive_model, 0.0))
    if positive_confidence >= single_keep_confidence:
        return {
            "keep": True,
            "status": "single_model_strong",
            "positive_model": positive_model,
            "positive_confidence": round(positive_confidence, 6),
        }
    return {
        "keep": False,
        "status": "rejected_single_model_low_confidence",
        "rejection": {
            "claim_id": claim_id,
            "reason": "single_model_below_strong_keep_confidence",
            "positive_model": positive_model,
            "positive_confidence": round(positive_confidence, 6),
            "strong_keep_confidence": round(single_keep_confidence, 6),
        },
    }


def _missing_provider_key(model: str) -> str | None:
    if _is_openai_model(model) and not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY"
    if _is_xai_model(model) and not os.getenv("XAI_API_KEY"):
        return "XAI_API_KEY"
    if _is_deepseek_model(model) and not os.getenv("DEEPSEEK_API_KEY"):
        return "DEEPSEEK_API_KEY"
    if _is_anthropic_model(model) and not os.getenv("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY"
    return None


def _rebuild_classified_claim_batches(claims: list[dict], contexts: list[dict], batch_size: int) -> list[dict]:
    """Group extracted claims with the context batches used by classified issue judging."""
    batches = [contexts[i:i + batch_size] for i in range(0, len(contexts), batch_size)]
    batch_by_context_id: dict[str, int] = {}
    batch_claims: dict[int, tuple[list[dict], list[dict]]] = {}

    for batch in batches:
        batch_id = id(batch)
        batch_claims[batch_id] = (batch, [])
        for context in batch:
            context_id = str(context.get("context_id", "") or "")
            if context_id:
                batch_by_context_id[context_id] = batch_id

    for claim in claims:
        context_id = str(claim.get("context_id", "") or "")
        batch_id = batch_by_context_id.get(context_id)
        if batch_id in batch_claims:
            batch_claims[batch_id][1].append(claim)

    return [
        {"batch": batch, "claims": grouped_claims}
        for batch, grouped_claims in batch_claims.values()
        if grouped_claims
    ]


def _classified_issue_judge_worker(args_tuple):
    """Worker used by the classified issue pipeline's first issue judge."""
    started_at = time.perf_counter()
    try:
        (
            merged_path,
            model,
            claims_serialized,
            current_date,
            root,
            env_vars,
        ) = args_tuple
        _setup_worker(root, env_vars, model)
        min_confidence = _issue_judge_min_confidence_for_model(model)
        from pipeline.verifier.claim_pipeline import prepare_verification as _prepare_verification
        from pipeline.verifier.issue_detector import judge_issue_candidates_only

        ctx = _prepare_verification(merged_path, current_date=current_date)
        print(f"\n  [{model}] 1차 issue judge 시작 (min_confidence={min_confidence:.2f})", flush=True)

        claims_by_batch = [(item["batch"], item["claims"]) for item in claims_serialized]
        issues, claim_scores, api_calls, token_usage = judge_issue_candidates_only(
            claims_by_batch,
            ctx["current_date"],
            ctx["hint"],
            ctx["slide_ctx"],
            min_confidence=min_confidence,
            log_prefix=model,
        )

        print(f"  [{model}] 1차 issue judge 완료: {len(issues)}건", flush=True)
        return {
            "model": model,
            "ok": True,
            "issues": issues,
            "claim_scores": claim_scores,
            "api_calls": api_calls,
            "token_usage": token_usage,
            "elapsed_sec": round(time.perf_counter() - started_at, 6),
        }
    except Exception as e:
        import traceback

        model = args_tuple[1] if len(args_tuple) > 1 else "unknown"
        detail = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise RuntimeError(
            f"[{model}] classified issue judge worker failed: "
            f"{type(e).__name__}: {e}\n{detail}"
        ) from e


def _load_claims_jsonl(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []

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


def _model_file_slug(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(model or "").strip()).strip("-")
    return slug or "model"


def _issue_judge_payload(
    *,
    model: str,
    claims_path: str,
    merged_path: Path,
    input_claim_count: int,
    result: dict,
) -> dict:
    issues = []
    for index, issue in enumerate(result.get("issues", []) or [], start=1):
        row = dict(issue)
        row["issue_id"] = f"I{index:04d}"
        ordered = {
            "issue_id": row.get("issue_id", ""),
            "claim_id": row.get("claim_id", ""),
            "resolved_claim": row.get("resolved_claim", ""),
            "claim_text": row.get("claim_text", ""),
            "issue": row.get("issue", ""),
            "basis_code": row.get("basis_code", ""),
            "confidence": row.get("confidence", 0),
            "context_id": row.get("context_id", ""),
            "context_ids": row.get("context_ids", []),
            "slide_number": row.get("slide_number"),
            "start_time": row.get("start_time"),
            "end_time": row.get("end_time"),
        }
        ordered.update({
            key: value
            for key, value in row.items()
            if key not in ordered and key not in {"context_id", "type", "issue_type", "claim_type"}
        })
        issues.append(ordered)

    claim_scores = []
    for score in result.get("claim_scores", []) or []:
        if not isinstance(score, dict):
            continue
        claim_scores.append({
            "claim_id": score.get("claim_id", ""),
            "context_id": score.get("context_id", ""),
            "resolved_claim": score.get("resolved_claim", ""),
            "claim_text": score.get("claim_text", ""),
            "claim_type": score.get("claim_type", ""),
            "basis_code": score.get("basis_code", ""),
            "confidence": score.get("confidence", 0),
        })

    ok = bool(result.get("ok", True))
    summary = {
        "input_claim_count": input_claim_count,
        "scored_claim_count": len(claim_scores),
        "issue_count": len(issues),
        "api_calls": int(result.get("api_calls", 0) or 0),
        "elapsed_sec": round(float(result.get("elapsed_sec", 0.0) or 0.0), 6),
        "status": "ok" if ok else "failed",
    }
    if not ok and result.get("error"):
        summary["error"] = str(result.get("error"))

    return {
        "schema_version": "issue_judge_model.v1",
        "stage": "claim_to_issue_judge",
        "model": model,
        "merged_path": str(merged_path),
        "source_claims_path": claims_path,
        "summary": summary,
        "claim_scores": claim_scores,
        "issues": issues,
        "token_usage": result.get("token_usage", _empty_token_usage()),
    }


def _write_issue_judge_model_outputs(
    *,
    output_dir: Path,
    base_stem: str,
    merged_path: Path,
    claims_path: str,
    claims: list[dict],
    judge_results: dict[str, dict],
) -> dict[str, str]:
    paths = {}
    for model, result in judge_results.items():
        payload = _issue_judge_payload(
            model=model,
            claims_path=claims_path,
            merged_path=merged_path,
            input_claim_count=len(claims),
            result=result,
        )
        path = output_dir / f"{base_stem}_issue_judge_{_model_file_slug(model)}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        paths[model] = str(path)
        result["claim_scores"] = payload["claim_scores"]
        result["issues"] = payload["issues"]
    return paths


def _build_issue_judge_comparison(
    *,
    models: list[str],
    claims: list[dict],
    judge_results: dict[str, dict],
    issue_judge_paths: dict[str, str],
    claims_path: str,
) -> dict:
    by_claim = []
    exclusive_by_model = {model: [] for model in models}
    failed_models = [
        model
        for model in models
        if judge_results.get(model, {}).get("ok") is False
    ]
    evaluated_models = [model for model in models if model not in failed_models]
    issue_counts = {model: len(judge_results.get(model, {}).get("issues", []) or []) for model in models}
    issues_by_model_claim: dict[str, dict[str, list[dict]]] = {}
    scores_by_claim = _issue_judge_score_lookup(models=models, judge_results=judge_results)

    for model in models:
        grouped: dict[str, list[dict]] = {}
        for issue in judge_results.get(model, {}).get("issues", []) or []:
            claim_id = str(issue.get("claim_id", "") or "")
            if claim_id:
                grouped.setdefault(claim_id, []).append(issue)
        issues_by_model_claim[model] = grouped

    all_model_agreed_count = 0
    single_model_only_count = 0
    no_issue_claim_count = 0
    disagreement_count = 0
    rejected_single_model_count = 0
    union_issue_claim_ids = set()

    for claim in claims:
        claim_id = str(claim.get("claim_id") or _claim_key(claim))
        model_rows = {}
        issue_models = []
        for model in models:
            if model in failed_models:
                model_rows[model] = {
                    "status": "failed",
                    "has_issue": None,
                    "error": str(judge_results.get(model, {}).get("error", "") or ""),
                }
                continue
            model_issues = issues_by_model_claim.get(model, {}).get(claim_id, [])
            if model_issues:
                issue_models.append(model)
                model_rows[model] = {
                    "status": "ok",
                    "has_issue": True,
                    "issue_count": len(model_issues),
                    "issues": [
                        {
                            "issue_id": issue.get("issue_id", ""),
                            "issue": issue.get("issue", ""),
                            "basis_code": issue.get("basis_code", ""),
                            "confidence": issue.get("confidence", 0),
                        }
                        for issue in model_issues
                    ],
                }
            else:
                model_rows[model] = {"status": "ok", "has_issue": False}

        evaluated_count = len(evaluated_models)
        decision = _issue_judge_consensus_decision(
            claim_id,
            issue_models=issue_models,
            evaluated_models=evaluated_models,
            scores_by_claim=scores_by_claim,
        )
        status = str(decision["status"])
        consensus_rejection = decision.get("rejection")
        if status == "all_models_failed":
            pass
        elif status == "no_issue":
            no_issue_claim_count += 1
        elif status == "all_models_agreed":
            all_model_agreed_count += 1
            union_issue_claim_ids.add(claim_id)
        elif status == "single_model_strong":
            single_model_only_count += 1
            union_issue_claim_ids.add(claim_id)
            exclusive_by_model[issue_models[0]].append(claim_id)
        elif status == "partial_agreement":
            disagreement_count += 1
            union_issue_claim_ids.add(claim_id)
        else:
            rejected_single_model_count += 1

        by_claim.append({
            "claim_id": claim_id,
            "resolved_claim": claim.get("resolved_claim", ""),
            "claim_text": claim.get("claim_text", ""),
            "context_id": claim.get("context_id", ""),
            "context_ids": claim.get("context_ids", []),
            "models": model_rows,
            "agreement": {
                "status": status,
                "issue_model_count": len(issue_models),
                "issue_models": issue_models,
                "single_model_rejection": consensus_rejection or {},
            },
        })

    return {
        "schema_version": "issue_judge_comparison.v1",
        "stage": "claim_to_issue_judge",
        "models": models,
        "source_claims_path": claims_path,
        "issue_judge_result_paths": issue_judge_paths,
        "summary": {
            "input_claim_count": len(claims),
            "evaluated_model_count": len(evaluated_models),
            "failed_models": failed_models,
            "union_issue_claim_count": len(union_issue_claim_ids),
            "issue_counts_by_model": issue_counts,
            "all_models_agreed_count": all_model_agreed_count,
            "partial_agreement_count": disagreement_count,
            "single_model_only_count": single_model_only_count,
            "rejected_single_model_low_confidence_count": rejected_single_model_count,
            "single_model_strong_keep_confidence": _issue_judge_single_model_keep_confidence(),
            "no_issue_claim_count": no_issue_claim_count,
        },
        "exclusive_by_model": exclusive_by_model,
        "by_claim": by_claim,
    }


def _write_issue_judge_merged_output(
    *,
    output_dir: Path,
    base_stem: str,
    merged_path: Path,
    claims_path: str,
    models: list[str],
    judge_results: dict[str, dict],
) -> tuple[str, dict]:
    merged_issues: list[dict] = []
    seen_by_claim: dict[str, dict] = {}
    duplicate_claim_ids: list[str] = []
    skipped_without_claim_id = 0
    scores_by_claim = _issue_judge_score_lookup(models=models, judge_results=judge_results)
    rejected_single_model: dict[str, dict] = {}
    failed_models = [
        model for model in models
        if (judge_results.get(model, {}) or {}).get("ok") is False
    ]
    evaluated_models = [model for model in models if model not in failed_models]
    issue_models_by_claim: dict[str, list[str]] = {}
    for model in evaluated_models:
        for issue in (judge_results.get(model, {}) or {}).get("issues", []) or []:
            if not isinstance(issue, dict):
                continue
            claim_id = str(issue.get("claim_id", "") or "").strip()
            if claim_id and model not in issue_models_by_claim.setdefault(claim_id, []):
                issue_models_by_claim[claim_id].append(model)
    decisions_by_claim = {
        claim_id: _issue_judge_consensus_decision(
            claim_id,
            issue_models=issue_models,
            evaluated_models=evaluated_models,
            scores_by_claim=scores_by_claim,
        )
        for claim_id, issue_models in issue_models_by_claim.items()
    }

    for model in models:
        result = judge_results.get(model, {}) or {}
        if result.get("ok") is False:
            continue
        for issue in result.get("issues", []) or []:
            if not isinstance(issue, dict):
                continue
            claim_id = str(issue.get("claim_id", "") or "").strip()
            if not claim_id:
                skipped_without_claim_id += 1
                continue
            decision = decisions_by_claim.get(claim_id, {})
            if not decision.get("keep", False):
                rejection = decision.get("rejection") or {
                    "claim_id": claim_id,
                    "reason": str(decision.get("status", "rejected")),
                }
                rejected_single_model.setdefault(claim_id, rejection)
                continue

            source_summary = {
                "model": model,
                "issue_id": issue.get("issue_id", ""),
                "issue": issue.get("issue", ""),
                "basis_code": issue.get("basis_code", ""),
                "confidence": issue.get("confidence", 0),
            }
            if claim_id in seen_by_claim:
                existing = seen_by_claim[claim_id]
                existing.setdefault("detected_by_models", [])
                if model not in existing["detected_by_models"]:
                    existing["detected_by_models"].append(model)
                existing.setdefault("source_model_issues", []).append(source_summary)
                duplicate_claim_ids.append(claim_id)
                try:
                    new_conf = float(issue.get("confidence", 0) or 0)
                    old_conf = float(existing.get("confidence", 0) or 0)
                except Exception:
                    new_conf = old_conf = 0.0
                if new_conf > old_conf:
                    for key in ("issue", "basis_code", "confidence"):
                        existing[key] = issue.get(key, existing.get(key))
                    existing["representative_model"] = model
                continue

            row = dict(issue)
            row["detected_by_models"] = [model]
            row["representative_model"] = model
            row["source_model_issues"] = [source_summary]
            row["detector_consensus"] = decision
            seen_by_claim[claim_id] = row
            merged_issues.append(row)

    for index, issue in enumerate(merged_issues, start=1):
        issue["issue_id"] = f"I{index:04d}"

    model_issue_counts = {
        model: len((judge_results.get(model, {}) or {}).get("issues", []) or [])
        for model in models
    }
    summary = {
        "input_model_count": len(models),
        "failed_models": failed_models,
        "model_issue_counts": model_issue_counts,
        "merged_issue_count": len(merged_issues),
        "dedupe_key": "claim_id",
        "duplicate_claim_count": len(set(duplicate_claim_ids)),
        "skipped_without_claim_id": skipped_without_claim_id,
        "rejected_single_model_low_confidence_count": len(rejected_single_model),
        "single_model_strong_keep_confidence": _issue_judge_single_model_keep_confidence(),
    }
    payload = {
        "schema_version": "issue_judge_merged.v1",
        "stage": "claim_to_issue_judge_merged",
        "merged_path": str(merged_path),
        "source_claims_path": claims_path,
        "models": models,
        "dedupe_key": "claim_id",
        "summary": summary,
        "issues": merged_issues,
        "rejected_single_model_low_confidence": list(rejected_single_model.values()),
    }
    path = output_dir / f"{base_stem}_issue_judge.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path), payload


def _claim_key(payload: dict) -> str:
    if payload.get("claim_id"):
        return str(payload.get("claim_id"))
    cid = str(payload.get("context_id", "") or "")
    text = str(payload.get("claim_text", "") or "")[:60]
    return f"{cid}::{text}"


def run_issue_judge_only(
    merged_path: str,
    *,
    output_dir: str | None = None,
    claims_jsonl: str | None = None,
    issue_judge_models: list[str] | None = None,
    issue_judge_batch_size: int = ISSUE_DETECTOR_BATCH_SIZE,
    current_date: str | None = None,
    issue_judge_max_workers: int = DEFAULT_ISSUE_JUDGE_MAX_WORKERS,
) -> dict:
    merged_file = Path(merged_path).resolve()
    if not merged_file.exists():
        raise FileNotFoundError(f"merged_clean 파일 없음: {merged_file}")
    if not claims_jsonl:
        raise FileNotFoundError("1차 issue judge에는 claims_jsonl 경로가 필요합니다.")

    claims_path = Path(claims_jsonl).resolve()
    if not claims_path.exists():
        raise FileNotFoundError(f"claims jsonl 파일 없음: {claims_path}")

    base_stem = _base_stem(merged_file)
    out_dir = Path(output_dir).resolve() if output_dir else merged_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = out_dir / f"{base_stem}_issue_judge_compare.json"
    merged_issue_judge_path = out_dir / f"{base_stem}_issue_judge.json"
    summary_path = out_dir / f"{base_stem}_issue_judge_summary.json"

    if (
        _json_file_exists(summary_path)
        and _json_file_exists(comparison_path)
        and _json_file_exists(merged_issue_judge_path)
    ):
        print(f"  ⏭  issue judge — 출력 파일 존재, 스킵")
        print(f"     {merged_issue_judge_path}")
        merged_issue_judge = _load_json_file(merged_issue_judge_path)
        summary_payload = _load_json_file(summary_path)
        cached_summary = summary_payload.get("summary", {}) or {}
        cached_evaluated_count = int(cached_summary.get("evaluated_model_count", 0) or 0)
        cached_failed_models = cached_summary.get("failed_models", []) or []
        if cached_evaluated_count == 0 and cached_failed_models:
            raise RuntimeError(
                "캐시된 issue judge 결과가 전체 모델 실패 상태라 verifier를 계속 진행할 수 없습니다. "
                f"summary={summary_path}, failed_models={cached_failed_models}"
            )
        return {
            "merged_path": str(merged_file),
            "output_dir": str(out_dir),
            "issue_judge_summary": str(summary_path),
            "issue_judge_comparison": str(comparison_path),
            "issue_judge_merged": str(merged_issue_judge_path),
            "issue_judge_paths": summary_payload.get("issue_judge_result_paths", {}) or {},
            "issue_judge_count": (merged_issue_judge.get("summary", {}) or {}).get("merged_issue_count", 0),
            "skipped": True,
        }

    models = issue_judge_models or _default_issue_judge_models()
    if len(models) < 1:
        raise RuntimeError("issue judge 모델이 필요합니다. ISSUE_JUDGE_MODELS 또는 --issue-judge-models를 확인하세요.")
    missing_judge_keys = [
        f"{model}({missing} 없음)"
        for model in models
        for missing in [_missing_provider_key(model)]
        if missing
    ]
    if missing_judge_keys:
        raise RuntimeError(f"issue judge 모델 키가 필요합니다: {', '.join(missing_judge_keys)}")
    ctx = prepare_verification(str(merged_file), current_date=current_date)
    claims = _load_claims_jsonl(claims_path)

    claims_by_batch = _rebuild_classified_claim_batches(claims, ctx["contexts"], issue_judge_batch_size)
    claims_serialized = [{"batch": item["batch"], "claims": item["claims"]} for item in claims_by_batch]

    print(f"  issue judge 입력 claim 수: {len(claims)}개")
    print(f"  issue judge 모델: {', '.join(models)}")

    env_vars = _collect_env_vars()
    root = str(_ROOT)
    judge_results = {}
    issue_judge_max_workers = _env_int("ISSUE_JUDGE_MAX_WORKERS", issue_judge_max_workers)
    print(f"  issue judge worker: {issue_judge_max_workers}개")
    with ProcessPoolExecutor(max_workers=issue_judge_max_workers) as executor:
        futures = {
            executor.submit(
                _classified_issue_judge_worker,
                (
                    str(merged_file),
                    model,
                    claims_serialized,
                    current_date,
                    root,
                    env_vars,
                ),
            ): model
            for model in models
        }
        for future in as_completed(futures):
            model = futures[future]
            try:
                judge_results[model] = future.result()
            except Exception as e:
                print(f"  ❌ [{model}] 1차 issue judge 실패: {e}")
                judge_results[model] = {
                    "model": model,
                    "ok": False,
                    "error": str(e),
                    "issues": [],
                    "api_calls": 0,
                    "token_usage": _empty_token_usage(),
                }

    issue_judge_paths = _write_issue_judge_model_outputs(
        output_dir=out_dir,
        base_stem=base_stem,
        merged_path=merged_file,
        claims_path=str(claims_path),
        claims=claims,
        judge_results=judge_results,
    )
    comparison = _build_issue_judge_comparison(
        models=models,
        claims=claims,
        judge_results=judge_results,
        issue_judge_paths=issue_judge_paths,
        claims_path=str(claims_path),
    )
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    merged_issue_judge_path_str, merged_issue_judge = _write_issue_judge_merged_output(
        output_dir=out_dir,
        base_stem=base_stem,
        merged_path=merged_file,
        claims_path=str(claims_path),
        models=models,
        judge_results=judge_results,
    )

    total_token_usage = _empty_token_usage()
    token_usage_per_model = {}
    for model, result in judge_results.items():
        token_usage_per_model[model] = result.get("token_usage", _empty_token_usage())
        total_token_usage = _merge_token_usage(total_token_usage, token_usage_per_model[model])

    summary = {
        "schema_version": "issue_judge_summary.v1",
        "stage": "claim_to_issue_judge",
        "merged_path": str(merged_file),
        "source_claims_path": str(claims_path),
        "models": models,
        "issue_judge_result_paths": issue_judge_paths,
        "issue_judge_comparison_path": str(comparison_path),
        "issue_judge_merged_path": merged_issue_judge_path_str,
        "summary": comparison.get("summary", {}),
        "merged_summary": merged_issue_judge.get("summary", {}),
        "token_usage_per_model": token_usage_per_model,
        "elapsed_sec_per_model": {
            model: round(float(result.get("elapsed_sec", 0.0) or 0.0), 6)
            for model, result in judge_results.items()
        },
        "token_usage": total_token_usage,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    failed_models = summary.get("summary", {}).get("failed_models", []) or []
    evaluated_model_count = int(summary.get("summary", {}).get("evaluated_model_count", 0) or 0)
    if evaluated_model_count == 0:
        failure_details = "; ".join(
            f"{model}: {judge_results.get(model, {}).get('error', 'unknown error')}"
            for model in failed_models
        )
        raise RuntimeError(
            "issue judge 전체 모델이 실패해서 verifier를 계속 진행할 수 없습니다. "
            f"summary={summary_path}, failed_models={failed_models}. {failure_details}"
        )

    return {
        "merged_path": str(merged_file),
        "output_dir": str(out_dir),
        "issue_judge_summary": str(summary_path),
        "issue_judge_comparison": str(comparison_path),
        "issue_judge_merged": merged_issue_judge_path_str,
        "issue_judge_paths": issue_judge_paths,
        "issue_judge_count": merged_issue_judge.get("summary", {}).get("merged_issue_count", 0),
    }


def _claim_output_payload_for_classified_pipeline(claim: dict) -> dict:
    context_id = str(claim.get("context_id") or "").strip()
    payload = {
        "claim_id": claim.get("claim_id", ""),
        "context_id": context_id,
        "claim_text": claim.get("claim_text", ""),
        "resolved_claim": claim.get("resolved_claim", ""),
        "claim_type": claim.get("claim_type", ""),
    }
    return {key: value for key, value in payload.items() if value not in ("", [], None)}


def _extract_or_reuse_claims_for_classified_pipeline(
    merged_file: Path,
    out_dir: Path,
    *,
    claims_jsonl: str | None = None,
    reuse_claims: bool = False,
    current_date: str | None = None,
    claim_batch_size: int = CLAIM_EXTRACT_BATCH_SIZE,
) -> dict:
    base_stem = _base_stem(merged_file)
    result_json_path = out_dir / f"{base_stem}_verification_final.json"
    claims_jsonl_path = out_dir / f"{base_stem}_claims.jsonl"
    claims_json_path = out_dir / f"{base_stem}_claims.json"

    effective_claims_jsonl = claims_jsonl
    if not effective_claims_jsonl and _json_file_exists(claims_jsonl_path):
        effective_claims_jsonl = str(claims_jsonl_path)
        print(f"  ⏭  claim 추출 — 출력 파일 존재, 스킵")
        print(f"     {effective_claims_jsonl}")
    elif not effective_claims_jsonl and reuse_claims and claims_jsonl_path.exists():
        effective_claims_jsonl = str(claims_jsonl_path)
        print(f"  기존 claim 추출 결과 재사용: {effective_claims_jsonl}")
    if effective_claims_jsonl:
        claims = _load_claims_jsonl(effective_claims_jsonl)
        return {
            "claims_jsonl": str(Path(effective_claims_jsonl).resolve()),
            "claims_json": str(claims_json_path) if claims_json_path.exists() else "",
            "claims": claims,
            "claim_count": len(claims),
            "api_calls": 0,
            "token_usage": _empty_token_usage(),
            "reused": True,
        }

    from .claim_extractor import extract_claims_only

    print("  classified issue pipeline: claim 추출 시작")
    ctx = prepare_verification(str(merged_file), current_date=current_date)
    claims_by_batch, api_calls, token_usage = extract_claims_only(
        ctx["contexts"],
        ctx["current_date"],
        ctx["hint"],
        ctx["slide_ctx"],
        batch_size=claim_batch_size,
    )
    claims: list[dict] = []
    for _, batch_claims in claims_by_batch:
        claims.extend(_claim_output_payload_for_classified_pipeline(claim) for claim in batch_claims)

    claims_log_path = _write_claims_jsonl(claims, result_json_path)
    claims_json_path.write_text(
        json.dumps(
            {
                "mode": "claim_extraction",
                "merged_path": str(merged_file),
                "claims_log_path": claims_log_path,
                "claim_count": len(claims),
                "api_calls": api_calls,
                "token_usage": token_usage,
                "claims": claims,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "claims_jsonl": claims_log_path,
        "claims_json": str(claims_json_path),
        "claims": claims,
        "claim_count": len(claims),
        "api_calls": api_calls,
        "token_usage": token_usage,
        "reused": False,
    }


def _related_pipeline_path(merged_file: Path, suffix: str) -> Path:
    base_stem = _base_stem(merged_file)
    output_dir = merged_file.parent.parent if merged_file.parent.name.endswith("_analyzer") else merged_file.parent
    return output_dir / f"{base_stem}{suffix}"


def _format_classified_issue_report(content_view: dict) -> str:
    feedback_items = content_view.get("feedback_items", []) or []
    confirmed = [item for item in feedback_items if item.get("status") == STATUS_CONFIRMED]
    needs_review = [item for item in feedback_items if item.get("status") == STATUS_PROFESSOR_CHECK]
    rejected = [item for item in feedback_items if item.get("status") == STATUS_REJECTED]
    slide_errors = content_view.get("slide_errors", []) or []
    verifier_result = (content_view.get("views", {}) or {}).get("classified_issue_verifier", {}) or {}
    summary = content_view.get("summary", {}) or {}

    lines = [
        "=" * 60,
        "강의 내용 검증 리포트 (classified issue pipeline)",
        "=" * 60,
        f"\n검증일: {content_view.get('verification_date', '')}",
        f"전체 후보: {summary.get('total_feedback_count', len(feedback_items))}건",
        f"확정: {len(confirmed)}건 / 검토 필요: {len(needs_review)}건 / 기각: {len(rejected)}건",
        f"슬라이드 오타: {len(slide_errors)}건",
    ]

    breakdown = summary.get("breakdown_by_type", {}) if isinstance(summary.get("breakdown_by_type"), dict) else {}
    if breakdown:
        labels = {
            "factual_error": "사실 오류",
            "temporal_error": "오래된 내용",
            "scope_overclaim": "과도한 일반화",
            "confusing_explanation": "혼동 가능 설명",
            "composite_issue": "복합 오류",
        }
        parts = [f"{labels.get(key, key)} {value}건" for key, value in breakdown.items()]
        lines.append(f"유형별: {', '.join(parts)}")

    def add_feedback_section(title: str, rows: list[dict], limit: int | None = None) -> None:
        if not rows:
            return
        shown = rows if limit is None else rows[:limit]
        lines.append(f"\n{'-' * 40}")
        lines.append(f"{title} ({len(rows)}건)")
        lines.append("-" * 40)
        for index, item in enumerate(shown, 1):
            location = item.get("location") if isinstance(item.get("location"), dict) else {}
            score = float(item.get("severity_score", 0.0) or 0.0)
            problem = item.get("problem") if isinstance(item.get("problem"), dict) else {}
            lines.append(
                f"\n  [{index}] {item.get('feedback_label', item.get('feedback_type', ''))}"
                f" | {score * 100:.1f}점"
                f" | 슬라이드 {location.get('slide_number', '?')}"
            )
            claim = item.get("resolved_claim") or item.get("claim_text") or ""
            if claim:
                lines.append(f"    claim: {claim[:180]}")
            summary_text = problem.get("summary") or problem.get("why_wrong") or ""
            if summary_text:
                lines.append(f"    근거: {summary_text[:240]}")
            recommendation = problem.get("recommendation") or ""
            if recommendation:
                lines.append(f"    수정안: {recommendation[:180]}")
        if limit is not None and len(rows) > limit:
            lines.append(f"\n  ... 외 {len(rows) - limit}건")

    add_feedback_section("확정 이슈", confirmed)
    add_feedback_section("검토 필요", needs_review)
    add_feedback_section("기각", rejected, limit=10)

    if slide_errors:
        lines.append(f"\n{'-' * 40}")
        lines.append(f"슬라이드 오타 ({len(slide_errors)}건)")
        lines.append("-" * 40)
        for index, error in enumerate(slide_errors, 1):
            lines.append(f"\n  [{index}] 슬라이드 {error.get('slide_number', '?')} ({error.get('slide_title', '')})")
            lines.append(f"    유형: {error.get('error_type_label', error.get('error_type', ''))}")
            lines.append(f"    문제: {error.get('problematic_text', '')}")
            lines.append(f"    수정: {error.get('corrected_text', '')}")
            lines.append(f"    이유: {error.get('reason', '')}")
            lines.append(f"    신뢰도: {float(error.get('confidence', 0) or 0):.0%}")

    slide_error_status = content_view.get("slide_error_status", "")
    if slide_error_status and slide_error_status != "ok":
        lines.append(f"\n슬라이드 오타 검사 상태: {slide_error_status}")

    model_breakdown = (verifier_result.get("summary", {}) or {}).get("model_breakdown", {})
    if model_breakdown:
        lines.append(f"\n{'=' * 60}")
        lines.append("모델 판정 요약")
        lines.append("=" * 60)
        for model, row in model_breakdown.items():
            lines.append(
                f"  {model}: {row.get('status', '')}, "
                f"parsed={row.get('judgment_count', 0)}, "
                f"parse_failed={row.get('parse_failed_count', 0)}"
            )

    return "\n".join(lines)


def run_classified_issue_pipeline(
    merged_path: str,
    *,
    output_dir: str | None = None,
    claims_jsonl: str | None = None,
    reuse_claims: bool = False,
    current_date: str | None = None,
    issue_judge_models: list[str] | None = None,
    issue_type_models: list[str] | None = None,
    verifier_models: list[str] | None = None,
    issue_type_model_weights: str | None = None,
    verifier_model_weights: str | None = None,
    claim_batch_size: int = CLAIM_EXTRACT_BATCH_SIZE,
    issue_judge_batch_size: int = ISSUE_DETECTOR_BATCH_SIZE,
    issue_type_batch_size: int = ISSUE_TYPE_CLASSIFIER_BATCH_SIZE,
    verifier_batch_size: int = CLASSIFIED_ISSUE_VERIFIER_BATCH_SIZE,
    max_workers: int | None = None,
    max_tokens: int = 8192,
    stage_notify: Callable[[str, str], None] | None = None,
) -> dict:
    """Run the user's classified issue flow end-to-end.

    Flow:
    claim extraction -> first issue judge -> issue type classifier ->
    category-specific issue verifier -> web-friendly verification_final.json.
    """

    # The backend calls this function directly instead of entering CLI main().
    # Keep verifier output visible in both pipeline.log and Docker logs.
    _enable_docker_log_tee()

    merged_file = Path(merged_path).resolve()
    if not merged_file.exists():
        raise FileNotFoundError(f"merged_clean 파일 없음: {merged_file}")
    base_stem = _base_stem(merged_file)
    out_dir = Path(output_dir).resolve() if output_dir else merged_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_started_at = time.perf_counter()
    stage_timings: dict[str, float] = {}
    merged_payload = _load_json_file(merged_file)
    slide_count = int(
        merged_payload.get("total_slides", 0)
        or len(merged_payload.get("slides", []) or [])
    )
    context_count = int(merged_payload.get("total_contexts", 0) or 0)
    if context_count <= 0:
        context_count = sum(
            len((slide.get("contexts") or []))
            for slide in (merged_payload.get("slides") or [])
            if isinstance(slide, dict)
        )
    _llm_pipeline_banner(
        merged_file=merged_file,
        output_dir=out_dir,
        duration=merged_payload.get("total_duration_formatted", ""),
        slide_count=slide_count,
        context_count=context_count,
    )

    # The full video pipeline does not pass worker counts explicitly. Resolve
    # them here so it uses the same per-model concurrency as the standalone
    # verifier commands instead of silently falling back to one worker.
    shared_max_workers = _env_int("VERIFIER_STAGE_MAX_WORKERS", max_workers or 20)
    issue_type_max_workers = _env_int("ISSUE_TYPE_CLASSIFIER_MAX_WORKERS", shared_max_workers)
    final_verifier_max_workers = _env_int(
        "CLASSIFIED_ISSUE_VERIFIER_MAX_WORKERS", shared_max_workers
    )
    evidence_max_workers = _env_int(
        "CLASSIFIED_ISSUE_EVIDENCE_MAX_WORKERS",
        _env_int("CLASSIFIED_ISSUE_GROUNDING_MAX_WORKERS", shared_max_workers),
    )
    slide_error_max_workers = _env_int(
        "CLASSIFIED_SLIDE_ERROR_MAX_WORKERS", shared_max_workers
    )
    print(
        "  verifier 병렬 설정: "
        f"shared={shared_max_workers}, classifier_per_model={issue_type_max_workers}, "
        f"final_per_model={final_verifier_max_workers}, "
        f"evidence={evidence_max_workers}, "
        f"slide_error_per_model={slide_error_max_workers}",
        flush=True,
    )

    # 서브스테이지별 소요시간. notify()가 이미 모든 스테이지 경계(run/done/error)를
    # 호출하고 있어서 그 지점을 그대로 재사용한다 — 별도로 새 계측 지점을 만들지
    # 않는다. 이 함수는 각 스테이지의 작업을 동기 호출하는 메인 스레드에서만
    # 실행되므로(스테이지 내부의 ThreadPoolExecutor 동시성과는 무관), 락 없이
    # dict에 써도 경쟁 조건이 없다.
    stage_timings: dict[str, float] = {}
    _stage_started_at: dict[str, float] = {}

    def notify(stage: str, status: str) -> None:
        if status == "run":
            _stage_started_at[stage] = time.time()
        elif status in ("done", "error"):
            started_at = _stage_started_at.pop(stage, None)
            if started_at is not None:
                stage_timings[stage] = time.time() - started_at
        if not stage_notify:
            return
        stage_notify(stage, status)

    claim_stage_started = _llm_stage_start(
        "L1",
        "extract_claims — Claim 추출",
        [
            ("입력 Context", f"{context_count}개"),
            ("배치 크기", claim_batch_size),
        ],
    )
    notify("verifier_claim_extraction", "run")
    try:
        claims_result = _extract_or_reuse_claims_for_classified_pipeline(
            merged_file,
            out_dir,
            claims_jsonl=claims_jsonl,
            reuse_claims=reuse_claims,
            current_date=current_date,
            claim_batch_size=claim_batch_size,
        )
    except Exception as exc:
        _llm_stage_failed("L1", "extract_claims — Claim 추출", claim_stage_started, exc)
        notify("verifier_claim_extraction", "error")
        raise
    notify("verifier_claim_extraction", "done")
    stage_timings["L1 Claim 추출"] = _llm_stage_done(
        "L1",
        "extract_claims — Claim 추출",
        claim_stage_started,
        rows=[
            ("추출 Claim", f"{claims_result.get('claim_count', 0)}개"),
            ("API 호출", f"{claims_result.get('api_calls', 0)}회"),
            ("실행 방식", "기존 결과 재사용" if claims_result.get("reused") else "새로 추출"),
        ],
        files=[claims_result.get("claims_jsonl", ""), claims_result.get("claims_json", "")],
    )

    resolved_issue_judge_models = issue_judge_models or _default_issue_judge_models()
    issue_judge_stage_started = _llm_stage_start(
        "L2",
        "detect_issues — Detector 앙상블",
        [
            ("입력 Claim", f"{claims_result.get('claim_count', 0)}개"),
            ("모델", ", ".join(resolved_issue_judge_models)),
            ("모델 병렬 수", shared_max_workers),
            ("Context 배치 크기", issue_judge_batch_size),
        ],
    )
    notify("verifier_issue_judge", "run")
    try:
        issue_judge_result = run_issue_judge_only(
            str(merged_file),
            output_dir=str(out_dir),
            claims_jsonl=claims_result["claims_jsonl"],
            issue_judge_models=resolved_issue_judge_models,
            current_date=current_date,
            issue_judge_batch_size=issue_judge_batch_size,
            issue_judge_max_workers=shared_max_workers,
        )
    except Exception as exc:
        _llm_stage_failed("L2", "detect_issues — Detector 앙상블", issue_judge_stage_started, exc)
        notify("verifier_issue_judge", "error")
        raise
    notify("verifier_issue_judge", "done")

    issue_judge_summary_payload = _load_json_file(Path(issue_judge_result["issue_judge_summary"]))
    issue_judge_summary = issue_judge_summary_payload.get("summary", {}) or {}
    issue_counts_by_model = issue_judge_summary.get("issue_counts_by_model", {}) or {}
    elapsed_by_model = issue_judge_summary_payload.get("elapsed_sec_per_model", {}) or {}
    detector_rows: list[tuple[str, object]] = []
    for model in resolved_issue_judge_models:
        model_count = int(issue_counts_by_model.get(model, 0) or 0)
        model_elapsed = float(elapsed_by_model.get(model, 0.0) or 0.0)
        elapsed_suffix = f", {_format_elapsed(model_elapsed)}" if model_elapsed > 0 else ""
        detector_rows.append((model, f"후보 {model_count}개{elapsed_suffix}"))
    detector_rows.extend([
        ("통합 후보", f"{issue_judge_result.get('issue_judge_count', 0)}개"),
        (
            "단일 모델 기준 미달 기각",
            f"{issue_judge_summary.get('rejected_single_model_low_confidence_count', 0)}개",
        ),
        ("모델 실패", ", ".join(issue_judge_summary.get("failed_models", []) or []) or "없음"),
    ])
    stage_timings["L2 Detector 앙상블"] = _llm_stage_done(
        "L2",
        "detect_issues — Detector 앙상블",
        issue_judge_stage_started,
        rows=detector_rows,
        files=[
            issue_judge_result.get("issue_judge_comparison", ""),
            issue_judge_result.get("issue_judge_merged", ""),
            issue_judge_result.get("issue_judge_summary", ""),
        ],
    )

    from .issue_type_classifier import (
        build_next_stage_input,
        classify_issues,
        _default_output_path as _issue_type_default_output_path,
        _default_next_input_path as _issue_type_default_next_input_path,
        _default_models as _issue_type_default_models,
    )
    from .classified_issue_verifier import (
        build_content_verification_view,
        judge_classified_issues,
        _default_output_path as _verifier_default_output_path,
        _default_models as _verifier_default_models,
    )
    from .classified_issue_grounder import collect_pre_verifier_evidence_batched
    from .classified_slide_error_checker import (
        detect_classified_slide_errors,
    )

    issue_judge_merged_path = Path(issue_judge_result["issue_judge_merged"]).resolve()
    issue_judge_payload = json.loads(issue_judge_merged_path.read_text(encoding="utf-8"))
    issue_type_output_path = _issue_type_default_output_path(issue_judge_merged_path)
    classified_input_path = _issue_type_default_next_input_path(issue_type_output_path)
    resolved_issue_type_models = issue_type_models or _issue_type_default_models()
    classifier_cached = _json_file_exists(issue_type_output_path) and _json_file_exists(classified_input_path)
    classifier_stage_started = _llm_stage_start(
        "L3",
        "classify_issue_types — 오류 유형 분류",
        [
            ("입력 후보", f"{issue_judge_result.get('issue_judge_count', 0)}개"),
            ("모델", ", ".join(resolved_issue_type_models)),
            ("배치 크기", issue_type_batch_size),
            ("모델별 병렬 수", issue_type_max_workers),
        ],
    )
    notify("verifier_issue_classification", "run")
    try:
        if classifier_cached:
            print(f"  ⏭  issue type classifier — 출력 파일 존재, 스킵")
            print(f"     {issue_type_output_path}")
            issue_type_result = _load_json_file(issue_type_output_path)
            classified_input = _load_json_file(classified_input_path)
        else:
            print(f"  issue type classifier 모델: {', '.join(resolved_issue_type_models)}")
            issue_type_result = classify_issues(
                issue_judge_payload,
                input_path=issue_judge_merged_path,
                merged_clean_path=merged_file,
                models=resolved_issue_type_models,
                list_keys=["issues"],
                batch_size=max(1, issue_type_batch_size),
                current_date=current_date or datetime.now().date().isoformat(),
                max_tokens=max(256, max_tokens),
                max_workers=issue_type_max_workers,
                model_weights_spec=issue_type_model_weights,
            )
            issue_type_output_path.write_text(json.dumps(issue_type_result, ensure_ascii=False, indent=2), encoding="utf-8")
            classified_input = build_next_stage_input(issue_type_result, classification_path=issue_type_output_path)
            classified_input_path.write_text(json.dumps(classified_input, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        _llm_stage_failed("L3", "classify_issue_types — 오류 유형 분류", classifier_stage_started, exc)
        notify("verifier_issue_classification", "error")
        raise
    notify("verifier_issue_classification", "done")

    classifier_summary = issue_type_result.get("summary", {}) or {}
    classifier_model_summary = classifier_summary.get("model_breakdown_by_type", {}) or {}
    classifier_parse_failures = sum(
        int((row or {}).get("parse_failed_count", 0) or 0)
        for row in classifier_model_summary.values()
        if isinstance(row, dict)
    )
    stage_timings["L3 오류 유형 분류"] = _llm_stage_done(
        "L3",
        "classify_issue_types — 오류 유형 분류",
        classifier_stage_started,
        rows=[
            ("분류 완료", f"{classifier_summary.get('input_issue_count', 0)}개"),
            ("유형별", _format_breakdown(classifier_summary.get("breakdown_by_type"))),
            ("복합 오류 라우팅", f"{classifier_summary.get('composite_count', 0)}개"),
            ("파싱 실패", f"{classifier_parse_failures}건"),
            ("실행 방식", "기존 결과 재사용" if classifier_cached else "새로 분류"),
        ],
        files=[issue_type_output_path, classified_input_path],
    )

    evidence_enabled = os.getenv("CLASSIFIED_ISSUE_EVIDENCE_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    evidence_output_path = out_dir / f"{base_stem}_classified_issue_evidence.json"
    evidence_result: dict = {}
    if evidence_enabled:
        evidence_cached = _json_file_exists(evidence_output_path)
        evidence_stage_started = _llm_stage_start(
            "L4",
            "ground_evidence — 웹 근거 수집",
            [
                ("입력 후보", f"{classifier_summary.get('input_issue_count', 0)}개"),
                ("병렬 수", evidence_max_workers),
            ],
        )
        notify("verifier_web_grounding", "run")
        try:
            if evidence_cached:
                print(f"  ⏭  pre-verifier web evidence — 출력 파일 존재, 스킵")
                print(f"     {evidence_output_path}")
                evidence_result = _load_json_file(evidence_output_path)
            else:
                evidence_result = collect_pre_verifier_evidence_batched(
                    classified_input,
                    input_path=classified_input_path,
                    merged_clean_path=merged_file,
                    current_date=current_date or datetime.now().date().isoformat(),
                    max_workers=evidence_max_workers,
                    max_tokens=int(os.getenv("CLASSIFIED_ISSUE_EVIDENCE_MAX_TOKENS", "600")),
                )
                evidence_output_path.write_text(
                    json.dumps(evidence_result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception as exc:
            _llm_stage_failed("L4", "ground_evidence — 웹 근거 수집", evidence_stage_started, exc)
            notify("verifier_web_grounding", "error")
            raise
        notify("verifier_web_grounding", "done")
        evidence_summary = evidence_result.get("summary", {}) or {}
        stage_timings["L4 웹 근거 수집"] = _llm_stage_done(
            "L4",
            "ground_evidence — 웹 근거 수집",
            evidence_stage_started,
            rows=[
                ("근거 검색 대상", f"{evidence_summary.get('target_count', 0)}개"),
                ("고유 검색", f"{evidence_summary.get('unique_retrieval_count', 0)}회"),
                ("근거 확인", f"{evidence_summary.get('verified_count', 0)}개"),
                ("근거 불충분", f"{evidence_summary.get('insufficient_evidence_count', 0)}개"),
                ("실행 방식", "기존 결과 재사용" if evidence_cached else "새로 검색"),
            ],
            files=[evidence_output_path],
        )
    else:
        _llm_stage_skipped("L4", "ground_evidence — 웹 근거 수집", "설정에서 비활성화됨")

    verifier_output_path = _verifier_default_output_path(classified_input_path)
    resolved_verifier_models = verifier_models or _verifier_default_models()
    verifier_cached = _json_file_exists(verifier_output_path)
    verifier_stage_started = _llm_stage_start(
        "L5",
        "verify_issues — 최종 다중 모델 검증",
        [
            ("입력 후보", f"{classifier_summary.get('input_issue_count', 0)}개"),
            ("모델", ", ".join(resolved_verifier_models)),
            ("후보 배치 크기", verifier_batch_size),
            ("모델별 병렬 수", final_verifier_max_workers),
        ],
    )
    notify("verifier_final_verification", "run")
    try:
        if verifier_cached:
            print(f"  ⏭  classified issue verifier — 출력 파일 존재, 스킵")
            print(f"     {verifier_output_path}")
            verifier_result = _load_json_file(verifier_output_path)
            verifier_result["output_path"] = str(verifier_output_path)
        else:
            print(f"  classified issue verifier 모델: {', '.join(resolved_verifier_models)}")
            verifier_result = judge_classified_issues(
                classified_input,
                input_path=classified_input_path,
                merged_clean_path=merged_file,
                slide_textualized_path=_related_pipeline_path(merged_file, "_slide_textualized.json"),
                slide_classified_path=_related_pipeline_path(merged_file, "_slide_classified.json"),
                models=resolved_verifier_models,
                batch_size=max(1, verifier_batch_size),
                current_date=current_date or datetime.now().date().isoformat(),
                max_tokens=max(256, max_tokens),
                max_workers=final_verifier_max_workers,
                context_window=5,
                model_weights_spec=verifier_model_weights,
                web_evidence_payload=evidence_result,
                web_evidence_path=evidence_output_path if evidence_enabled else None,
            )
            verifier_result["output_path"] = str(verifier_output_path)
            verifier_output_path.write_text(json.dumps(verifier_result, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        _llm_stage_failed("L5", "verify_issues — 최종 다중 모델 검증", verifier_stage_started, exc)
        notify("verifier_final_verification", "error")
        raise
    notify("verifier_final_verification", "done")

    content_view = build_content_verification_view(verifier_result)
    verifier_summary = content_view.get("summary", {}) or {}
    raw_verifier_summary = verifier_result.get("summary", {}) or {}
    verifier_model_breakdown = raw_verifier_summary.get("model_breakdown", {}) or {}
    verifier_parse_failures = sum(
        int((row or {}).get("parse_failed_count", 0) or 0)
        for row in verifier_model_breakdown.values()
        if isinstance(row, dict)
    )
    stage_timings["L5 최종 다중 모델 검증"] = _llm_stage_done(
        "L5",
        "verify_issues — 최종 다중 모델 검증",
        verifier_stage_started,
        rows=[
            ("최종 후보", f"{verifier_summary.get('total_feedback_count', 0)}개"),
            ("확정", f"{verifier_summary.get('confirmed_feedback_count', 0)}개"),
            ("검토 필요", f"{verifier_summary.get('review_needed_feedback_count', 0)}개"),
            ("기각", f"{verifier_summary.get('rejected_feedback_count', 0)}개"),
            ("파싱 실패", f"{verifier_parse_failures}건"),
            ("실행 방식", "기존 결과 재사용" if verifier_cached else "새로 검증"),
        ],
        files=[verifier_output_path],
    )

    slide_textualized_path = _related_pipeline_path(merged_file, "_slide_textualized.json")
    slide_classified_path = _related_pipeline_path(merged_file, "_slide_classified.json")
    slide_error_output_path = out_dir / f"{base_stem}_slide_errors.json"
    slide_error_cached = _json_file_exists(slide_error_output_path)
    slide_error_stage_started = _llm_stage_start(
        "L6",
        "check_slide_errors — 슬라이드 오류 검사",
        [
            ("입력 슬라이드", f"{slide_count}개"),
            ("병렬 수", slide_error_max_workers),
        ],
    )
    notify("verify_slide_errors", "run")
    try:
        if slide_error_cached:
            print(f"  ⏭  classified slide error checker — 출력 파일 존재, 스킵")
            print(f"     {slide_error_output_path}")
            slide_error_result = _load_json_file(slide_error_output_path)
            slide_error_result["output_path"] = str(slide_error_output_path)
        else:
            slide_error_result = detect_classified_slide_errors(
                merged_clean_path=merged_file,
                slide_textualized_path=slide_textualized_path,
                slide_classified_path=slide_classified_path,
                batch_size=int(os.getenv("CLASSIFIED_SLIDE_ERROR_BATCH_SIZE", "5")),
                max_workers=slide_error_max_workers,
                max_tokens=int(os.getenv("CLASSIFIED_SLIDE_ERROR_MAX_TOKENS", "4096")),
                current_date=current_date or datetime.now().date().isoformat(),
            )
            slide_error_result["output_path"] = str(slide_error_output_path)
            slide_error_output_path.write_text(json.dumps(slide_error_result, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        _llm_stage_failed("L6", "check_slide_errors — 슬라이드 오류 검사", slide_error_stage_started, exc)
        notify("verify_slide_errors", "error")
        raise
    notify("verify_slide_errors", "done")

    slide_errors = slide_error_result.get("slide_errors", []) or []
    slide_error_summary = slide_error_result.get("summary", {}) or {}
    slide_error_failures = sum(
        len((row.get("batch_errors") or []))
        for row in (slide_error_result.get("model_results", {}) or {}).values()
        if isinstance(row, dict)
    )
    stage_timings["L6 슬라이드 오류 검사"] = _llm_stage_done(
        "L6",
        "check_slide_errors — 슬라이드 오류 검사",
        slide_error_stage_started,
        rows=[
            ("검사 슬라이드", f"{slide_error_summary.get('total_slide_count', slide_count)}개"),
            ("탐지 오류", f"{len(slide_errors)}개"),
            ("배치 실패", f"{slide_error_failures}건"),
            ("실행 방식", "기존 결과 재사용" if slide_error_cached else "새로 검사"),
        ],
        files=[slide_error_output_path],
    )

    content_view["slide_errors"] = slide_errors
    content_view["slide_error_status"] = "ok"
    content_view["slide_error_summary"] = slide_error_result.get("summary", {})
    content_view["slide_error_token_usage"] = slide_error_result.get("token_usage", {})
    content_view["slide_error_path"] = str(slide_error_output_path)
    content_view["slide_error_needs_review"] = []
    content_view["slide_error_consensus"] = slide_error_result.get("summary", {})
    content_view["slide_error_checker"] = "classified_slide_error_checker"
    content_view["slide_error_failures"] = slide_error_failures
    content_view["summary"]["slide_error_count"] = len(slide_errors)
    content_view["summary"]["slide_error_needs_review_count"] = 0
    content_view["counts"]["slide_errors"] = len(slide_errors)
    content_view["counts"]["slide_error_needs_review"] = 0
    content_view["classified_issue_artifacts"] = {
        "claims_jsonl": claims_result.get("claims_jsonl", ""),
        "claims_json": claims_result.get("claims_json", ""),
        "issue_judge_summary": issue_judge_result.get("issue_judge_summary", ""),
        "issue_judge": issue_judge_result.get("issue_judge_merged", ""),
        "issue_types": str(issue_type_output_path),
        "classified_issues": str(classified_input_path),
        "classified_issue_evidence": str(evidence_output_path) if evidence_enabled else "",
        "classified_issue_verifier": str(verifier_output_path),
        "classified_issue_grounding": "",
        "slide_errors": str(slide_error_output_path),
    }
    result_json_path = out_dir / f"{base_stem}_verification_final.json"
    report_path = out_dir / f"{base_stem}_report.txt"
    if _json_file_exists(result_json_path):
        print(f"  ⏭  content verification view — 출력 파일 존재, 스킵")
        print(f"     {result_json_path}")
    else:
        result_json_path.write_text(json.dumps(content_view, ensure_ascii=False, indent=2), encoding="utf-8")
    if report_path.exists() and report_path.stat().st_size > 0:
        print(f"  ⏭  content verification report — 출력 파일 존재, 스킵")
        print(f"     {report_path}")
    else:
        report_path.write_text(_format_classified_issue_report(content_view), encoding="utf-8")

    final_summary = content_view.get("summary", {}) or {}
    pipeline_elapsed = _llm_pipeline_done(
        started_at=pipeline_started_at,
        timings=stage_timings,
        rows=[
            ("추출 Claim", f"{claims_result.get('claim_count', 0)}개"),
            ("Detector 통합 후보", f"{issue_judge_result.get('issue_judge_count', 0)}개"),
            ("최종 판정", f"{final_summary.get('total_feedback_count', 0)}개"),
            ("확정", f"{final_summary.get('confirmed_feedback_count', 0)}개"),
            ("검토 필요", f"{final_summary.get('review_needed_feedback_count', 0)}개"),
            ("기각", f"{final_summary.get('rejected_feedback_count', 0)}개"),
            ("슬라이드 오류", f"{len(slide_errors)}개"),
            ("전체 파싱 실패", f"{classifier_parse_failures + verifier_parse_failures}건"),
        ],
        files=[
            claims_result.get("claims_json", ""),
            issue_judge_result.get("issue_judge_summary", ""),
            issue_type_output_path,
            verifier_output_path,
            slide_error_output_path,
            result_json_path,
            report_path,
        ],
    )

    return {
        "merged_path": str(merged_file),
        "output_dir": str(out_dir),
        "claim_output": str(result_json_path),
        "claim_report": str(report_path),
        "claim_issue_count": len(content_view.get("feedback_items", []) or []),
        "used_classified_issue_pipeline": True,
        "classified_issue_pipeline": True,
        "classified_issue_artifacts": content_view["classified_issue_artifacts"],
        "slide_error_count": len(slide_errors),
        "slide_error_failures": content_view["slide_error_failures"],
        "stage_timings": stage_timings,
        "llm_stage_timings_sec": stage_timings,
        "elapsed_sec": pipeline_elapsed,
    }


def main():
    _enable_docker_log_tee()

    parser = argparse.ArgumentParser(description="merged_clean 입력 기준 verifier 실행")
    parser.add_argument("merged_path", help="merged_clean.json 경로")
    parser.add_argument("--output-dir", default=None, help="결과 저장 디렉토리 (기본: merged 파일 폴더)")
    parser.add_argument("--claims-jsonl", default=None, help="이미 추출된 claims.jsonl 경로. 지정하면 claim 추출을 건너뜀")
    parser.add_argument(
        "--reuse-claims",
        action="store_true",
        help="결과 폴더의 기존 *_claims.jsonl이 있으면 claim 추출을 건너뜀",
    )
    parser.add_argument(
        "--issue-judge-only",
        action="store_true",
        help="claims_jsonl로 1차 issue judge 결과만 생성",
    )
    parser.add_argument(
        "--issue-judge-models",
        nargs="+",
        default=None,
        help="1차 issue judge 모델 목록",
    )
    parser.add_argument(
        "--claim-batch-size",
        type=int,
        default=CLAIM_EXTRACT_BATCH_SIZE,
        help="Claim_extraction fallback 배치 크기. context 입력은 VERIFIER_CLAIM_EXTRACT_CONTEXT_GROUP_SIZE(기본 3)를 우선 사용",
    )
    parser.add_argument(
        "--issue-judge-batch-size",
        type=int,
        default=ISSUE_DETECTOR_BATCH_SIZE,
        help="1차 Issue_detection core context 배치 크기. 기본 VERIFIER_ISSUE_DETECTOR_BATCH_SIZE 또는 4",
    )
    parser.add_argument("--issue-judge-max-workers", type=int, default=DEFAULT_ISSUE_JUDGE_MAX_WORKERS)
    parser.add_argument(
        "--issue-type-batch-size",
        type=int,
        default=ISSUE_TYPE_CLASSIFIER_BATCH_SIZE,
        help="Issue_type classification에서 한 prompt에 넣을 issue 후보 수. 기본 VERIFIER_ISSUE_CLASSIFIER_BATCH_SIZE 또는 20",
    )
    parser.add_argument(
        "--verifier-batch-size",
        "--cross-batch-size",
        dest="verifier_batch_size",
        type=int,
        default=CLASSIFIED_ISSUE_VERIFIER_BATCH_SIZE,
        help="Multi_LLM_Verification에서 한 prompt에 넣을 issue 수. 기본 VERIFIER_CROSSCHECK_MAX_ISSUES_PER_BATCH 또는 5",
    )
    parser.add_argument("--date", default=None, help="검증 기준 날짜 (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.issue_judge_only:
        result = run_issue_judge_only(
            args.merged_path,
            output_dir=args.output_dir,
            claims_jsonl=args.claims_jsonl,
            issue_judge_models=args.issue_judge_models,
            issue_judge_batch_size=args.issue_judge_batch_size,
            current_date=args.date,
            issue_judge_max_workers=args.issue_judge_max_workers,
        )
        print("\n=== 1차 Issue Judge 완료 ===")
        print(f"merged    : {result['merged_path']}")
        print(f"summary   : {result['issue_judge_summary']}")
        print(f"comparison: {result['issue_judge_comparison']}")
        print(f"merged issues: {result['issue_judge_merged']}")
        print(f"issue claim 후보: {result['issue_judge_count']}건")
        return

    result = run_classified_issue_pipeline(
        args.merged_path,
        output_dir=args.output_dir,
        claims_jsonl=args.claims_jsonl,
        reuse_claims=args.reuse_claims,
        current_date=args.date,
        issue_judge_models=args.issue_judge_models,
        claim_batch_size=args.claim_batch_size,
        issue_judge_batch_size=args.issue_judge_batch_size,
        issue_type_batch_size=args.issue_type_batch_size,
        verifier_batch_size=args.verifier_batch_size,
        max_workers=args.issue_judge_max_workers,
        max_tokens=int(os.getenv("CLASSIFIED_ISSUE_PIPELINE_MAX_TOKENS", "8192")),
    )

    print("\n=== Classified Issue Pipeline 완료 ===")
    print(f"merged    : {result['merged_path']}")
    print(f"web result: {result['claim_output']}")
    print(f"issue 후보: {result['claim_issue_count']}건")
    print(f"slide 오류: {result.get('slide_error_count', 0)}건")
    for label, path in (result.get("classified_issue_artifacts") or {}).items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
