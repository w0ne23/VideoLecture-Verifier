from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


JUDGE_MIN_CONFIDENCE = 0.6


def issue_judge_min_confidence() -> float:
    raw = str(os.getenv("VERIFIER_ISSUE_JUDGE_MIN_CONFIDENCE", str(JUDGE_MIN_CONFIDENCE)) or "").strip()
    try:
        value = float(raw)
    except ValueError:
        return JUDGE_MIN_CONFIDENCE
    return max(0.0, min(1.0, value))


def _issue_judge_batch_max_workers() -> int:
    raw = (
        os.getenv("VERIFIER_ISSUE_JUDGE_BATCH_MAX_WORKERS")
        or os.getenv("VERIFIER_JUDGE_BATCH_MAX_WORKERS")
        or "2"
    )
    try:
        return max(1, int(raw))
    except ValueError:
        return 2


def _claim_id(claim: dict) -> str:
    return str(
        claim.get("claim_id")
        or claim.get("claim_fingerprint")
        or claim.get("context_id")
        or ""
    ).strip()


def _context_id(claim: dict) -> str:
    return str(claim.get("context_id") or "").strip()


def _context_ids(claim: dict) -> list[str]:
    values = claim.get("context_ids")
    if isinstance(values, list):
        ids = [str(value).strip() for value in values if str(value).strip()]
        if ids:
            return ids
    cid = _context_id(claim)
    return [cid] if cid else []


def _build_shared_judge_context(contexts: list[dict], slide_ctx: dict) -> str:
    from . import claim_common as cv

    slide_numbers = []
    seen_slides = set()
    for u in contexts:
        slide_number = int(u.get("slide_number", 0) or 0)
        if slide_number and slide_number not in seen_slides:
            seen_slides.add(slide_number)
            slide_numbers.append(slide_number)

    slide_lines = []
    for slide_number in slide_numbers:
        slide = slide_ctx.get(slide_number, {})
        title = slide.get("title", f"슬라이드 {slide_number}")
        time_range = slide.get("time_range", "")
        slide_lines.append(f"- 슬라이드 {slide_number}: {title} ({time_range})")

    context_lines = [
        f"- {cv._format_context_for_prompt(u)}"
        for u in contexts
    ]

    return (
        "[배치 공통 슬라이드]\n"
        + ("\n".join(slide_lines) if slide_lines else "(슬라이드 정보 없음)")
        + "\n\n[배치 공통 context]\n"
        + ("\n".join(context_lines) if context_lines else "(context 없음)")
    )


def _build_issue_candidate_prompt(
    claims: list[dict],
    contexts: list[dict],
    current_date: str,
    hint: dict,
    slide_ctx: dict,
) -> str:
    shared_context = _build_shared_judge_context(contexts, slide_ctx)
    min_confidence = issue_judge_min_confidence()
    claim_lines = []
    for i, c in enumerate(claims, 1):
        claim_id = _claim_id(c) or f"claim_{i}"
        context_id = _context_id(c)
        claim_text = str(c.get("claim_text") or "").strip()
        resolved = str(c.get("resolved_claim") or "").strip()
        lines = [
            f"{i}. claim_id: {claim_id}",
            f"   context_id: {context_id}",
            f"   claim_type: {c.get('claim_type', '?')}",
            f"   resolved_claim: {resolved or claim_text}",
            f"   claim_text: {claim_text}",
        ]
        if c.get("context_ids"):
            lines.append(f"   context_ids: {', '.join(str(x) for x in c.get('context_ids') or [])}")
        claim_lines.append("\n".join(lines))

    return f"""당신은 강의 claim 목록에서 1차 issue 후보만 선별하는 판정자입니다.
업로드/검증 기준일: {current_date}
강의 도메인: {hint['label']}

아래는 claim이 나온 강의 문맥과 claim 목록입니다.
이 단계는 후속 classified issue type/severity 단계 전의 1차 후보 추출 단계입니다. 최종 확정, 교수 피드백 작성, 정밀한 issue type 확정은 후속 단계에서 수행합니다.

배치 공통 문맥:
{shared_context}

판정 대상 claim 목록:
{chr(10).join(claim_lines)}

### 판정 기준

이 단계는 최종 사실 판정이 아니라, 후속 issue classification과 crosscheck가 다시 볼 만한 **검증 후보**를 고르는 단계입니다.
여기서 검증 후보란, LLM 자신의 일반적인 강의 도메인 지식으로 보았을 때 해당 claim이 틀렸거나 확인할 가치가 있어 보이는 후보를 뜻합니다.
따라서 "확정 오류"가 아니라도 원문 문맥 안에 모델 지식 기준으로 의심스러운 명제가 실제로 남으면 후보로 출력하세요.

입력 claim은 아래 순서로 해석하세요.

1. 기본 판정 대상은 `resolved_claim`입니다.
2. 먼저 `resolved_claim`이 `claim_text`, 배치 공통 문맥에서 실제로 전달된 의미를 충실히 보존했는지 확인하세요.
3. `resolved_claim`이 원문 context보다 주체, 대상, 분류명, 조건, 범위, 인과, 일반성을 과도하게 바꾸거나 넓혔다면 그 강해진 문장을 그대로 믿지 마세요.
4. 이 경우 `claim_text`와 배치 문맥에서 실제 전달된 더 좁은 명제를 기준으로 판단하세요.
5. 배치 공통 문맥은 지시어, 생략, 즉시 정정 여부와 claim의 실제 의미 범위를 확인하는 보조 문맥입니다.

각 claim의 confidence는 반드시 그 claim이 속한 `context_id`의 target context를 1차 판단 근거로 산정하세요.
배치에 함께 제공된 다른 context는 캐시/효율을 위해 같이 제공된 참고 자료이며,
해당 claim의 오류 가능성을 낮추거나 높이는 주된 근거로 사용하지 마세요.
다른 context는 claim_text/resolved_claim의 지시어, 생략된 주체, 조건, 대상이 target context만으로 불명확하거나,
바로 앞뒤 context가 같은 문장을 이어 설명하여 target context의 의미를 직접 완성하는 경우에만 보조 근거로 사용하세요.
특히 batch 안의 다른 context에 일반적으로 맞는 설명이 있다는 이유로
target context 안의 수치, 단위, 용어, 관계 오류를 낮게 평가하지 마세요.

다음 중 하나라도 구체적으로 남으면 issue 후보로 출력하세요.

- 정의, 분류, 포함 관계, 주체, 과정, 원인-결과, 작동 방식이 잘못 연결되었을 가능성
- 수치, 순서, 조건, 가능/불가능, 전체/일부, 일시/영구 같은 범위가 뒤바뀌었을 가능성
- 여러 구체 대상을 하나의 범주로 묶어 말했는데 그 범주명이 부정확할 가능성
- 강의가 특정 분류 체계, 대비표, 상하위 범주를 설명하는 문맥에서 대상을 다른 범주명으로 명시적으로 부른 가능성
  이 경우 그 용어가 넓은 일상적 의미로는 가능해 보여도, 강의 분류 기준상 혼동을 만들 수 있으면 `terminology` 또는 `definition_relation` 후보로 출력하세요.
- 수치와 단위, 기호 표기, 단위 변환이 결합되어 일반 도메인 지식상 불가능하거나 명백히 다른 물리량/척도로 전달될 가능성
  이 경우 단순 근사치 문제가 아니라 표기 해석 또는 단위가 바뀐 문제이면 `definition_relation` 후보로 출력하세요.
- 특정 조건에서만 맞는 말을 조건 없이 일반 사실처럼 말했을 가능성
- 결과를 과장하여 잘못된 일반 규칙이나 배제 관계가 남을 가능성
- 현재성, 최신성, 지원 여부, 사용 여부, 버전, 정책, 통계, 시장 상황처럼 업로드/검증 기준일({current_date}) 기준 확인이 필요한 가능성
- "현재", "요즘", "최근", "최신", "지원된다", "더 이상 사용하지 않는다" 같은 표현이 시점 의존 사실처럼 전달되는 경우
- 개념, 주체, 과정, 권한, 대상이 섞여 LLM 자신의 일반 도메인 지식 기준으로 확인할 가치가 있는 경우

다음 경우는 issue 후보로 출력하지 마세요.

- 제공된 배치 문맥이 같은 대상, 같은 관계, 같은 조건을 명확히 정정하거나 충분히 보완한 경우
- 바로 뒤 문장이 앞 표현을 같은 의미로 정확히 풀어 최종 의미가 올바르게 좁혀진 경우
- 애매한 지시어를 특정 대상으로 강제 해석해야만 문제가 생기는 경우
- 단순 표현 취향, 더 좋은 설명 가능성, 강의 수준 밖의 매우 엄밀한 예외만 남는 경우
- 단정을 하였지만, 일반적으로 통용하는 표현이거나, LLM 자신의 일반 도메인 지식으로 보았을 때 충분히 맞는 말로 보이는 경우
- 약/대략/정도/조금 같은 근사 표현이 있고, 수치가 보조 예시나 감각적 환산으로 쓰였으며 일반적으로 통용되는 근사이면 출력하지 마세요.

confidence는 최종 오류 확률이 아니라, 위의 지침을 확인한 후, LLM 자신의 일반 도메인 지식 기준으로 판단했을 때, 해당 claim이 틀렸거나 확인할 가치가 있어 보이는 정도를 0.0~1.0 사이의 숫자로 표현한 것입니다. 일반적으로 0.6 이상이면 후속 검증이 충분히 가치 있다고 판단한 경우입니다.

- 0.00~0.19: 일반적으로 통용되는 설명이며 후속 검증 후보로 보기 어렵습니다.
- 0.20~0.39: 대체로 맞는 설명이고, 표현 개선이나 엄밀성 보충 수준입니다.
- 0.40~0.59: 일반적으로 사용될 수 있으나 예외나 조건이 있어 약한 검증 후보입니다.
- 0.60~0.79: 표현이 애매하거나 예외/조건이 명확해 후속 검증할 가치가 있습니다.
- 0.80~1.00: LLM 자신의 일반 도메인 지식 기준으로 일반적으로 맞지 않는 말에 가깝습니다.

### 응답 (JSON만)

`basis_code`는 후속 검증이 필요한 주된 근거를 아래 코드 중 하나로만 쓰세요.
- `definition_relation`: 정의, 분류, 포함 관계, 개념 관계가 의심됨
- 수치/단위/기호 표기가 일반 도메인 지식상 불가능한 척도로 전달되는 경우도 `definition_relation`으로 쓰세요.
- `mechanism_actor`: 주체, 과정, 원인-결과, 작동 방식이 의심됨
- `scope_condition`: 조건, 범위, 전체/일부, 가능/불가능, 결과 과장이 의심됨
- `currentness`: 현재성, 최신성, 지원 여부, 사용 여부, 버전/정책/통계 확인이 필요함
- `terminology`: 용어 또는 분류명이 문맥상 다른 대상을 가리킬 가능성이 있음

```json
{{
  "claim_scores": [
    {{
      "claim_id": "CL0001",
      "basis_code": "definition_relation",
      "confidence": 0.0
    }}
  ],
  "issues": [
    {{
      "claim_id": "CL0001",
      "basis_code": "definition_relation",
      "confidence": 0.0
    }}
  ]
}}
```

지침:
1. `claim_scores`에는 입력된 모든 claim_id에 대해 정확히 하나씩, 입력 순서대로 confidence와 basis_code를 출력하세요.
2. `claim_scores`의 confidence는 threshold와 무관하게 0.0~1.0 사이 점수를 출력하세요.
3. `issues`에는 confidence {min_confidence:.2f} 이상인 후보만 출력하세요.
4. 입력 claim_id에 없는 새 claim을 만들지 마세요.
5. type, issue_type, context_id는 출력하지 마세요.
6. 같은 claim에서 같은 문제는 한 건만 출력하세요.
7. 문제가 없으면 {{"claim_scores": [...], "issues": []}}만 출력하세요.
8. basis_code는 위 다섯 코드 중 하나만 출력하세요.
9. issue, reason, candidate_reason, student_wrong_takeaway, wrong_claim 같은 설명/재작성 필드는 출력하지 마세요.
10. issues에는 claim_id, basis_code, confidence만 출력하세요. claim_text/resolved_claim은 서버가 claim_id로 원본 claim에서 붙입니다.
11. JSON 외 텍스트를 출력하지 마세요.
"""


def _order_issue_candidate_fields(issue: dict) -> dict:
    preferred = (
        "issue_id",
        "claim_id",
        "resolved_claim",
        "claim_text",
        "issue",
        "basis_code",
        "confidence",
        "context_id",
        "context_ids",
        "slide_number",
        "start_time",
        "end_time",
        "claim_type",
    )
    ordered = {key: issue[key] for key in preferred if key in issue}
    ordered.update({key: value for key, value in issue.items() if key not in ordered})
    return ordered


def _clamp_confidence(value) -> float:
    try:
        number = float(value or 0)
    except Exception:
        number = 0.0
    return max(0.0, min(1.0, number))


def _normalize_claim_scores(raw_scores: list, claims: list[dict]) -> list[dict]:
    scores_by_claim: dict[str, dict] = {}
    if isinstance(raw_scores, list):
        for raw_score in raw_scores:
            if not isinstance(raw_score, dict):
                continue
            claim_id = str(raw_score.get("claim_id", "") or "").strip()
            if not claim_id:
                continue
            scores_by_claim[claim_id] = {
                "claim_id": claim_id,
                "basis_code": str(raw_score.get("basis_code") or "").strip(),
                "confidence": _clamp_confidence(raw_score.get("confidence")),
            }

    normalized = []
    for claim in claims:
        claim_id = _claim_id(claim)
        if not claim_id:
            continue
        row = scores_by_claim.get(claim_id, {})
        normalized.append({
            "claim_id": claim_id,
            "context_id": _context_id(claim),
            "resolved_claim": str(claim.get("resolved_claim") or claim.get("claim_text") or "").strip(),
            "claim_text": str(claim.get("claim_text") or "").strip(),
            "claim_type": claim.get("claim_type", ""),
            "basis_code": row.get("basis_code", ""),
            "confidence": _clamp_confidence(row.get("confidence")),
        })
    return normalized


def _judge_issue_candidates(
    claims: list[dict],
    contexts: list[dict],
    current_date: str,
    hint: dict,
    context_map: dict,
    slide_ctx: dict,
) -> tuple[list[dict], bool, int, dict]:
    from . import claim_common as cv

    if not claims:
        return [], [], False, 0, cv._empty_token_usage()

    claim_by_id = {_claim_id(claim): claim for claim in claims if _claim_id(claim)}
    full_prompt = _build_issue_candidate_prompt(claims, contexts, current_date, hint, slide_ctx)
    system_prompt, prompt = _split_judge_prompt_for_cache(full_prompt)
    response_format = (
        {"type": "json_object"}
        if cv._supports_json_object_response_format(cv._resolve_stage_model("judge"))
        else None
    )
    api_calls = 0
    token_usage = cv._empty_token_usage()

    for attempt in range(cv.VERIFIER_PARSE_RETRIES + 1):
        text, call_usage = cv._call_llm(
            prompt,
            system_prompt=system_prompt,
            max_tokens=8192,
            temperature=0.0,
            thinking_budget=2048,
            response_format=response_format,
            stage="judge",
        )
        api_calls += 1
        cv._add_call_usage(token_usage, call_usage)
        try:
            payload = json.loads(cv._strip_json_fence(text.strip()))
        except json.JSONDecodeError:
            if attempt < cv.VERIFIER_PARSE_RETRIES:
                print(f"    ↺ 1차 issue judge JSON 파싱 재시도 ({attempt+1}/{cv.VERIFIER_PARSE_RETRIES})")
                continue
            return [], [], True, api_calls, token_usage

        raw = payload.get("issues", [])
        raw_scores = payload.get("claim_scores", [])
        if not isinstance(raw, list):
            return [], _normalize_claim_scores(raw_scores, claims), False, api_calls, token_usage

        claim_scores = _normalize_claim_scores(raw_scores, claims)
        issues = []
        seen = set()
        for raw_issue in raw:
            if not isinstance(raw_issue, dict):
                continue
            claim_id = str(raw_issue.get("claim_id", "") or "").strip()
            source_claim = claim_by_id.get(claim_id)
            if not source_claim:
                continue
            try:
                confidence = _clamp_confidence(raw_issue.get("confidence"))
            except Exception:
                confidence = 0.0
            if confidence < issue_judge_min_confidence():
                continue

            context_id = _context_id(source_claim)
            ref = context_map.get(context_id, {})
            issue_key = (
                claim_id,
            )
            if issue_key in seen:
                continue
            seen.add(issue_key)

            resolved_claim = str(source_claim.get("resolved_claim") or source_claim.get("claim_text") or "").strip()
            claim_text = str(source_claim.get("claim_text") or "").strip()
            source_issue_text = resolved_claim or claim_text
            basis_code = str(raw_issue.get("basis_code") or "").strip()
            issue = {
                "claim_id": claim_id,
                "resolved_claim": resolved_claim,
                "claim_text": claim_text,
                "issue": source_issue_text,
                "basis_code": basis_code,
                "confidence": confidence,
                "context_id": context_id,
                "context_ids": _context_ids(source_claim),
                "slide_number": ref.get("slide_number", source_claim.get("slide_number")),
                "start_time": ref.get("start_time", source_claim.get("start_time")),
                "end_time": ref.get("end_time", source_claim.get("end_time")),
                "claim_type": source_claim.get("claim_type", ""),
            }
            issues.append(_order_issue_candidate_fields(issue))

        return issues, claim_scores, False, api_calls, token_usage

    return [], [], True, api_calls, token_usage


def recover_issue_candidate_judgement(
    claims: list[dict],
    contexts: list[dict],
    current_date: str,
    hint: dict,
    context_map: dict,
    slide_ctx: dict,
    label: str,
) -> tuple[list[dict], list[dict], bool, int, dict, bool]:
    from . import claim_common as cv

    total_api_calls = 0
    total_token_usage = cv._empty_token_usage()
    last_issues: list[dict] = []
    last_claim_scores: list[dict] = []
    parse_failed = False
    last_exc = None

    for attempt in range(cv.VERIFIER_BATCH_RECOVERY_RETRIES + 1):
        if attempt > 0:
            print(f"    ↺ {label} 1차 issue judge 재처리 ({attempt}/{cv.VERIFIER_BATCH_RECOVERY_RETRIES})")
        try:
            issues, claim_scores, parse_failed, api_calls, token_usage = _judge_issue_candidates(
                claims, contexts, current_date, hint, context_map, slide_ctx
            )
            total_api_calls += api_calls
            total_token_usage = cv._merge_token_usage(total_token_usage, token_usage)
            last_issues = issues
            last_claim_scores = claim_scores
            if not parse_failed:
                return issues, claim_scores, False, total_api_calls, total_token_usage, True
        except Exception as e:
            last_exc = e

    if last_exc and not last_issues and total_api_calls == 0:
        raise last_exc
    return last_issues, last_claim_scores, True, total_api_calls, total_token_usage, False


def judge_issue_candidates_only(
    all_claims_by_batch: list[tuple],
    current_date: str,
    hint: dict,
    slide_ctx: dict,
    log_prefix: str = "",
) -> tuple[list[dict], list[dict], int, dict]:
    """Run only the first issue judge and return issue candidates."""
    from . import claim_common as cv

    total_api = 0
    total_token_usage = cv._empty_token_usage()
    all_issues = []
    all_claim_scores = []
    prefix = f"[{log_prefix}] " if log_prefix else ""
    total_batches = sum(1 for _, claims in all_claims_by_batch if claims)
    jobs = []

    for batch_idx, (batch, claims) in enumerate(all_claims_by_batch, start=1):
        if not claims:
            continue
        batch_map = {u["context_id"]: u for u in batch}
        ids = f"{batch[0]['context_id']}..{batch[-1]['context_id']}"
        jobs.append((len(jobs) + 1, batch_idx, batch, claims, batch_map, ids))

    def _run_job(job):
        active_idx, batch_idx, batch, claims, batch_map, ids = job
        print(f"  {prefix}1차 issue judge ({active_idx}/{total_batches}) {ids}", flush=True)
        issues, claim_scores, parse_failed, api_calls, token_usage, ok = recover_issue_candidate_judgement(
            claims,
            batch,
            current_date,
            hint,
            batch_map,
            slide_ctx,
            f"1차 issue judge 배치 {batch_idx} {ids}",
        )
        if not ok:
            print(f"    ⚠️ 1차 issue judge 실패: {ids} — 이 batch는 빈 결과로 기록됩니다.")
        return active_idx, issues, claim_scores, api_calls, token_usage

    worker_count = min(_issue_judge_batch_max_workers(), max(1, len(jobs)))
    if worker_count > 1 and len(jobs) > 1:
        print(f"  {prefix}1차 issue judge batch 병렬 처리: max_workers={worker_count}", flush=True)
        print(f"  {prefix}cache warm-up: 첫 batch 선실행 후 나머지 병렬 처리", flush=True)
        results = {}
        warmup_result = _run_job(jobs[0])
        results[warmup_result[0]] = warmup_result
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_run_job, job): job[0] for job in jobs[1:]}
            for future in as_completed(futures):
                active_idx = futures[future]
                try:
                    results[active_idx] = future.result()
                except Exception as e:
                    raise RuntimeError(f"issue judge batch {active_idx} failed: {e}") from e
        ordered_results = [results[job[0]] for job in jobs]
    else:
        ordered_results = [_run_job(job) for job in jobs]

    for _active_idx, issues, claim_scores, api_calls, token_usage in ordered_results:
        all_issues.extend(issues)
        all_claim_scores.extend(claim_scores)
        total_api += api_calls
        total_token_usage = cv._merge_token_usage(total_token_usage, token_usage)

    for index, issue in enumerate(all_issues, start=1):
        issue["issue_id"] = f"I{index:04d}"
        ordered = _order_issue_candidate_fields(issue)
        issue.clear()
        issue.update(ordered)

    return all_issues, all_claim_scores, total_api, total_token_usage


def _split_judge_prompt_for_cache(prompt: str) -> tuple[str | None, str]:
    dynamic_marker = "배치 공통 문맥:"
    instruction_marker = "### 판정 기준"
    dynamic_start = prompt.find(dynamic_marker)
    instruction_start = prompt.find(instruction_marker)
    if dynamic_start < 0 or instruction_start < 0 or dynamic_start >= instruction_start:
        return None, prompt
    system_prompt = prompt[:dynamic_start].rstrip() + "\n\n" + prompt[instruction_start:].lstrip()
    user_prompt = prompt[dynamic_start:instruction_start].strip()
    return system_prompt, user_prompt
