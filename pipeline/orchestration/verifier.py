from pathlib import Path

# run_classified_issue_pipeline이 반환하는 stage_timings의 raw stage key를
# pipeline_timings.json에 쓰는 라벨로 매핑. 기존 "V2A extract_claims" /
# "V2B judge_issues" 네이밍(--stop-after-claim-extract 등 CLI 단독 실행 경로에서
# 이미 쓰이던 접두사)을 그대로 이어받아 C~F를 추가 — 두 경로가 같은 스테이지를
# 같은 키 계열로 기록하도록 맞춤
_VERIFIER_STAGE_TIMING_LABELS = {
    "verifier_claim_extraction": "V2A extract_claims — claim 추출",
    "verifier_issue_judge": "V2B judge_issues — 1차 issue 판단",
    "verifier_issue_classification": "V2C issue_classification — 이슈 유형 분류",
    "verifier_web_grounding": "V2D web_grounding — 웹 근거 검증",
    "verifier_final_verification": "V2E final_verification — 멀티 LLM 검증",
    "verify_slide_inspect": "V2F1 slide_inspect — 슬라이드 검사",
    "verify_slide_syntax": "V2F2 slide_syntax — 문법/코드 오류 점검",
}


def run_verifier_pipeline(
    args,
    *,
    preprocess_result: dict,
    output_dir: Path,
    paths: dict,
    timings: dict[str, float],
    background: bool = True,
    notify_stage=lambda _stage, _status, _progress=None: None,
    helpers,
) -> dict:
    """검증 입력을 구성하고 verifier 경로를 실행

    args의 stop_after_claim_extract/stop_after_issue_judge/skip_analyzer 여부에 따라
    claim 추출까지만, issue 판단까지만, verifier 자체를 건너뜀, 백그라운드 실행,
    동기 실행 중 하나의 경로를 선택
    """
    meta_path = preprocess_result["meta_path"]
    duration = preprocess_result["duration"]
    textualized_path = preprocess_result["textualized_path"]
    audio_result = preprocess_result["audio_result"]
    slides_structure = audio_result.get("scenes_structure") or audio_result.get("slides_structure")

    notify_stage("verifier_build_analyzer_input", "run")
    r9 = helpers.build_analyzer_input(
        args,
        meta_path=meta_path,
        textualized_path=textualized_path,
        segments_path=audio_result.get("segments_path", str(paths["segments"])),
        output_dir=output_dir,
        duration=audio_result.get("duration", duration),
        slides_structure=slides_structure,
    )
    timings["V1 build_analyzer_input — verifier 입력 생성"] = r9["elapsed"]
    notify_stage("verifier_build_analyzer_input", "done")

    r10: dict = {}
    r10a: dict = {}
    r10b: dict = {}

    if getattr(args, "stop_after_claim_extract", False) or getattr(args, "stop_after_issue_judge", False):
        notify_stage("verifier_claim_extraction", "run")
        r10a = helpers.extract_claims(
            args,
            merged_clean_path=r9["merged_clean_path"],
            output_dir=output_dir,
        )
        timings["V2A extract_claims — claim 추출"] = r10a["elapsed"]
        notify_stage("verifier_claim_extraction", "done")
        if getattr(args, "stop_after_issue_judge", False):
            notify_stage("verifier_issue_judge", "run")
            r10b = helpers.judge_issues(
                args,
                merged_clean_path=r9["merged_clean_path"],
                output_dir=output_dir,
                claims_jsonl=r10a["claims_jsonl"],
            )
            timings["V2B judge_issues — 1차 issue 판단"] = r10b["elapsed"]
            notify_stage("verifier_issue_judge", "done")
        timings["V2 verifier 백그라운드 시작"] = 0.0
    elif getattr(args, "skip_analyzer", False):
        timings["V2 verifier 백그라운드 시작"] = 0.0
    elif background:
        notify_stage("verifier_run", "run")
        r10 = helpers.start_verifier_background(
            args,
            merged_clean_path=r9["merged_clean_path"],
            output_dir=output_dir,
        )
        timings["V2 verifier 백그라운드 시작"] = r10["elapsed"]
        notify_stage("verifier_run", "done")
    else:
        r10 = helpers.run_verifier(
            args,
            merged_clean_path=r9["merged_clean_path"],
            output_dir=output_dir,
            notify_stage=notify_stage,
        )
        timings["V2 run_verifier — verifier 실행"] = r10["elapsed"]
        timings["V2 verifier 백그라운드 시작"] = 0.0
        for stage_key, elapsed in (r10.get("stage_timings") or {}).items():
            label = _VERIFIER_STAGE_TIMING_LABELS.get(stage_key)
            if label:
                timings[label] = elapsed

    return {
        "analyzer_input": r9,
        "verifier_result": r10,
        "claims_result": r10a,
        "issue_judge_result": r10b,
    }
