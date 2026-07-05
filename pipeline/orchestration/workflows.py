import json
import time
from pathlib import Path

from .graph import run_graph_pipeline
from .preprocess import load_preprocess_result_from_outputs, run_preprocess_pipeline
from .verifier import run_verifier_pipeline


def _print_generated_files(output_files: list[str]) -> None:
    print("\n  생성된 파일:")
    for path_str in output_files:
        if not path_str:
            continue
        p = Path(path_str)
        print(f"    {'✓' if p.exists() else '✗'}  {p}")


def run_pipeline(args, progress_callback=None, *, helpers):
    total_start = time.time()
    timings: dict[str, float] = {}
    stage_status: dict[str, str] = {}

    def notify_stage(stage_key, status):
        if progress_callback:
            try:
                progress_callback(stage_key, status)
            except Exception as e:
                helpers.log.warning(f"progress_callback failed for {stage_key}: {e}")

    from ..config import output_paths

    stem = Path(args.input).stem
    slides_dir = Path(args.slides)
    output_dir = Path(args.output)
    job_type = helpers._normalize_pipeline_job_type(getattr(args, "job_type", None))
    output_dir.mkdir(parents=True, exist_ok=True)
    slides_dir.mkdir(parents=True, exist_ok=True)
    timing_path = output_dir / "pipeline_timings.json"

    def write_timings(current_stage: str | None = None) -> None:
        payload = {
            "stem": stem,
            "status": "running",
            "current_stage": current_stage,
            "started_at_epoch": total_start,
            "updated_at_epoch": time.time(),
            "elapsed_total_sec": time.time() - total_start,
            "timings": timings,
            "stage_status": stage_status,
        }
        tmp_path = timing_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(timing_path)

    def record_timing(stage: str, elapsed: float, status: str = "done") -> None:
        timings[stage] = elapsed
        stage_status[stage] = status
        write_timings(stage)

    write_timings("pipeline_start")

    paths = output_paths(stem, output_dir, slides_dir)
    try:
        from ..cost_report import configure as configure_cost_report, reset as reset_cost_report

        reset_cost_report()
        configure_cost_report(stem=stem, output_dir=output_dir)
    except Exception as e:
        helpers.log.warning(f"cost_report 초기화 실패: {e}")

    print("\n" + "═" * 70)
    print("  강의 영상 분석 통합 파이프라인")
    print("═" * 70)
    print(f"  입력 영상 : {args.input}")
    print(f"  슬라이드  : {slides_dir}")
    print(f"  출력      : {output_dir}")
    print(f"  workflow  : {job_type}")
    if args.force:
        print("  ⚠️  --force: 모든 단계 강제 재실행")

    try:
        r9: dict = {}
        r10: dict = {}
        r10a: dict = {}
        r10b: dict = {}
        should_run_verifier = job_type in {
            helpers.JOB_TYPE_LEGACY_FULL,
            helpers.JOB_TYPE_VERIFY,
            helpers.JOB_TYPE_VERIFIED_UPLOAD,
        }
        should_run_preprocess = job_type in {
            helpers.JOB_TYPE_LEGACY_FULL,
            helpers.JOB_TYPE_VERIFY,
            helpers.JOB_TYPE_PUBLISH,
            helpers.JOB_TYPE_DIRECT_UPLOAD,
            helpers.JOB_TYPE_VERIFIED_UPLOAD,
        }
        should_run_graph = job_type in {
            helpers.JOB_TYPE_LEGACY_FULL,
            helpers.JOB_TYPE_PUBLISH,
            helpers.JOB_TYPE_DIRECT_UPLOAD,
            helpers.JOB_TYPE_GRAPH_UPLOAD,
        }

        if job_type == helpers.JOB_TYPE_GRAPH_UPLOAD:
            preprocess_result = load_preprocess_result_from_outputs(stem, output_dir, paths)
            print("\n  ⏭  preprocess 단계 — 저장된 산출물 manifest에서 복원")
            print("─" * 70)
            timings["P1A extract_slides — 슬라이드 추출"] = 0.0
            timings["P1B analyze_audio_quality — 오디오 품질 분석"] = 0.0
            timings["P1 extract_media total — 슬라이드 추출 + 오디오 품질 분석 총합"] = 0.0
            timings["P2A textualize_slides — 슬라이드 텍스트화"] = 0.0
            timings["P2B transcribe_audio — 전체 전사"] = 0.0
            timings["P2 textualize_transcribe total — 텍스트화 + 전사 총합"] = 0.0
            timings["P3A analyze_annotation — 필기 강조 분석"] = 0.0
            timings["P3 enrich_audio_annotation total — 보강 분석 총합"] = 0.0
        elif should_run_preprocess:
            preprocess_result = run_preprocess_pipeline(
                args,
                stem=stem,
                output_dir=output_dir,
                slides_dir=slides_dir,
                paths=paths,
                timings=timings,
                notify_stage=notify_stage,
                helpers=helpers,
            )
        else:
            raise RuntimeError(f"지원하지 않는 workflow 타입입니다: {job_type}")

        textualized_path = preprocess_result["textualized_path"]
        audio_result = preprocess_result["audio_result"]

        if should_run_verifier:
            verifier_result = run_verifier_pipeline(
                args,
                preprocess_result=preprocess_result,
                output_dir=output_dir,
                paths=paths,
                timings=timings,
                background=job_type not in {
                    helpers.JOB_TYPE_VERIFY,
                    helpers.JOB_TYPE_VERIFIED_UPLOAD,
                },
                notify_stage=notify_stage,
                helpers=helpers,
            )
            r9 = verifier_result["analyzer_input"]
            r10 = verifier_result["verifier_result"]
            r10a = verifier_result["claims_result"]
            r10b = verifier_result["issue_judge_result"]
        else:
            print("\n  ⏭  verifier 단계 — workflow 설정으로 스킵")
            print("─" * 70)
            timings["V1 build_analyzer_input — verifier 입력 생성"] = 0.0
            timings["V2 verifier 백그라운드 시작"] = 0.0

        if should_run_verifier and (
            getattr(args, "stop_after_claim_extract", False)
            or getattr(args, "stop_after_issue_judge", False)
            or getattr(args, "stop_after_verifier_start", False)
        ):
            if getattr(args, "stop_after_issue_judge", False):
                option_name = "--stop-after-issue-judge"
            elif getattr(args, "stop_after_claim_extract", False):
                option_name = "--stop-after-claim-extract"
            else:
                option_name = "--stop-after-verifier-start"
            print(f"\n  ⏹  {option_name}: 요청한 analyzer 단계 후 파이프라인을 종료합니다.")
            print("  생성된 analyzer 관련 파일:")
            for path_str in (
                r9.get("merged_clean_path", str(output_dir / f"{stem}_analyzer" / f"{stem}_merged_clean.json")),
                r10a.get("claims_jsonl", ""),
                r10a.get("claims_json", ""),
                r10b.get("issue_judge_summary", ""),
                r10b.get("issue_judge_comparison", ""),
                *list((r10b.get("issue_judge_paths") or {}).values()),
                r10.get("claim_output", ""),
                r10.get("log_path", ""),
            ):
                if path_str:
                    p = Path(path_str)
                    print(f"    {'✓' if p.exists() else '…'}  {p}")
            if r10.get("spawned"):
                print(f"\n  verifier는 백그라운드에서 계속 실행 중입니다. (PID {r10.get('pid')})")
            return

        if job_type in {helpers.JOB_TYPE_VERIFY, helpers.JOB_TYPE_VERIFIED_UPLOAD}:
            print(f"\n  ⏹  {job_type}: graph 단계는 승인 이후 publish에서 실행합니다.")
            output_files = [
                audio_result.get("segments_path", ""),
                audio_result.get("silences_path", ""),
                audio_result.get("emphasis_path", ""),
                preprocess_result.get("annotation_path", ""),
                textualized_path,
                r9.get("merged_clean_path", str(output_dir / f"{stem}_analyzer" / f"{stem}_merged_clean.json")),
                r10a.get("claims_jsonl", ""),
                r10a.get("claims_json", ""),
                r10b.get("issue_judge_summary", ""),
                r10b.get("issue_judge_comparison", ""),
                *list((r10b.get("issue_judge_paths") or {}).values()),
                r10.get("log_path", ""),
            ]
            for analyzer_path in (
                r10.get("claim_output", ""),
                r10.get("claim_report", ""),
            ):
                if analyzer_path and Path(analyzer_path).exists():
                    output_files.append(analyzer_path)
            _print_generated_files(output_files)
            if r10.get("spawned"):
                print(f"\n  verifier는 백그라운드에서 계속 실행 중입니다. (PID {r10.get('pid')})")
                print(f"  로그 파일: {r10.get('log_path')}")
                print()
            return

        if not should_run_graph:
            print("\n  ⏭  graph 단계 — workflow 설정으로 스킵")
            print("─" * 70)
            return

        graph_result = run_graph_pipeline(
            args,
            output_dir=output_dir,
            slides_dir=slides_dir,
            preprocess_result=preprocess_result,
            paths=paths,
            timings=timings,
            stage_status=stage_status,
            notify_stage=notify_stage,
            write_timings=write_timings,
            record_timing=record_timing,
            helpers=helpers,
        )
        classified_result = graph_result["classified_result"]
        by_scene_result = graph_result["by_scene_result"]
        r5 = graph_result["fusion_result"]
        r6 = graph_result["graph_triples_result"]
        r7b = graph_result["graphrag_result"]
        r8 = graph_result["metadata_result"]
        annotation_path = graph_result["annotation_path"]

        if not timings.get("V1 build_analyzer_input — verifier 입력 생성"):
            timings["V1 build_analyzer_input — verifier 입력 생성"] = 0.0
        if "V2 verifier 백그라운드 시작" not in timings:
            timings["V2 verifier 백그라운드 시작"] = 0.0
        if "V2A extract_claims — claim 추출" not in timings and (
            getattr(args, "stop_after_claim_extract", False) or getattr(args, "stop_after_issue_judge", False)
        ):
            timings["V2A extract_claims — claim 추출"] = 0.0
        if "V2B judge_issues — 1차 issue 판단" not in timings and getattr(args, "stop_after_issue_judge", False):
            timings["V2B judge_issues — 1차 issue 판단"] = 0.0

        output_files = [
            audio_result.get("segments_path", ""),
            audio_result.get("silences_path", ""),
            audio_result.get("emphasis_path", ""),
            annotation_path,
            textualized_path,
            classified_result.get("classified_path", ""),
            by_scene_result.get("by_scene_path", ""),
            r5.get("fused_path", ""),
            r6.get("triples_parquet", ""),
            r6.get("nodes_parquet", ""),
            r6.get("edges_parquet", ""),
            str(output_dir / f"{stem}_chunks_lance.parquet"),
            r7b.get("input_path", ""),
            r7b.get("entities_parquet", ""),
            r7b.get("relationships_parquet", ""),
            r8.get("metadata_path", ""),
            str(Path(getattr(args, "recommender_db_dir", helpers.DEFAULT_RECOMMENDER_DB_DIR))),
        ]
        if should_run_verifier:
            output_files.extend([
                r9.get("merged_clean_path", str(output_dir / f"{stem}_analyzer" / f"{stem}_merged_clean.json")),
                r10a.get("claims_jsonl", ""),
                r10a.get("claims_json", ""),
                r10b.get("issue_judge_summary", ""),
                r10b.get("issue_judge_comparison", ""),
                *list((r10b.get("issue_judge_paths") or {}).values()),
                r10.get("log_path", ""),
            ])
            for analyzer_path in (
                r10.get("claim_output", ""),
                r10.get("claim_report", ""),
            ):
                if analyzer_path and Path(analyzer_path).exists():
                    output_files.append(analyzer_path)
        _print_generated_files(output_files)
        if r10.get("spawned"):
            print(f"\n  verifier는 백그라운드에서 계속 실행 중입니다. (PID {r10.get('pid')})")
            print(f"  로그 파일: {r10.get('log_path')}")
            print()

    except Exception as e:
        print(f"\n❌ 파이프라인 오류: {e}")
        raise

    finally:
        try:
            from ..cost_report import write_report

            cost_report_path = write_report(
                stem=stem,
                output_dir=output_dir,
                timings=timings,
                analyzer_output_path=output_dir / f"{stem}_analyzer" / f"{stem}_verification_final.json",
            )
            print(f"\n  ✓ 비용 리포트 저장: {cost_report_path}")
        except Exception as e:
            print(f"\n  ⚠️ 비용 리포트 저장 실패: {e}")

        total_elapsed = time.time() - total_start
        try:
            final_payload = {
                "stem": stem,
                "status": "finished",
                "current_stage": "finished",
                "started_at_epoch": total_start,
                "updated_at_epoch": time.time(),
                "elapsed_total_sec": total_elapsed,
                "timings": timings,
                "stage_status": stage_status,
            }
            tmp_path = timing_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(timing_path)
        except Exception as e:
            print(f"\n  ⚠️ 타이밍 파일 저장 실패: {e}")
        print("\n" + "═" * 70)
        print("  단계별 소요 시간 (현재까지)")
        print("═" * 70)
        for stage, t in timings.items():
            label = "  (스킵)" if t == 0.0 else f"  {t:>7.1f}초"
            print(f"    {stage:<30} {label}")
        print(f"\n    {'총 소요 시간':<30}  {total_elapsed:>7.1f}초")

