import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_preprocess_pipeline(
    args,
    *,
    stem: str,
    output_dir: Path,
    slides_dir: Path,
    paths: dict,
    timings: dict[str, float],
    notify_stage,
    helpers,
) -> dict:
    """Run shared preprocessing stages used by verifier and graph workflows."""
    helpers._banner("P1 extract_media — 슬라이드 추출 + 오디오 품질 분석")
    t_parallel = time.time()
    audio_analyze_result: dict = {}

    # 영상 레인(슬라이드 추출→텍스트화)과 오디오 레인(품질 분석→전사)은 실제로 서로
    # 독립된 ThreadPoolExecutor 작업이라, 각 future가 끝나는 시점에 그 레인만 개별
    # notify하면 두 레인이 진짜 따로 진행되다가 통합 텍스트(P3) 앞에서 자연히 합류하는
    # 모습을 프론트가 그대로 보여줄 수 있다. 예전에는 두 레인을 "preprocess_extract_media"
    # 하나로 묶어 보고해서 둘 중 하나만 끝나도 둘 다 끝난 것처럼 보였다.
    notify_stage("preprocess_slide_extract", "run")
    notify_stage("preprocess_audio_quality", "run")

    if args.skip_extract:
        helpers.log.info("P1A extract_slides — 슬라이드 추출 건너뜀 (--skip-extract)")
        meta_path = str(paths["metadata"])
        timings["P1A extract_slides — 슬라이드 추출"] = 0.0
        notify_stage("preprocess_slide_extract", "done")
        audio_analyze_result = helpers.analyze_audio_quality(args, output_dir)
        timings["P1B analyze_audio_quality — 오디오 품질 분석"] = audio_analyze_result["elapsed"]
        notify_stage("preprocess_audio_quality", "done")
    else:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_1a = executor.submit(helpers.extract_slides, args, slides_dir, output_dir)
            future_1b = executor.submit(helpers.analyze_audio_quality, args, output_dir)
            for future in as_completed([future_1a, future_1b]):
                if future is future_1a:
                    r1 = future.result()
                    meta_path = r1["meta_path"]
                    timings["P1A extract_slides — 슬라이드 추출"] = r1["elapsed"]
                    notify_stage("preprocess_slide_extract", "done")
                else:
                    audio_analyze_result = future.result()
                    timings["P1B analyze_audio_quality — 오디오 품질 분석"] = audio_analyze_result["elapsed"]
                    notify_stage("preprocess_audio_quality", "done")

    duration = audio_analyze_result.get("duration", 0.0)
    timings["P1 extract_media total — 슬라이드 추출 + 오디오 품질 분석 총합"] = time.time() - t_parallel

    print(f"\n  ✓ P1 extract_media 완료 — 슬라이드 추출 + 오디오 품질 분석  ({timings['P1 extract_media total — 슬라이드 추출 + 오디오 품질 분석 총합']:.1f}초)")
    print("─" * 70)

    helpers._banner("P2 textualize_transcribe — 슬라이드 텍스트화 + 전체 전사")
    t_parallel = time.time()
    transcript_result: dict = {}

    # 다이어그램은 영상/오디오 레인 각각 2단계(슬라이드 추출→분석, 오디오 품질 분석→
    # 음성 전사)만 갖고 있어서, 슬라이드 쪽은 textualize_slides 하나로 "슬라이드 분석"이
    # 끝나지만 오디오 쪽은 transcribe_audio(P2B)에 이어 P3 process_audio(오디오 맥락
    # 후처리)까지 마쳐야 "음성 전사"가 끝난 것으로 본다 — process_audio는 원래도 오디오
    # 세그먼트/장면 구조를 만드는 오디오 전용 후처리라 별도 노드를 새로 두지 않고 음성
    # 전사 단계에 그대로 묶었다. 그래서 preprocess_audio_transcribe는 P2B가 끝나도 바로
    # done 처리하지 않고 P3까지 끝난 뒤에 done을 보고한다.
    notify_stage("preprocess_slide_analyze", "run")
    notify_stage("preprocess_audio_transcribe", "run")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_2a = executor.submit(helpers.textualize_slides, args, slides_dir, output_dir)
        future_2b = executor.submit(helpers.transcribe_audio, args, meta_path, duration, output_dir)
        for future in as_completed([future_2a, future_2b]):
            if future is future_2a:
                r2 = future.result()
                textualized_path = r2["textualized_path"]
                timings["P2A textualize_slides — 슬라이드 텍스트화"] = r2["elapsed"]
                notify_stage("preprocess_slide_analyze", "done")
            else:
                transcript_result = future.result()
                timings["P2B transcribe_audio — 전체 전사"] = transcript_result["elapsed"]

    timings["P2 textualize_transcribe total — 텍스트화 + 전사 총합"] = time.time() - t_parallel
    print(f"\n  ✓ P2 textualize_transcribe 완료 — 슬라이드 텍스트화 + 전체 전사  ({timings['P2 textualize_transcribe total — 텍스트화 + 전사 총합']:.1f}초)")
    print("─" * 70)

    helpers._banner("P3 build_audio_context — 오디오 context 후처리")
    t_parallel = time.time()

    transcript_raw_path = transcript_result.get(
        "transcript_raw_path",
        str(output_dir / f"{stem}_transcript_raw.json"),
    )

    audio_result = helpers.process_audio(
        args,
        meta_path,
        textualized_path,
        duration,
        output_dir,
        transcript_raw_path,
    )
    timings["P3 process_audio — 오디오 context 후처리"] = audio_result.get("elapsed", 0.0)

    timings["P3 build_audio_context total — 오디오 context 후처리 총합"] = time.time() - t_parallel
    notify_stage("preprocess_audio_transcribe", "done")

    print(f"\n  ✓ P3 build_audio_context 완료 — 오디오 context 후처리  ({timings['P3 build_audio_context total — 오디오 context 후처리 총합']:.1f}초)")
    print("─" * 70)

    preprocess_result = {
        "meta_path": meta_path,
        "duration": duration,
        "textualized_path": textualized_path,
        "transcript_result": transcript_result,
        "transcript_raw_path": transcript_raw_path,
        "audio_result": audio_result,
    }
    save_preprocess_manifest(stem, output_dir, preprocess_result, helpers=helpers)
    return preprocess_result


def save_preprocess_manifest(stem: str, output_dir: Path, preprocess_result: dict, *, helpers) -> Path:
    """Persist the in-memory preprocess payload so graph_upload can resume later."""
    audio_result = preprocess_result.get("audio_result") or {}
    transcript_result = preprocess_result.get("transcript_result") or {}
    manifest = {
        "schema_version": 1,
        "stem": stem,
        "meta_path": preprocess_result.get("meta_path"),
        "duration": preprocess_result.get("duration"),
        "textualized_path": preprocess_result.get("textualized_path"),
        "transcript_result": {
            "transcript_raw_path": transcript_result.get("transcript_raw_path"),
        },
        "transcript_raw_path": preprocess_result.get("transcript_raw_path"),
        "audio_result": {
            "segments_path": audio_result.get("segments_path"),
            "silences_path": audio_result.get("silences_path"),
            "annotated_segments": audio_result.get("annotated_segments") or [],
            "annotated_groups": audio_result.get("annotated_groups") or [],
            "scenes_structure": audio_result.get("scenes_structure"),
            "slide_ranges": audio_result.get("slide_ranges") or [],
            "duration": audio_result.get("duration", preprocess_result.get("duration", 0.0)),
        },
    }
    manifest_path = output_dir / f"{stem}_preprocess_result.json"
    helpers._save_json(manifest_path, manifest)
    return manifest_path


def load_preprocess_result_from_outputs(stem: str, output_dir: Path, paths: dict) -> dict:
    """Restore preprocess_result from the manifest written by run_preprocess_pipeline."""
    manifest_path = output_dir / f"{stem}_preprocess_result.json"
    if not manifest_path.exists() or manifest_path.stat().st_size <= 0:
        raise FileNotFoundError(
            f"graph_upload 실행에 필요한 preprocess manifest가 없습니다: {manifest_path}"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    audio_result = manifest.get("audio_result") or {}
    transcript_result = manifest.get("transcript_result") or {}

    restored = {
        "meta_path": manifest.get("meta_path") or str(paths["metadata"]),
        "duration": manifest.get("duration", audio_result.get("duration", 0.0)),
        "textualized_path": manifest.get("textualized_path") or str(paths["textualized"]),
        "transcript_result": {
            "transcript_raw_path": (
                transcript_result.get("transcript_raw_path")
                or manifest.get("transcript_raw_path")
                or str(output_dir / f"{stem}_transcript_raw.json")
            ),
            "elapsed": 0.0,
        },
        "transcript_raw_path": (
            manifest.get("transcript_raw_path")
            or transcript_result.get("transcript_raw_path")
            or str(output_dir / f"{stem}_transcript_raw.json")
        ),
        "audio_result": {
            "segments_path": audio_result.get("segments_path") or str(paths["segments"]),
            "silences_path": audio_result.get("silences_path") or str(paths["silences"]),
            "annotated_segments": audio_result.get("annotated_segments") or [],
            "annotated_groups": audio_result.get("annotated_groups") or [],
            "scenes_structure": audio_result.get("scenes_structure"),
            "slide_ranges": audio_result.get("slide_ranges") or [],
            "duration": audio_result.get("duration", manifest.get("duration", 0.0)),
        },
    }

    required_paths = [
        ("metadata", restored["meta_path"]),
        ("textualized", restored["textualized_path"]),
        ("segments", restored["audio_result"]["segments_path"]),
        ("silences", restored["audio_result"]["silences_path"]),
    ]
    missing = [f"{label}: {path}" for label, path in required_paths if not path or not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "graph_upload 실행에 필요한 preprocess 산출물이 없습니다:\n" + "\n".join(missing)
        )
    return restored
