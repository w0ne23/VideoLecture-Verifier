import json
import time
from pathlib import Path

from .preprocess import run_preprocess_pipeline
from .verifier import run_verifier_pipeline


def _print_generated_files(output_files: list[str]) -> None:
    print('\n  생성된 파일:')
    for path_str in output_files:
        if not path_str:
            continue
        path = Path(path_str)
        print(f"    {'✓' if path.exists() else '✗'}  {path}")


def run_pipeline(args, progress_callback=None, *, helpers):
    total_start = time.time()
    timings: dict[str, float] = {}
    stage_status: dict[str, str] = {}

    def notify_stage(stage_key, status):
        if progress_callback:
            try:
                progress_callback(stage_key, status)
            except Exception as exc:
                helpers.log.warning('progress_callback failed for %s: %s', stage_key, exc)

    from ..config import output_paths

    stem = Path(args.input).stem
    slides_dir = Path(args.slides)
    output_dir = Path(args.output)
    job_type = helpers._normalize_pipeline_job_type(getattr(args, 'job_type', None))
    output_dir.mkdir(parents=True, exist_ok=True)
    slides_dir.mkdir(parents=True, exist_ok=True)
    timing_path = output_dir / 'pipeline_timings.json'

    def write_timings(current_stage: str | None = None) -> None:
        payload = {
            'stem': stem,
            'status': 'done' if current_stage == 'pipeline_done' else 'running',
            'current_stage': current_stage,
            'started_at_epoch': total_start,
            'updated_at_epoch': time.time(),
            'elapsed_total_sec': time.time() - total_start,
            'timings': timings,
            'stage_status': stage_status,
        }
        tmp_path = timing_path.with_suffix('.json.tmp')
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp_path.replace(timing_path)

    write_timings('pipeline_start')
    paths = output_paths(stem, output_dir, slides_dir)

    if job_type not in {
        helpers.JOB_TYPE_VERIFY,
        helpers.JOB_TYPE_VERIFY_ONLY,
        helpers.JOB_TYPE_VERIFIED_UPLOAD,
        helpers.JOB_TYPE_LEGACY_FULL,
    }:
        raise RuntimeError(f'VeriLec supports verify workflow only, got: {job_type}')

    if job_type == helpers.JOB_TYPE_VERIFY_ONLY:
        # 전처리 산출물이 이미 있다는 전제 하에, 재전처리 없이 이전 실행이 남긴
        # manifest(`{stem}_preprocess_result.json`)에서 preprocess_result를 복원한다.
        # merged_clean.json을 손으로 편집해 검증만 다시 태우고 싶을 때(예: 오류 주입
        # 테스트) 전처리 API 비용을 반복하지 않기 위한 경로다.
        preprocess_result = helpers.load_preprocess_result_from_outputs(stem, output_dir, paths)
    else:
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

    verifier_result = run_verifier_pipeline(
        args,
        preprocess_result=preprocess_result,
        output_dir=output_dir,
        paths=paths,
        timings=timings,
        background=False,
        notify_stage=notify_stage,
        helpers=helpers,
    )

    r9 = verifier_result.get('analyzer_input', {})
    r10 = verifier_result.get('verifier_result', {})
    audio_result = preprocess_result.get('audio_result', {})
    output_files = [
        audio_result.get('segments_path', ''),
        audio_result.get('silences_path', ''),
        preprocess_result.get('textualized_path', ''),
        r9.get('merged_clean_path', str(output_dir / f'{stem}_analyzer' / f'{stem}_merged_clean.json')),
        r10.get('claim_output', ''),
        r10.get('claim_report', ''),
    ]
    _print_generated_files(output_files)
    write_timings('pipeline_done')
