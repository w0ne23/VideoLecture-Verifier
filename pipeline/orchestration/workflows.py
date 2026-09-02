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


class _TimingsDict(dict):
    """각 stage 함수는 캐시된 산출물을 발견해 건너뛸 때 elapsed=0.0을 돌려준다.
    run_pipeline이 실행마다 기존 timings.json 값을 이어받게 되면서, 이 0.0을
    그대로 덮어쓰면 실제로 그 stage가 돌았을 때 기록된 소요 시간이 스킵될 때마다
    지워진다. 새 값이 0.0이고 기존에 0이 아닌 값이 있으면 갱신을 무시해서, 실제로
    다시 실행된 stage의 시간만 갱신되고 스킵된 stage는 마지막 실측치를 유지한다."""

    def __setitem__(self, key, value):
        if value == 0.0 and self.get(key, 0.0) != 0.0:
            return
        super().__setitem__(key, value)


def run_pipeline(args, progress_callback=None, *, helpers):
    total_start = time.time()

    def notify_stage(stage_key, status, progress=None):
        if progress_callback:
            try:
                progress_callback(stage_key, status, progress)
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

    # verify_only처럼 이전 실행의 산출물을 재사용하는 job은 전처리를 다시 태우지
    # 않는다. 그렇다고 이전 실행이 기록해 둔 전처리 소요 시간까지 지우면 안 되므로,
    # 기존 timings.json이 있으면 그 내용을 이어받아 이번 실행에서 갱신되는 stage만
    # 덮어쓰고 나머지는 그대로 보존한다. run_history에는 실행할 때마다(예: 전처리
    # 1회 + verify_only 1회) 각각의 소요 시간이 항목으로 쌓인다.
    existing_payload: dict = {}
    if timing_path.exists():
        try:
            existing_payload = json.loads(timing_path.read_text(encoding='utf-8'))
        except Exception:
            existing_payload = {}

    started_at_epoch = existing_payload.get('started_at_epoch', total_start)
    timings: dict[str, float] = _TimingsDict(existing_payload.get('timings') or {})
    stage_status: dict[str, str] = dict(existing_payload.get('stage_status') or {})
    run_history: list[dict] = list(existing_payload.get('run_history') or [])

    def write_timings(current_stage: str | None = None) -> None:
        now = time.time()
        payload = {
            'stem': stem,
            'status': 'done' if current_stage == 'pipeline_done' else 'running',
            'current_stage': current_stage,
            'started_at_epoch': started_at_epoch,
            'updated_at_epoch': now,
            'elapsed_total_sec': now - total_start,
            'timings': timings,
            'stage_status': stage_status,
            'run_history': run_history,
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
        raise RuntimeError(f'VLVerifier supports verify workflow only, got: {job_type}')

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
        preprocess_result.get('textualized_path', ''),
        r9.get('merged_clean_path', str(output_dir / f'{stem}_analyzer' / f'{stem}_merged_clean.json')),
        r10.get('claim_output', ''),
        r10.get('claim_report', ''),
    ]
    _print_generated_files(output_files)
    run_history.append({
        'job_type': job_type,
        'started_at_epoch': total_start,
        'completed_at_epoch': time.time(),
        'elapsed_sec': time.time() - total_start,
    })
    write_timings('pipeline_done')
