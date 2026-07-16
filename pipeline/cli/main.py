from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import main as pipeline_main


def _pipeline_args(input_path: str, output_dir: str, title: str = ''):
    output = Path(output_dir)
    slides = output / 'slides'
    args = pipeline_main.get_parser().parse_args([
        '--input', input_path,
        '--output', str(output),
        '--slides', str(slides),
        *(['--title', title] if title else []),
    ])
    args.job_type = pipeline_main.JOB_TYPE_VERIFY
    return args


def run_video(args):
    pipeline_main.run_pipeline(_pipeline_args(args.input, args.output, args.title))


def run_verify(args):
    from pipeline.verifier.run_all import run_classified_issue_pipeline

    run_classified_issue_pipeline(args.merged_clean, output_dir=args.output)


def run_text_processor(args):
    """Re-run only transcript correction from existing preprocessing artifacts."""
    from pipeline.preprocess.text_processor import correct_segments_two_pass

    transcript_path = Path(args.transcript_raw)
    metadata_path = Path(args.metadata)
    textualized_path = Path(args.slide_textualized)
    output_path = Path(args.output)

    with transcript_path.open(encoding="utf-8") as f:
        transcript_payload = json.load(f)
    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)
    with textualized_path.open(encoding="utf-8") as f:
        textualized_data = json.load(f)

    segments_raw = transcript_payload.get("segments", [])
    if not isinstance(segments_raw, list) or not segments_raw:
        raise ValueError(f"전사 구간이 없습니다: {transcript_path}")
    if not isinstance(metadata, list):
        raise ValueError(f"메타데이터 형식이 올바르지 않습니다: {metadata_path}")

    print(f"텍스트 교정 단독 실행: {len(segments_raw)}개 구간")
    corrected = correct_segments_two_pass(
        segments=segments_raw,
        metadata=metadata,
        textualized_data=textualized_data,
        textualized_dir=textualized_path.parent,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "video_path": transcript_payload.get("video_path", ""),
                "segment_count": len(corrected),
                "segments": corrected,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"저장 완료: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(description='VeriLec pipeline CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_video_parser = subparsers.add_parser('run-video', help='run full video verify pipeline')
    run_video_parser.add_argument('--input', required=True, help='input video path')
    run_video_parser.add_argument('--output', required=True, help='output directory')
    run_video_parser.add_argument('--title', default='', help='lecture title')
    run_video_parser.set_defaults(func=run_video)

    run_verify_parser = subparsers.add_parser('run-verify', help='run verifier from merged_clean.json')
    run_verify_parser.add_argument('--merged-clean', required=True, help='merged_clean.json path')
    run_verify_parser.add_argument('--output', required=True, help='analyzer output directory')
    run_verify_parser.set_defaults(func=run_verify)

    text_processor_parser = subparsers.add_parser(
        'run-text-processor',
        help='rerun only transcript correction from existing preprocessing artifacts',
    )
    text_processor_parser.add_argument('--transcript-raw', required=True, help='raw transcript JSON path')
    text_processor_parser.add_argument('--metadata', required=True, help='slide metadata JSON path')
    text_processor_parser.add_argument('--slide-textualized', required=True, help='slide textualized JSON path')
    text_processor_parser.add_argument('--output', required=True, help='new corrected segments JSON path')
    text_processor_parser.set_defaults(func=run_text_processor)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
