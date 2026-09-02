# VLVerifier 파이프라인 CLI 진입점
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import main as pipeline_main
from pipeline.logging_utils import pipeline_log_context


# run-video 서브커맨드 인자를 pipeline.main의 argparse 인자로 변환
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


# 영상 전체 파이프라인(전처리+검증) 실행
def run_video(args):
    with pipeline_log_context(args.output):
        pipeline_main.run_pipeline(_pipeline_args(args.input, args.output, args.title))


# merged_clean.json으로부터 검증 단계만 실행
def run_verify(args):
    from pipeline.verifier.run_all import run_classified_issue_pipeline

    with pipeline_log_context(args.output):
        run_classified_issue_pipeline(args.merged_clean, output_dir=args.output)


# run-video/run-verify 서브커맨드를 갖는 argparse 파서 구성
def build_parser():
    parser = argparse.ArgumentParser(description='VLVerifier 파이프라인 CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_video_parser = subparsers.add_parser('run-video', help='영상 전체 검증 파이프라인 실행')
    run_video_parser.add_argument('--input', required=True, help='입력 영상 경로')
    run_video_parser.add_argument('--output', required=True, help='출력 디렉터리')
    run_video_parser.add_argument('--title', default='', help='강의 제목')
    run_video_parser.set_defaults(func=run_video)

    run_verify_parser = subparsers.add_parser('run-verify', help='merged_clean.json으로 verifier 실행')
    run_verify_parser.add_argument('--merged-clean', required=True, help='merged_clean.json 경로')
    run_verify_parser.add_argument('--output', required=True, help='analyzer 출력 디렉터리')
    run_verify_parser.set_defaults(func=run_verify)
    return parser


# CLI 진입점, 파싱된 서브커맨드 핸들러 실행
def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
