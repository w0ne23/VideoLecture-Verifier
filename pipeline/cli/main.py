from __future__ import annotations

import argparse
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
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
