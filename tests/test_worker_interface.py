"""backend 워커 ↔ pipeline CLI 인터페이스 계약 테스트.

worker.py가 넘기는 인자 조합이 get_parser에서 항상 파싱돼야 한다.
파서에서 옵션을 제거할 때 이 테스트가 깨지면 워커도 함께 고쳐야 한다는 신호다.
"""
import pytest

pytest.importorskip('librosa', reason='pipeline.main이 librosa를 import함')


def test_worker_argument_contract():
    import pipeline.main as pipeline_main

    args = pipeline_main.get_parser().parse_args([
        '--input', 'video.mp4',
        '--output', 'out',
        '--slides', 'out/slides',
        '--lecture-id', 'lecture-uuid',
        '--title', '제목',
        '--uploaded-at', '2026-07-07T00:00:00',
    ])
    args.job_type = pipeline_main.JOB_TYPE_VERIFY

    assert args.input == 'video.mp4'
    assert args.job_type == 'verify'
    assert callable(pipeline_main.run_pipeline)
    assert callable(pipeline_main.get_parser)


def test_graph_remnants_are_gone():
    import pipeline.main as pipeline_main

    for name in [
        'classify_slides', 'fuse_preprocessed_data', 'generate_graph_triples',
        'build_lance_index', 'build_graphrag_index', 'generate_metadata',
        'build_recommender_index', 'run_graph_pipeline',
    ]:
        assert not hasattr(pipeline_main, name), f'{name}이(가) 다시 생겼습니다'
