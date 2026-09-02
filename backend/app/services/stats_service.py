"""통계 페이지용 데이터 계층

- `record_verification_stats`: 완료된 verify 실행에서 요약을 뽑아 `verification_stats`
  테이블에 1행으로 적재 (worker 완료 hook + 백필 스크립트가 호출)
- 집계 API(`aggregate`)는 Phase 2에서 추가 예정

원본은 여전히 디스크의 `..._verification_final.json` / `pipeline_timings.json` 등이고
이 모듈은 그것을 조회 가능한 형태로 투영
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import VerificationStats
from app.services.lecture_service import _verification_result_candidates
from app.services.storage_service import resolve_storage_path

logger = logging.getLogger(__name__)

# 프론트(statsConfig.ISSUE_TYPES)와 동일한 5개 지식 오류 유형
# composite_issue = 슬라이드 오류, 지식 오류에 포함
KNOWN_ISSUE_TYPES = (
    'factual_error',
    'temporal_error',
    'scope_overclaim',
    'confusing_explanation',
    'composite_issue',
)

# verification_final.json 의 feedback status -> 통계 버킷
_STATUS_BUCKET = {
    'confirmed': 'confirmed',
    'professor_check': 'review',
    'rejected': 'rejected',
}


# JSON 파일 로드, 읽기/파싱 실패 시 None
def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


# 값을 int로 안전 변환, 실패 시 0
def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# bool을 제외한 int/float 값만 통과, 아니면 None
def _num(value: Any) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _split_preprocess_verify(timings_doc: dict) -> tuple[float | None, float | None, float | None]:
    """pipeline_timings.json에서 전처리/검증/총 소요 시간(초) 추출

    파이프라인은 전처리 → 검증 순차 실행이지만, pipeline_timings.json은 스킵된 스테이지의
    이전 실행 시간을 보존 ("이 강의 전처리는 원래 N초 걸린다"를 계속 보여주기 위함)
    따라서 `총 − 전처리`로 검증 시간을 구하면 재검증 시 0이 되므로, 대신 각 단계의
    스테이지 시간을 직접 합산

    - 전처리: `P{n} ... total` 롤업 키 합 (하위 P1A/P1B와 이중 카운트 방지)
    - 검증: `V1` + `V2A~V2F*` 스테이지 시간 합, 검증 내부는 병렬이라 wall-clock보다
      약간 크게 나올 수 있으나 "검증 시간은 강의 길이와 무관"이라는 뷰의 취지에는 부합
      (`V2 run_verifier` 롤업은 stale 될 수 있어 폴백으로만 사용)
    - 총합: run_history의 마지막 실행 elapsed_sec 우선(진짜 wall-clock), 없으면
      elapsed_total_sec, 그마저 없으면 전처리+검증
    """
    timings = timings_doc.get('timings') or {}
    if not isinstance(timings, dict):
        timings = {}

    pre = sum(
        v for k, v in timings.items()
        if isinstance(k, str) and re.match(r'P\d+ .*\btotal\b', k) and _num(v) is not None
    )

    # V1 build_analyzer_input + V2A/V2B/.../V2F1/V2F2 (스킵되는 'V2 run_verifier',
    # 'V2 verifier 백그라운드 시작' 은 제외 — 이들은 'V2' 뒤에 공백).
    ver = sum(
        v for k, v in timings.items()
        if isinstance(k, str) and re.match(r'V(1 |2[A-Z])', k) and _num(v) is not None
    )
    if not ver:
        ver = next(
            (v for k, v in timings.items()
             if isinstance(k, str) and k.startswith('V2 run_verifier') and _num(v) is not None),
            0.0,
        )

    total = None
    for run in reversed(timings_doc.get('run_history') or []):
        if _num(run.get('elapsed_sec')):
            total = run['elapsed_sec']
            break
    if total is None:
        total = _num(timings_doc.get('elapsed_total_sec'))
    if total is None:
        total = pre + ver

    return (
        round(pre, 3) if pre else None,
        round(ver, 3) if ver else None,
        round(total, 3) if total else None,
    )


def _breakdown_by_type(feedback_items: list, slide_summary: dict, slide_needs_review: list) -> dict:
    """{ confirmed: {type: n}, review: {...}, rejected: {...} } 형태로 집계

    슬라이드 오류는 composite_issue 유형으로 합산 (reportable=확정, needs_review=교수확인)
    """
    buckets: dict[str, dict[str, int]] = {'confirmed': {}, 'review': {}, 'rejected': {}}

    for item in feedback_items or []:
        bucket = _STATUS_BUCKET.get(item.get('status'))
        ftype = item.get('feedback_type')
        if not bucket or ftype not in KNOWN_ISSUE_TYPES:
            continue
        buckets[bucket][ftype] = buckets[bucket].get(ftype, 0) + 1

    reportable = _safe_int((slide_summary or {}).get('reportable_error_count'))
    if reportable:
        buckets['confirmed']['composite_issue'] = buckets['confirmed'].get('composite_issue', 0) + reportable
    review_slides = len(slide_needs_review or [])
    if review_slides:
        buckets['review']['composite_issue'] = buckets['review'].get('composite_issue', 0) + review_slides

    return buckets


def extract_stats(lecture, job_id) -> dict | None:
    """lecture의 디스크 산출물에서 verification_stats 행 kwargs 생성

    결과 JSON이 없으면 None (호출부에서 skip)
    """
    stem = str(lecture.id)
    output_dir = resolve_storage_path(lecture.output_dir)
    if not output_dir:
        return None
    output_dir = Path(output_dir)
    analyzer_dir = output_dir / f'{stem}_analyzer'

    candidates = _verification_result_candidates(analyzer_dir, output_dir, stem)
    vpath = next((p for p in candidates if p.exists()), None)
    if not vpath:
        return None
    vf = _load_json(vpath)
    if not isinstance(vf, dict):
        return None

    summary = vf.get('summary') or {}
    slide_summary = vf.get('slide_error_summary') or {}
    slide_needs_review = vf.get('slide_error_needs_review') or []
    breakdown = _breakdown_by_type(vf.get('feedback_items') or [], slide_summary, slide_needs_review)

    confirmed = sum(breakdown['confirmed'].values())
    review = sum(breakdown['review'].values())
    rejected = sum(breakdown['rejected'].values())

    merged = _load_json(analyzer_dir / f'{stem}_merged_clean.json') or {}
    domain = str(merged.get('domain') or merged.get('primary_domain') or '').strip() or 'etc'
    sub_domain = str(merged.get('subdomain') or merged.get('sub_domain') or '').strip()

    preprocess_doc = _load_json(output_dir / f'{stem}_preprocess_result.json') or {}
    duration = _num(preprocess_doc.get('duration'))

    timings_doc = _load_json(output_dir / 'pipeline_timings.json') or {}
    pre_sec, ver_sec, total_sec = _split_preprocess_verify(timings_doc)

    return {
        'lecture_id': lecture.id,
        'job_id': job_id,
        'source_tag': (lecture.source_tag or 'etc').strip().lower() or 'etc',
        'domain': domain,
        'sub_domain': sub_domain,
        'video_duration_sec': float(duration) if duration is not None else None,
        'preprocess_sec': pre_sec,
        'verify_sec': ver_sec,
        'total_sec': total_sec,
        'confirmed_count': confirmed,
        'review_count': review,
        'rejected_count': rejected,
        'slide_error_count': _safe_int(
            summary.get('slide_error_count') or slide_summary.get('reportable_error_count')
        ),
        'breakdown_by_type': breakdown,
        'verifier_models': vf.get('models') or vf.get('verifier_source_models') or [],
        'verifier_version': _safe_int(os.getenv('VERIFIER_VERSION')) or None,
        'verification_date': _parse_dt(vf.get('verification_date')),
    }


# ISO 8601 문자열을 datetime으로 파싱, 실패/비문자열이면 None
def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


async def record_verification_stats(db: AsyncSession, lecture_id, job_id) -> bool:
    """완료된 verify 실행 1건을 verification_stats에 적재 (1강의 1행, upsert 성격)

    호출부(worker hook / 백필)에서 job_type == 'verify'인지 이미 걸렀다고 가정
    통계 적재 실패가 상위 흐름을 막지 않도록 예외는 호출부에서 처리
    """
    from app.models import Lecture

    lecture = await db.get(Lecture, lecture_id if not isinstance(lecture_id, str) else _to_uuid(lecture_id))
    if not lecture:
        return False

    data = extract_stats(lecture, job_id)
    if not data:
        logger.info('verification_stats: 결과 파일 없음, skip (lecture=%s)', lecture_id)
        return False

    await db.execute(
        text('DELETE FROM verification_stats WHERE lecture_id = :lid'),
        {'lid': str(lecture.id)},
    )
    db.add(VerificationStats(**data))
    return True


# 문자열을 UUID로 변환
def _to_uuid(value: str):
    import uuid
    return uuid.UUID(value)


# ---------------------------------------------------------------------------
# 집계 (통계 페이지 GET /stats)
# ---------------------------------------------------------------------------

# 파이프라인이 분류하는 도메인 키, 이 목록 밖이거나 비면 'etc'(기타)로 묶음
KNOWN_DOMAINS = (
    'engineering',
    'natural_science',
    'humanities',
    'social_science',
    'education',
    'medicine',
    'arts_sports',
)

_DURATION_BUCKET_SEC = 15 * 60  # 15분 단위


# 알려진 이슈 유형 전부 0으로 초기화한 분포 dict
def _empty_type_dist() -> dict:
    return {t: 0 for t in KNOWN_ISSUE_TYPES}


def _reported_type_dist(breakdown: dict) -> dict:
    """confirmed + review만 합산 (기각 제외), 슬라이드 오류는 이미 composite_issue로 포함"""
    dist = _empty_type_dist()
    for bucket in ('confirmed', 'review'):
        for ftype, n in (breakdown or {}).get(bucket, {}).items():
            if ftype in dist:
                dist[ftype] += _safe_int(n)
    return dist


# other의 값을 target에 누적 합산
def _add_dist(target: dict, other: dict) -> None:
    for k, v in other.items():
        target[k] = target.get(k, 0) + v


# 숫자 값들의 평균, 값이 없으면 None
def _mean(values: list) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 2) if nums else None


async def aggregate(db: AsyncSession) -> dict:
    """통계 페이지가 쓰는 3개 뷰(태그별/도메인별/영상 길이별)로 집계

    verification_stats에는 job_type='verify' 완료 건만 들어있음 (verify_only 제외)
    """
    from sqlalchemy import select

    rows = (await db.execute(select(VerificationStats))).scalars().all()

    by_tag: dict[str, dict] = {}
    by_domain: dict[str, dict] = {}
    duration_buckets: dict[int, dict] = {}

    for r in rows:
        rdist = _reported_type_dist(r.breakdown_by_type)

        tag = (r.source_tag or 'etc').strip().lower() or 'etc'
        slot = by_tag.setdefault(tag, {'key': tag, 'typeDist': _empty_type_dist(), 'lectureCount': 0})
        _add_dist(slot['typeDist'], rdist)
        slot['lectureCount'] += 1

        domain = r.domain if r.domain in KNOWN_DOMAINS else 'etc'
        dslot = by_domain.setdefault(domain, {'key': domain, 'typeDist': _empty_type_dist(), 'lectureCount': 0})
        _add_dist(dslot['typeDist'], rdist)
        dslot['lectureCount'] += 1

        if isinstance(r.video_duration_sec, (int, float)) and r.video_duration_sec > 0:
            idx = int(r.video_duration_sec // _DURATION_BUCKET_SEC)
            b = duration_buckets.setdefault(
                idx, {'lecture_sec': [], 'preprocess_sec': [], 'verify_sec': []}
            )
            b['lecture_sec'].append(r.video_duration_sec)
            b['preprocess_sec'].append(r.preprocess_sec)
            b['verify_sec'].append(r.verify_sec)

    def _finish(slot: dict) -> dict:
        slot['total'] = sum(slot['typeDist'].values())
        return slot

    by_duration = []
    for idx in sorted(duration_buckets):
        b = duration_buckets[idx]
        lo, hi = idx * 15, (idx + 1) * 15
        pre_min = round((_mean(b['preprocess_sec']) or 0) / 60, 1)
        ver_min = round((_mean(b['verify_sec']) or 0) / 60, 1)
        by_duration.append({
            'key': f'{lo}_{hi}',
            'label': f'{lo}–{hi}분',
            'lectureCount': len(b['lecture_sec']),
            'lectureMin': round((_mean(b['lecture_sec']) or 0) / 60, 1),
            'preprocessMin': pre_min,
            'verifyMin': ver_min,
            'total': round(pre_min + ver_min, 1),
        })

    return {
        'lecture_count': len(rows),
        'by_tag': [_finish(v) for v in by_tag.values()],
        'by_domain': [_finish(v) for v in by_domain.values()],
        'by_duration': by_duration,
    }
