"""
강조 감지 결과 통합

오디오(emphasis_audio.py)와 키워드(emphasis_keyword.py) 감지 결과를
세그먼트 단위로 합산해 최종 강조 구간을 생성한다.
"""


def _collect_keywords_from_det(det: dict) -> tuple[list[str], list[str], str]:
    """
    감지 결과에서 키워드 목록과 방법 이름 추출.
    반환: (topic_keywords, importance_keywords, method_name)
    """
    method = det.get("detection_method", "")
    topic_keywords: list[str] = []
    importance_keywords: list[str] = []
    if method == "topic_keyword_repeat" and det.get("repeated_topic_keywords"):
        topic_keywords = list(det["repeated_topic_keywords"])
    if method == "keyword_weighted" and det.get("matched_keywords"):
        importance_keywords = list(det["matched_keywords"])
    return topic_keywords, importance_keywords, method


def combine_emphasis_simple(
    audio_emphasis: list[dict],
    keyword_emphasis: list[dict],
    segments: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    오디오 feature 결과 + 키워드 feature 결과를 세그먼트 단위로 통합.

    반환:
    - annotated_segments: 원본 segments에 audio_emphasis / 점수 산출용 필드가 추가된 리스트
    - emphasis_only_sections: legacy 호환용으로 detected=True인 구간만 모은 리스트
    """
    segment_emphasis_info = {}

    for seg in segments:
        start = seg['start']
        segment_emphasis_info[start] = {
            'segment': seg,
            'methods': [],
            'keywords_by_method': {},
            'topic_keywords': [],
            'importance_keywords': [],
            'audio_topic_total_count_sum': 0,
            'std_based': {
                'std_based_score': 0.0,
                'audio_signal_score': 0,
                'volume_score': 0.0,
                'pitch_score': 0.0,
                'volume_metric': 0.0,
                'pitch_metric': 0.0,
                'volume_rank': None,
                'pitch_rank': None,
                'volume_ratio': 0.0,
                'pitch_variation': 0.0,
            },
            'importance_keyword': {
                'score': 0,
                'category_scores': {'exam': 0, 'strong': 0, 'summary': 0},
                'matched_categories': [],
                'matched_keywords': {'exam': [], 'strong': [], 'summary': []},
            },
            'topic': {
                'keywords': [],
                'keyword_scores': {},
                'topic_keyword_score': 0,
                'audio_topic_total_count_sum': 0,
            },
            'detected_methods': [],
        }

    for det in audio_emphasis + keyword_emphasis:
        start = det['start']
        if start not in segment_emphasis_info:
            continue
        info = segment_emphasis_info[start]
        method = det.get('detection_method')
        if method:
            info['methods'].append(method)
            if det.get('detected') and method not in info['detected_methods']:
                info['detected_methods'].append(method)

        if method == 'std_based':
            info['std_based'] = {
                'std_based_score': float(det.get('std_based_score', det.get('emphasis_score', 0.0)) or 0.0),
                'audio_signal_score': int(det.get('audio_signal_score', det.get('emphasis_score', 0)) or 0),
                'volume_score': float(det.get('volume_score', 0.0) or 0.0),
                'pitch_score': float(det.get('pitch_score', 0.0) or 0.0),
                'volume_metric': float(det.get('volume_metric', 0.0) or 0.0),
                'pitch_metric': float(det.get('pitch_metric', 0.0) or 0.0),
                'volume_rank': det.get('volume_rank'),
                'pitch_rank': det.get('pitch_rank'),
                'volume_ratio': float(det.get('volume_ratio', 0.0) or 0.0),
                'pitch_variation': float(det.get('pitch_variation', 0.0) or 0.0),
            }

        if method == 'keyword_weighted':
            matched_by_category = det.get('matched_keywords_by_category') or {}
            info['importance_keyword'] = {
                'score': int(det.get('importance_keyword_score', det.get('emphasis_score', 0)) or 0),
                'category_scores': det.get('category_scores') or {'exam': 0, 'strong': 0, 'summary': 0},
                'matched_categories': det.get('matched_categories') or [],
                'matched_keywords': {
                    'exam': list(matched_by_category.get('exam', [])),
                    'strong': list(matched_by_category.get('strong', [])),
                    'summary': list(matched_by_category.get('summary', [])),
                },
            }

        topic_kws, importance_kws, method = _collect_keywords_from_det(det)
        if method:
            if topic_kws:
                topic_keyword_scores = {
                    str(k): int(v)
                    for k, v in (det.get('repeated_topic_keyword_scores') or {}).items()
                }
                info['keywords_by_method'][method] = topic_kws
                info['topic_keywords'].extend(topic_kws)
                info['audio_topic_total_count_sum'] = max(
                    int(info.get('audio_topic_total_count_sum', 0) or 0),
                    int(det.get('audio_topic_total_count_sum', 0) or 0),
                )
                info['topic'] = {
                    'keywords': topic_kws,
                    'keyword_scores': topic_keyword_scores,
                    'topic_keyword_score': int(det.get('audio_topic_keyword_score', 0) or 0),
                    'audio_topic_total_count_sum': int(info.get('audio_topic_total_count_sum', 0) or 0),
                }
            if importance_kws:
                info['keywords_by_method'][method] = importance_kws
                info['importance_keywords'].extend(importance_kws)

    def _uniq(seq: list[str]) -> list[str]:
        return list(dict.fromkeys(seq))

    annotated_segments = []
    emphasis_only_sections = []

    for start, info in segment_emphasis_info.items():
        seg = info['segment'].copy()

        kw_by_method = info.get('keywords_by_method') or {}

        topic_keywords = _uniq(info.get('topic_keywords', []))
        importance_keywords = _uniq(info.get('importance_keywords', []))
        merged_keywords = topic_keywords + [k for k in importance_keywords if k not in set(topic_keywords)]
        audio_emphasis = {
            'audio': info['std_based'],
            'importance_keywords': info['importance_keyword'],
            'topic': {
                'keywords': topic_keywords,
                'keyword_scores': {
                    str(k): int(v)
                    for k, v in (info.get('topic') or {}).get('keyword_scores', {}).items()
                    if k in topic_keywords
                },
                'topic_keyword_score': int((info.get('topic') or {}).get('topic_keyword_score', 0) or 0),
                'audio_topic_total_count_sum': int(info.get('audio_topic_total_count_sum', 0) or 0),
            },
        }
        detected = bool(info.get('detected_methods'))

        seg['audio_emphasis'] = audio_emphasis
        seg['emphasis_methods'] = list(dict.fromkeys(info.get('detected_methods', [])))
        seg['detection_count'] = len(seg['emphasis_methods'])

        if merged_keywords:
            seg['emphasis_keywords'] = merged_keywords
            seg['emphasis_keywords_by_method'] = kw_by_method

        seg['emphasis_detail'] = {
            **audio_emphasis,
            'keywords': {
                'topic_keywords': topic_keywords,
                'importance_keywords': importance_keywords,
                'all_keywords': merged_keywords,
                'audio_topic_total_count_sum': int(info.get('audio_topic_total_count_sum', 0) or 0),
                'by_method': kw_by_method,
            },
            'methods': seg['emphasis_methods'],
            'detection_count': seg['detection_count'],
        }

        if detected:
            emphasis_only_sections.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'],
                'methods': seg['emphasis_methods'],
                'detection_count': seg['detection_count'],
                'keywords': merged_keywords,
                'keywords_by_method': kw_by_method,
                'audio_emphasis': audio_emphasis,
                'emphasis_detail': seg['emphasis_detail'],
            })

        annotated_segments.append(seg)

    emphasis_only_sections.sort(
        key=lambda x: (
            x.get('audio_emphasis', {}).get('topic', {}).get('audio_topic_total_count_sum', 0),
            x.get('audio_emphasis', {}).get('topic', {}).get('topic_keyword_score', 0),
            x.get('audio_emphasis', {}).get('audio', {}).get('audio_signal_score', 0),
            x.get('audio_emphasis', {}).get('importance_keywords', {}).get('score', 0),
        ),
        reverse=True,
    )
    return annotated_segments, emphasis_only_sections
