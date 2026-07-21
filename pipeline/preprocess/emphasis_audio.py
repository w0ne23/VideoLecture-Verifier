"""
오디오 신호 기반 강조 구간 감지 (볼륨 + 피치 표준편차)
"""

import numpy as np

try:
    import librosa
except ModuleNotFoundError:
    librosa = None


def _rank_scores(indices_and_values: list[tuple[int, float]]) -> dict[int, dict]:
    """값 내림차순으로 20% 버킷마다 5~1점을 부여한다."""
    ranked = sorted(indices_and_values, key=lambda item: item[1], reverse=True)
    if not ranked:
        return {}
    bucket_size = max(1, int(np.ceil(len(ranked) / 5)))
    result = {}
    for rank, (idx, value) in enumerate(ranked):
        score = max(1, 5 - rank // bucket_size)
        result[idx] = {
            "score": int(score),
            "rank": rank + 1,
            "metric": float(value),
        }
    return result


def detect_emphasis_by_std(y, sr, segments: list[dict]) -> list[dict]:
    """
    평균 초과 볼륨/피치 구간을 각각 랭킹화해 5~1점으로 점수화한다.
    - volume_score: 전체 구간 평균 volume_metric을 넘는 구간 중 상위 20%부터 5~1점
    - pitch_score: 전체 구간 평균 pitch_metric을 넘는 구간 중 상위 20%부터 5~1점
    - audio_signal_score: volume_score + pitch_score
    """
    print("  [오디오 랭킹 기반] 분석 중...")
    if librosa is None:
        print("    -> librosa 없음: 오디오 신호 강조 분석을 건너뜁니다.")
        return [
            {
                'start': seg.get('start', 0),
                'end': seg.get('end', seg.get('start', 0)),
                'text': seg.get('text', ''),
                'emphasis_score': 0.0,
                'std_based_score': 0.0,
                'audio_signal_score': 0,
                'volume_score': 0,
                'pitch_score': 0,
                'volume_metric': 0.0,
                'pitch_metric': 0.0,
                'volume_mean': 0.0,
                'pitch_mean': 0.0,
                'volume_rank': None,
                'pitch_rank': None,
                'volume_ratio': 0.0,
                'pitch_variation': 0.0,
                'detected': False,
                'score_method': 'skipped_no_librosa',
                'detection_method': 'std_based',
            }
            for seg in segments
        ]

    if isinstance(y, (str, bytes)):
        y, sr = librosa.load(y, sr=sr)

    # RMS 에너지 계산
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

    # 피치
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=512)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        pitch_values.append(pitch if pitch > 0 else 0)

    metrics = []
    for idx, seg in enumerate(segments):
        start_time = seg['start']
        end_time = seg['end']

        seg_rms_indices = np.where((rms_times >= start_time) & (rms_times <= end_time))[0]
        if len(seg_rms_indices) == 0:
            metrics.append({
                "idx": idx,
                "seg": seg,
                "volume_metric": 0.0,
                "pitch_metric": 0.0,
            })
            continue

        seg_rms = rms[seg_rms_indices]
        volume_metric = float(np.max(seg_rms))

        pitch_start_idx = int(start_time * sr / 512)
        pitch_end_idx = int(end_time * sr / 512)
        if pitch_end_idx > len(pitch_values):
            pitch_end_idx = len(pitch_values)

        seg_pitches = [p for p in pitch_values[pitch_start_idx:pitch_end_idx] if p > 0]
        pitch_metric = float(np.std(seg_pitches)) if len(seg_pitches) > 0 else 0.0

        metrics.append({
            "idx": idx,
            "seg": seg,
            "volume_metric": volume_metric,
            "pitch_metric": pitch_metric,
        })

    volume_mean = float(np.mean([m["volume_metric"] for m in metrics])) if metrics else 0.0
    pitch_mean = float(np.mean([m["pitch_metric"] for m in metrics])) if metrics else 0.0
    volume_rank_scores = _rank_scores([
        (m["idx"], m["volume_metric"])
        for m in metrics
        if m["volume_metric"] > volume_mean
    ])
    pitch_rank_scores = _rank_scores([
        (m["idx"], m["pitch_metric"])
        for m in metrics
        if m["pitch_metric"] > pitch_mean
    ])

    emphasis_segments = []

    for item in metrics:
        idx = item["idx"]
        seg = item["seg"]
        start_time = seg['start']
        end_time = seg['end']
        volume_info = volume_rank_scores.get(idx, {})
        pitch_info = pitch_rank_scores.get(idx, {})
        volume_score = int(volume_info.get("score", 0))
        pitch_score = int(pitch_info.get("score", 0))
        audio_signal_score = int(volume_score + pitch_score)

        emphasis_segments.append({
            'start': start_time,
            'end': end_time,
            'text': seg['text'],
            'emphasis_score': float(audio_signal_score),
            'std_based_score': float(audio_signal_score),
            'audio_signal_score': audio_signal_score,
            'volume_score': volume_score,
            'pitch_score': pitch_score,
            'volume_metric': item["volume_metric"],
            'pitch_metric': item["pitch_metric"],
            'volume_mean': volume_mean,
            'pitch_mean': pitch_mean,
            'volume_rank': volume_info.get("rank"),
            'pitch_rank': pitch_info.get("rank"),
            'volume_ratio': float(item["volume_metric"] / volume_mean) if volume_mean > 0 else 0.0,
            'pitch_variation': item["pitch_metric"],
            'detected': audio_signal_score > 0,
            'score_method': 'mean_above_rank_20pct',
            'detection_method': 'std_based',
        })

    detected_count = sum(1 for s in emphasis_segments if s.get("detected"))
    ratio = detected_count / len(segments) * 100 if segments else 0.0
    print(f"    -> 오디오 랭킹 기반: {detected_count}개 detected (전체의 {ratio:.1f}%), 점수 {len(emphasis_segments)}개")
    return emphasis_segments
