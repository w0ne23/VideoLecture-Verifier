"""
Groq Whisper 음성 전사 (무음 구간 사전 감지 기반 청크 분할)

핵심:
- ffmpeg silencedetect로 무음 구간을 사전 감지
- 1.5초 이상 무음에서 청크를 자르고 무음 자체는 API에 보내지 않음
  → Whisper의 무음 구간 환각(repetition loop, "감사합니다", "다음 영상에서 만나요" 등) 차단
- 무음을 제거한 발화 조각들을 다시 3분 단위로 이어붙여 Whisper에 전송
- 0.6초 이상 무음은 silences로 별도 반환 (slide_classifier 호환)

환경변수:
- TRANSCRIBE_SILENCE_DB           (default: -35)  silencedetect noise 임계값(dB)
- TRANSCRIBE_SILENCE_MIN_SEC      (default: 0.6)  silences 출력 최소 길이
- TRANSCRIBE_SILENCE_SPLIT_SEC    (default: 1.5)  청크 분할/드롭 임계값
- TRANSCRIBE_CHUNK_MAX_SEC        (default: 180)  무음 제거 후 Whisper에 보낼 청크 최대 길이
- TRANSCRIBE_CHUNK_MIN_SEC        (default: 30)   청크 최소 길이(짧은 발화는 다음 청크와 묶음)
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from .config import groq_client


# ── 환경변수 ──────────────────────────────────────────────────────
SILENCE_NOISE_DB    = float(os.getenv("TRANSCRIBE_SILENCE_DB",        "-35"))
SILENCE_MIN_SEC     = float(os.getenv("TRANSCRIBE_SILENCE_MIN_SEC",   "0.6"))
SILENCE_SPLIT_SEC   = float(os.getenv("TRANSCRIBE_SILENCE_SPLIT_SEC", "1.5"))
CHUNK_MAX_SEC       = float(os.getenv("TRANSCRIBE_CHUNK_MAX_SEC",     "180"))
CHUNK_MIN_SEC       = float(os.getenv("TRANSCRIBE_CHUNK_MIN_SEC",     "30"))
CHUNK_PAD_SEC       = 0.1  # 청크 양 끝에 100ms padding (어두/어말 보존)


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _normalize_words(words, chunk_start: float = 0.0):
    out = []
    if not isinstance(words, list):
        return out
    for w in words:
        if not isinstance(w, dict):
            continue
        text = str(w.get("word") or w.get("text") or "").strip()
        if not text:
            continue
        start = _safe_float(w.get("start"), 0.0) + chunk_start
        end = _safe_float(w.get("end"), start) + chunk_start
        out.append({"text": text, "start": start, "end": end})
    return out


def _normalize_words_mapped(words, time_map: list[dict], time_offset: float = 0.0):
    out = []
    if not isinstance(words, list):
        return out
    for w in words:
        if not isinstance(w, dict):
            continue
        text = str(w.get("word") or w.get("text") or "").strip()
        if not text:
            continue
        start_rel = _safe_float(w.get("start"), 0.0)
        end_rel = _safe_float(w.get("end"), start_rel)
        start = time_offset + _map_stitched_time(start_rel, time_map)
        end = time_offset + _map_stitched_time(end_rel, time_map)
        out.append({"text": text, "start": start, "end": max(start, end)})
    return out


# ── 무음 감지 ─────────────────────────────────────────────────────

def _extract_full_audio(video_path: str, audio_path: str, start_sec: float = 0.0, duration: float = 0.0) -> bool:
    """
    영상에서 mono 16kHz wav 추출.
    duration > 0 이면 [start_sec, start_sec + duration] 범위만 추출.
    """
    cmd = ["ffmpeg"]
    if start_sec > 0:
        cmd += ["-ss", str(start_sec)]
    if duration > 0:
        cmd += ["-t", str(duration)]
    cmd += [
        "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-y", audio_path,
    ]
    subprocess.run(cmd, capture_output=True)
    return Path(audio_path).exists() and Path(audio_path).stat().st_size > 0


_SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9]+\.?[0-9]*)")
_SILENCE_END_RE   = re.compile(r"silence_end:\s*([0-9]+\.?[0-9]*)\s*\|\s*silence_duration:\s*([0-9]+\.?[0-9]*)")


def detect_silences(
    audio_path: str,
    noise_db: float = SILENCE_NOISE_DB,
    min_dur: float = SILENCE_MIN_SEC,
) -> list[dict]:
    """
    ffmpeg silencedetect로 무음 구간을 감지한다.

    Returns: [{"start": float, "end": float, "duration": float}, ...]
             audio_path 기준 상대 시간.
    """
    if not Path(audio_path).exists():
        return []

    # silencedetect는 stderr로 결과를 출력
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    log = (result.stderr or "") + (result.stdout or "")

    # silence_start: 12.345
    # silence_end: 14.567 | silence_duration: 2.222
    starts: list[float] = []
    ends: list[tuple[float, float]] = []
    for line in log.splitlines():
        m = _SILENCE_START_RE.search(line)
        if m:
            starts.append(float(m.group(1)))
            continue
        m = _SILENCE_END_RE.search(line)
        if m:
            ends.append((float(m.group(1)), float(m.group(2))))

    silences: list[dict] = []
    # silence_start와 silence_end는 보통 짝을 이룸. 짝이 안 맞으면 무시.
    n = min(len(starts), len(ends))
    for i in range(n):
        s = starts[i]
        e, d = ends[i]
        if e <= s or d <= 0:
            continue
        silences.append({"start": s, "end": e, "duration": d})
    return silences


def build_speech_chunks(
    total_duration: float,
    silences: list[dict],
    split_min: float = SILENCE_SPLIT_SEC,
    chunk_max: float = CHUNK_MAX_SEC,
    chunk_min: float = CHUNK_MIN_SEC,
    pad: float = CHUNK_PAD_SEC,
) -> list[list[tuple[float, float]]]:
    """
    무음 구간 정보를 바탕으로 Whisper에 보낼 발화(speech) 청크 그룹을 산출한다.

    동작:
    - split_min 이상 무음만 청크 경계로 사용 (그 무음은 API에 보내지 않음 = 드롭)
    - split_min 미만 무음은 무시 (청크 안에 그대로 포함)
    - 무음을 제거하고 남은 발화 조각들을 다시 chunk_max초 단위로 묶음
    - 청크가 chunk_max를 넘으면 강제로 분할
    - 마지막 청크가 chunk_min 미만이면 직전 청크와 합칠 수 있을 때 합침
    - 양 끝에 pad 만큼 padding (어두/어말 보존)

    Returns: [[(start_sec, end_sec), ...], ...]  audio_path 기준 절대 시간.
    """
    if total_duration <= 0:
        return []

    # 1) split_min 이상 무음만 추려서 정렬
    cuts = sorted(
        [s for s in silences if s["duration"] >= split_min],
        key=lambda s: s["start"],
    )

    # 2) 무음 사이 발화 구간(segment) 산출
    raw_segments: list[tuple[float, float]] = []
    cursor = 0.0
    for cut in cuts:
        s = max(0.0, cut["start"])
        e = min(total_duration, cut["end"])
        if s > cursor:
            raw_segments.append((cursor, s))
        cursor = e
    if cursor < total_duration:
        raw_segments.append((cursor, total_duration))

    # 무음만 있는 영상이면 빈 리스트
    if not raw_segments:
        return []

    # 3) padding 적용 (양 끝 100ms 확장, 단 다른 segment와 안 겹치게)
    padded: list[tuple[float, float]] = []
    for i, (s, e) in enumerate(raw_segments):
        ps = max(0.0, s - pad)
        pe = min(total_duration, e + pad)
        if padded:
            ps = max(ps, padded[-1][1])
        padded.append((ps, pe))

    # 4) chunk_max 초과 segment는 먼저 강제 분할
    pieces: list[tuple[float, float]] = []
    for s, e in padded:
        if (e - s) <= chunk_max:
            pieces.append((s, e))
            continue
        cur = s
        while cur < e:
            nxt = min(e, cur + chunk_max)
            pieces.append((cur, nxt))
            cur = nxt

    # 5) 무음을 제거한 발화 조각들을 chunk_max초 단위로 다시 묶는다.
    groups: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    current_dur = 0.0
    for s, e in pieces:
        dur = max(0.0, e - s)
        if dur <= 0:
            continue
        if current and current_dur + dur > chunk_max:
            groups.append(current)
            current = []
            current_dur = 0.0
        current.append((s, e))
        current_dur += dur
    if current:
        groups.append(current)

    # 마지막 그룹이 너무 짧고 직전 그룹에 붙여도 max를 넘지 않으면 합친다.
    if len(groups) >= 2:
        last_dur = _range_group_duration(groups[-1])
        prev_dur = _range_group_duration(groups[-2])
        if last_dur < chunk_min and prev_dur + last_dur <= chunk_max:
            groups[-2].extend(groups[-1])
            groups.pop()

    return groups


def _range_group_duration(ranges: list[tuple[float, float]]) -> float:
    return sum(max(0.0, e - s) for s, e in ranges)


def _time_map_for_ranges(ranges: list[tuple[float, float]]) -> list[dict]:
    time_map = []
    cursor = 0.0
    for source_start, source_end in ranges:
        dur = max(0.0, source_end - source_start)
        if dur <= 0:
            continue
        time_map.append({
            "stitched_start": cursor,
            "stitched_end": cursor + dur,
            "source_start": source_start,
            "source_end": source_end,
        })
        cursor += dur
    return time_map


def _map_stitched_time(t: float, time_map: list[dict]) -> float:
    if not time_map:
        return t
    value = max(0.0, float(t or 0.0))
    for row in time_map:
        if value <= row["stitched_end"]:
            offset = max(0.0, value - row["stitched_start"])
            return min(row["source_end"], row["source_start"] + offset)
    last = time_map[-1]
    return last["source_end"]


def _extract_audio_slice(audio_path: str, out_path: Path, start: float, duration: float) -> bool:
    if duration <= 0:
        return False
    cmd = [
        "ffmpeg", "-ss", f"{start}", "-t", f"{duration}",
        "-i", audio_path, "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-y", str(out_path),
    ]
    subprocess.run(cmd, capture_output=True)
    return out_path.exists() and out_path.stat().st_size > 0


def _concat_audio_slices(audio_path: str, ranges: list[tuple[float, float]], chunk_path: Path, base_dir: Path, label: str) -> bool:
    if not ranges:
        return False
    if len(ranges) == 1:
        s, e = ranges[0]
        return _extract_audio_slice(audio_path, chunk_path, s, e - s)

    parts: list[Path] = []
    concat_list = base_dir / f"temp_concat_{label}.txt"
    try:
        for j, (s, e) in enumerate(ranges):
            part = base_dir / f"temp_part_{label}_{j}.wav"
            if _extract_audio_slice(audio_path, part, s, e - s):
                parts.append(part)
        if not parts:
            return False
        concat_list.write_text(
            "\n".join(f"file '{part.as_posix()}'" for part in parts),
            encoding="utf-8",
        )
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-y", str(chunk_path),
        ]
        subprocess.run(cmd, capture_output=True)
        return chunk_path.exists() and chunk_path.stat().st_size > 0
    finally:
        concat_list.unlink(missing_ok=True)
        for part in parts:
            part.unlink(missing_ok=True)


# ── Groq 호출 ────────────────────────────────────────────────────

def _call_groq(chunk_path: str) -> Optional[object]:
    """Groq Whisper API 호출. 실패 시 None."""
    try:
        with open(chunk_path, "rb") as f:
            return groq_client.audio.transcriptions.create(
                file=(chunk_path, f.read()),
                model="whisper-large-v3-turbo",
                language="ko",
                response_format="verbose_json",
            )
    except Exception as e:
        print(f"    [Groq 오류] {e}")
        return None


def _transcribe_chunks_from_audio(
    audio_path: str,
    chunks: list[list[tuple[float, float]]],
    time_offset: float,
    base_dir: Path,
    label_prefix: str = "",
) -> list[dict]:
    """
    이미 추출된 audio_path에서 무음 제거 후 묶은 chunks를 Groq에 전송한다.
    time_offset은 audio_path 기준 시간을 영상 절대 시간으로 변환하는 값.
    """
    segments: list[dict] = []
    for i, ranges in enumerate(chunks):
        dur = _range_group_duration(ranges)
        if dur <= 0:
            continue
        chunk_path = base_dir / f"temp_chunk_{label_prefix}{i}.wav"
        time_map = _time_map_for_ranges(ranges)
        label = f"{label_prefix}{i}"
        if not _concat_audio_slices(audio_path, ranges, chunk_path, base_dir, label):
            continue
        p = Path(chunk_path)
        if not p.exists() or p.stat().st_size == 0:
            continue

        transcription = _call_groq(str(chunk_path))
        try:
            from .cost_report import record_audio_call

            metadata = {
                "chunk_index": i,
                "chunk_start": ranges[0][0],
                "chunk_end": ranges[-1][1],
                "stitched_audio_seconds": dur,
                "source_span_seconds": ranges[-1][1] - ranges[0][0],
                "speech_range_count": len(ranges),
                "speech_ranges": [{"start": s, "end": e} for s, e in ranges],
            }
            record_audio_call(
                stage="stage2b_transcriber",
                provider="groq",
                model="whisper-large-v3-turbo",
                audio_seconds=dur,
                metadata=metadata,
            )
        except Exception:
            pass
        if transcription is not None:
            for seg in transcription.segments:
                # seg["start"]/seg["end"]는 stitched chunk 내 상대 시간
                # → 무음 제거 전 audio_path 기준 시간으로 되돌린 뒤 영상 절대 시간으로 변환
                abs_start = time_offset + _map_stitched_time(float(seg["start"]), time_map)
                abs_end = time_offset + _map_stitched_time(float(seg["end"]), time_map)
                words = _normalize_words_mapped(seg.get("words"), time_map, time_offset=time_offset)
                segments.append({
                    "start": abs_start,
                    "end":   max(abs_start, abs_end),
                    "text":  (seg["text"] or "").strip(),
                    "words": words,
                })

        p.unlink(missing_ok=True)

    return segments


# ── Public API ──────────────────────────────────────────────────

def transcribe_range(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_dir: Path,
) -> dict:
    """
    영상의 [start_sec, end_sec] 구간을 전사한다.

    1. 해당 구간 오디오 추출
    2. 무음 감지 (0.6초 이상)
    3. 1.5초 이상 무음 기준으로 청크 분할 + 무음 드롭
    4. 각 청크를 Groq에 전송
    5. 영상 절대 시간으로 보정 후 반환

    Returns:
        {
            "segments": [{"start", "end", "text", "words"}, ...],
            "silences": [{"start", "end", "duration"}, ...]   # 영상 절대 시간
        }
    """
    if end_sec <= start_sec:
        return {"segments": [], "silences": []}

    duration = end_sec - start_sec
    base_dir = Path(output_dir) if output_dir else Path(".")
    base_dir.mkdir(parents=True, exist_ok=True)
    audio_path = str(base_dir / f"temp_audio_{int(start_sec)}_{int(end_sec)}.wav")

    if not _extract_full_audio(video_path, audio_path, start_sec=start_sec, duration=duration):
        return {"segments": [], "silences": []}

    try:
        # 무음 감지 (audio 내 상대 시간)
        local_silences = detect_silences(audio_path, noise_db=SILENCE_NOISE_DB, min_dur=SILENCE_MIN_SEC)

        # 청크 산출 (audio 내 상대 시간)
        chunks = build_speech_chunks(
            total_duration=duration,
            silences=local_silences,
            split_min=SILENCE_SPLIT_SEC,
            chunk_max=CHUNK_MAX_SEC,
            chunk_min=CHUNK_MIN_SEC,
            pad=CHUNK_PAD_SEC,
        )

        # 청크 전사 (영상 절대 시간으로 변환)
        segments = _transcribe_chunks_from_audio(
            audio_path=audio_path,
            chunks=chunks,
            time_offset=start_sec,
            base_dir=base_dir,
            label_prefix=f"{int(start_sec)}_",
        )

        # silences를 영상 절대 시간으로 변환
        abs_silences = [
            {
                "start":    s["start"] + start_sec,
                "end":      s["end"]   + start_sec,
                "duration": s["duration"],
            }
            for s in local_silences
        ]

        return {"segments": segments, "silences": abs_silences}
    finally:
        Path(audio_path).unlink(missing_ok=True)


def transcribe_video(video_path: str, duration: float, output_dir=None) -> dict:
    """
    영상 전체를 전사한다 (metadata 없이 fallback 경로).

    Returns:
        {
            "segments": [{"start", "end", "text", "words"}, ...],
            "silences": [{"start", "end", "duration"}, ...]
        }
    """
    if duration <= 0:
        return {"segments": [], "silences": []}
    return transcribe_range(
        video_path=video_path,
        start_sec=0.0,
        end_sec=duration,
        output_dir=Path(output_dir) if output_dir else Path("."),
    )
