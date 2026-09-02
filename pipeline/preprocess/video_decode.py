# 영상 프레임 디코딩, OpenCV/FFmpeg(CUDA·VideoToolbox) 백엔드 자동 선택 및 폴백
from __future__ import annotations

import ctypes
import json
import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


log = logging.getLogger(__name__)
_FFMPEG_HWACCEL_DEVICE_CACHE: dict[str, bool] = {}


# GPU 전용 모드 여부, VLVERIFIER_GPU_ONLY 환경변수로 제어
def _gpu_only_enabled() -> bool:
    return os.getenv("VLVERIFIER_GPU_ONLY", "0").strip().lower() not in {"", "0", "false", "no"}


# OpenCV로 영상 메타데이터(fps/프레임 수/해상도/길이) 조회, 실패 시 None
def _opencv_video_metadata(input_path: str) -> dict | None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if fps <= 0.0 or width <= 0 or height <= 0:
        return None
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": (frame_count / fps) if frame_count > 0 else 0.0,
    }


# ffprobe로 영상 메타데이터 조회 (OpenCV보다 정확한 fps/길이 정보), 실패 시 None
def _ffprobe_video_metadata(input_path: str) -> dict | None:
    if shutil.which("ffprobe") is None:
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                input_path,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return None
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") or []
        if not streams:
            return None
        stream = streams[0]
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        if width <= 0 or height <= 0:
            return None

        fps = 0.0
        for key in ("avg_frame_rate", "r_frame_rate"):
            raw = str(stream.get(key) or "").strip()
            if not raw or raw in {"0/0", "N/A"}:
                continue
            if "/" in raw:
                num, den = raw.split("/", 1)
                den_val = float(den)
                if den_val != 0:
                    fps = float(num) / den_val
            else:
                fps = float(raw)
            if fps > 0:
                break
        if fps <= 0.0:
            return None

        frame_count = int(stream.get("nb_frames") or 0)
        duration_sec = 0.0
        for raw_duration in (
            stream.get("duration"),
            (payload.get("format") or {}).get("duration"),
        ):
            try:
                duration_sec = float(raw_duration or 0.0)
            except (TypeError, ValueError):
                duration_sec = 0.0
            if duration_sec > 0.0:
                break
        if frame_count <= 0 and duration_sec > 0.0:
            frame_count = int(round(duration_sec * fps))
        if duration_sec <= 0.0 and frame_count > 0:
            duration_sec = frame_count / fps
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": duration_sec,
        }
    except Exception:
        return None


# 영상 메타데이터 조회, ffprobe 우선 시도 후 실패하면 OpenCV로 폴백
def read_video_metadata(input_path: str) -> dict:
    metadata = _ffprobe_video_metadata(input_path) or _opencv_video_metadata(input_path)
    if metadata is None:
        raise RuntimeError(f"Cannot read video metadata: {input_path}")
    return metadata


# 현재 ffmpeg 빌드가 지원하는 하드웨어 가속 방식 목록 조회
def _ffmpeg_hwaccels() -> set[str]:
    if shutil.which("ffmpeg") is None:
        return set()
    try:
        result = subprocess.run(
            ["ffmpeg", "-hwaccels"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return set()
    accels: set[str] = set()
    for line in (result.stdout or "").splitlines():
        token = line.strip().lower()
        if token and token.isascii() and " " not in token and token != "hardware acceleration methods:":
            accels.add(token)
    return accels


# CUDA 런타임/드라이버 사용 가능 여부 확인 (디바이스 파일, nvidia-smi, libcuda 순으로 확인)
def _cuda_runtime_available() -> bool:
    if platform.system().lower() == "darwin":
        return False
    for marker in ("/dev/nvidiactl", "/dev/nvidia0", "/proc/driver/nvidia/version"):
        if Path(marker).exists():
            return True
    if shutil.which("nvidia-smi") is not None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and (result.stdout or "").strip():
                return True
        except Exception:
            pass
    try:
        ctypes.CDLL("libcuda.so.1")
        return True
    except OSError:
        return False


# 지정 hwaccel로 실제 디코딩이 되는지 더미 입력으로 검증, 결과는 캐시 (cuda 외에는 항상 True로 간주)
def _ffmpeg_hwaccel_device_available(hwaccel: str) -> bool:
    hwaccel = (hwaccel or "").strip().lower()
    if not hwaccel:
        return False
    cached = _FFMPEG_HWACCEL_DEVICE_CACHE.get(hwaccel)
    if cached is not None:
        return cached
    if hwaccel != "cuda":
        _FFMPEG_HWACCEL_DEVICE_CACHE[hwaccel] = True
        return True
    if shutil.which("ffmpeg") is None:
        _FFMPEG_HWACCEL_DEVICE_CACHE[hwaccel] = False
        return False
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-init_hw_device",
                "cuda=vlverifier_cuda",
                "-f",
                "lavfi",
                "-i",
                "nullsrc=s=16x16:d=0.01",
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        available = result.returncode == 0
    except Exception:
        available = False
    _FFMPEG_HWACCEL_DEVICE_CACHE[hwaccel] = available
    return available


# 요청된 백엔드(preferred_backend, 없으면 환경변수/auto)를 실제 사용 가능한 (backend, hwaccel)
# 조합으로 해석, GPU_ONLY 모드에서 요청한 가속을 쓸 수 없으면 예외, auto는 cuda -> (macOS)
# videotoolbox -> opencv 순으로 폴백
def resolve_decode_backend(preferred_backend: str | None) -> tuple[str, str | None]:
    backend = (preferred_backend or os.getenv("VLVERIFIER_SLIDE_DECODE_BACKEND", "auto")).strip().lower()
    hwaccels = _ffmpeg_hwaccels()
    system = platform.system().lower()
    gpu_only = _gpu_only_enabled()
    cuda_usable = (
        "cuda" in hwaccels
        and _cuda_runtime_available()
        and _ffmpeg_hwaccel_device_available("cuda")
    )
    if backend == "ffmpeg-cuda":
        if gpu_only and not cuda_usable:
            raise RuntimeError("VLVERIFIER_GPU_ONLY=1 but ffmpeg-cuda is not available")
        return ("ffmpeg", "cuda") if cuda_usable else ("opencv", None)
    if backend == "ffmpeg-videotoolbox":
        if gpu_only and "videotoolbox" not in hwaccels:
            raise RuntimeError("VLVERIFIER_GPU_ONLY=1 but ffmpeg-videotoolbox is not available")
        return ("ffmpeg", "videotoolbox") if "videotoolbox" in hwaccels else ("opencv", None)
    if backend == "auto":
        if cuda_usable:
            return "ffmpeg", "cuda"
        if system == "darwin" and "videotoolbox" in hwaccels:
            return "ffmpeg", "videotoolbox"
        if gpu_only:
            raise RuntimeError("VLVERIFIER_GPU_ONLY=1 but no GPU decode backend is available")
        return "opencv", None
    if gpu_only and backend == "opencv":
        raise RuntimeError("VLVERIFIER_GPU_ONLY=1 does not allow OpenCV decode backend")
    return "opencv", None


# 목표 크기와 다르면 리사이즈, 같으면 그대로 반환
def _resize_frame_if_needed(frame: np.ndarray, output_width: int | None, output_height: int | None) -> np.ndarray:
    if not output_width or not output_height:
        return frame
    if frame.shape[1] == int(output_width) and frame.shape[0] == int(output_height):
        return frame
    return cv2.resize(frame, (int(output_width), int(output_height)), interpolation=cv2.INTER_AREA)


# start/end 초 단위 구간을 fps 기준 시작 프레임 오프셋과 함께 정규화
def _normalize_range(
    fps: float,
    start_sec: float | None,
    end_sec: float | None,
) -> tuple[float, float | None, int]:
    start = max(0.0, float(start_sec or 0.0))
    end = None if end_sec is None else max(start, float(end_sec))
    start_frame_offset = max(0, int(round(start * fps)))
    return start, end, start_frame_offset


# OpenCV로 프레임을 순회하며 sample_every 또는 sample_fps 기준으로 샘플링해 yield
def _iter_frames_opencv(
    input_path: str,
    fps: float,
    sample_every: int,
    sample_fps: float | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")
    start, end, start_frame_offset = _normalize_range(fps, start_sec, end_sec)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_offset)
    try:
        frame_no = start_frame_offset
        next_sample_ts = start
        if sample_fps is not None and sample_fps > 0:
            step = 1.0 / float(sample_fps)
            # 첫 샘플을 요청 구간 시작에 맞춤
            next_sample_ts = start
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            timestamp = frame_no / fps
            if end is not None and timestamp > end + 1e-9:
                break
            if sample_fps is not None and sample_fps > 0:
                if timestamp + 1e-9 < next_sample_ts:
                    continue
                yield frame_no, timestamp, frame
                next_sample_ts += step
                continue
            if frame_no % sample_every != 0:
                continue
            yield frame_no, timestamp, frame
    finally:
        cap.release()


# ffmpeg 서브프로세스로 rawvideo(bgr24) 프레임을 파이프로 읽어 순회, hwaccel(cuda 등)로
# 하드웨어 디코딩 후 필요 시 스케일/포맷 필터 체인을 적용해 CPU 메모리로 프레임 전달
def _iter_frames_ffmpeg(
    input_path: str,
    fps: float,
    width: int,
    height: int,
    sample_every: int,
    hwaccel: str,
    output_width: int | None = None,
    output_height: int | None = None,
    sample_fps: float | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
):
    target_width = int(output_width or width)
    target_height = int(output_height or height)
    start, end, start_frame_offset = _normalize_range(fps, start_sec, end_sec)
    duration = None if end is None else max(0.0, end - start)

    filters: list[str] = []
    if hwaccel == "cuda" and (target_width != width or target_height != height):
        filters.append(f"scale_cuda={target_width}:{target_height}")
    elif target_width != width or target_height != height:
        filters.append(f"scale={target_width}:{target_height}")

    select_expr = None
    if sample_fps is not None and sample_fps > 0:
        filters.append(f"fps={float(sample_fps):.6f}")
        frame_step = max(1, int(round(fps / float(sample_fps))))
        frame_no = start_frame_offset + 1
    elif sample_every <= 1:
        frame_step = 1
        frame_no = start_frame_offset + 1
    else:
        # ffmpeg select의 n은 청크 상대적이므로, 청크 경계를 넘어서도 동일한 전역 샘플링
        # 패턴을 유지하기 위해 원래 프레임 오프셋을 더함
        select_expr = f"select='not(mod(n+{start_frame_offset + 1}\\,{sample_every}))'"
        frame_no = start_frame_offset + 1
        while frame_no % sample_every != 0:
            frame_no += 1
        frame_step = sample_every

    if hwaccel == "cuda":
        # 많은 FFmpeg 빌드에서 CUDA 프레임을 bgr24로 바로 다운로드할 수 없음,
        # 검증된 체인 유지: scale_cuda -> hwdownload -> format=nv12 -> format=bgr24
        filters.append("hwdownload")
        filters.append("format=nv12")
        filters.append("format=bgr24")
        if select_expr:
            filters.append(select_expr)
    else:
        if select_expr:
            filters.append(select_expr)
        filters.append("format=bgr24")

    select_filter = ",".join(filters) if filters else "format=bgr24"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start > 0:
        cmd.extend(["-ss", f"{start:.6f}"])
    cmd.extend(["-hwaccel", hwaccel])
    if hwaccel == "cuda":
        cmd.extend(["-hwaccel_output_format", "cuda"])
    cmd.extend(["-i", input_path])
    if duration is not None:
        cmd.extend(["-t", f"{duration:.6f}"])
    cmd.extend([
        "-vf",
        select_filter,
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ])

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 8)
    frame_size = target_width * target_height * 3
    out_idx = 0
    try:
        while True:
            raw = proc.stdout.read(frame_size) if proc.stdout else b""
            if not raw:
                break
            if len(raw) != frame_size:
                raise RuntimeError("ffmpeg rawvideo output truncated")
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((target_height, target_width, 3)).copy()
            if sample_fps is not None and sample_fps > 0:
                timestamp = start + (out_idx / float(sample_fps))
                current_frame_no = max(1, int(round(timestamp * fps)))
            else:
                current_frame_no = frame_no
                timestamp = current_frame_no / fps
                frame_no += frame_step
            out_idx += 1
            if end is not None and timestamp > end + 1e-9:
                break
            yield current_frame_no, timestamp, frame
    finally:
        if proc.stdout:
            proc.stdout.close()
    stderr = b""
    if proc.stderr:
        stderr = proc.stderr.read()
        proc.stderr.close()
    ret = proc.wait()
    if ret != 0:
        err_text = stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg decode failed (hwaccel={hwaccel}, exit={ret}): {err_text}")


# ffmpeg hwaccel 디코딩을 우선 시도, 중간에 실패하면 이미 처리한 프레임 이후부터 OpenCV로
# 이어서 디코딩 (GPU_ONLY면 폴백 없이 예외 전파)
def _iter_with_fallback(ffmpeg_iter, opencv_iter_factory, hwaccel: str):
    if _gpu_only_enabled():
        yield from ffmpeg_iter
        return
    last_frame_no = 0
    try:
        for frame_no, timestamp, frame in ffmpeg_iter:
            last_frame_no = max(last_frame_no, int(frame_no))
            yield frame_no, timestamp, frame
        return
    except Exception as exc:
        log.warning("ffmpeg hwaccel decode failed, falling back to OpenCV (hwaccel=%s): %s", hwaccel, exc)
    for frame_no, timestamp, frame in opencv_iter_factory():
        if int(frame_no) <= last_frame_no:
            continue
        yield frame_no, timestamp, frame


# 설정에 따라 ffmpeg(+hwaccel, 실패 시 OpenCV 폴백) 또는 OpenCV로 프레임 이터레이터 생성,
# (iterator, 백엔드 설명 문자열) 반환
def iter_video_frames(
    input_path: str,
    *,
    fps: float,
    width: int,
    height: int,
    sample_every: int = 1,
    decode_backend: str | None = None,
    output_width: int | None = None,
    output_height: int | None = None,
    sample_fps: float | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> tuple[object, str]:
    sample_every = max(1, int(sample_every))
    resolved_backend, hwaccel = resolve_decode_backend(decode_backend)
    if resolved_backend == "ffmpeg" and hwaccel:
        return (
            _iter_with_fallback(
                _iter_frames_ffmpeg(
                    input_path,
                    fps,
                    width,
                    height,
                    sample_every,
                    hwaccel,
                    output_width=output_width,
                    output_height=output_height,
                    sample_fps=sample_fps,
                    start_sec=start_sec,
                    end_sec=end_sec,
                ),
                lambda: (
                    (frame_no, timestamp, _resize_frame_if_needed(frame, output_width, output_height))
                    for frame_no, timestamp, frame in _iter_frames_opencv(
                        input_path,
                        fps,
                        sample_every,
                        sample_fps=sample_fps,
                        start_sec=start_sec,
                        end_sec=end_sec,
                    )
                ),
                hwaccel,
            ),
            f"ffmpeg-{hwaccel}/opencv-fallback",
        )
    return (
        (
            (frame_no, timestamp, _resize_frame_if_needed(frame, output_width, output_height))
            for frame_no, timestamp, frame in _iter_frames_opencv(
                input_path,
                fps,
                sample_every,
                sample_fps=sample_fps,
                start_sec=start_sec,
                end_sec=end_sec,
            )
        ),
        "opencv",
    )


# OpenCV로 지정 타임스탬프의 프레임 1장 읽기, 실패 시 None
def _read_frame_by_timestamp_opencv(input_path: str, timestamp_sec: float):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return frame
    finally:
        cap.release()


# ffmpeg로 지정 타임스탬프의 프레임 1장 읽기, 실패 시 None
def _read_frame_by_timestamp_ffmpeg(
    input_path: str,
    timestamp_sec: float,
    width: int,
    height: int,
    hwaccel: str | None,
):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, timestamp_sec):.6f}"]
    if hwaccel:
        cmd.extend(["-hwaccel", hwaccel])
        if hwaccel == "cuda":
            cmd.extend(["-hwaccel_output_format", "cuda"])
    cmd.extend(
        [
            "-i",
            input_path,
            "-vf",
            "hwdownload,format=nv12,format=bgr24" if hwaccel == "cuda" else "format=bgr24",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
    )
    frame_size = width * height * 3
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, timeout=30)
    except Exception:
        return None
    if proc.returncode != 0 or len(proc.stdout) < frame_size:
        return None
    raw = proc.stdout[:frame_size]
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()


# 지정 타임스탬프의 프레임 1장 조회, ffmpeg(hwaccel) 우선 시도 후 실패하면 ffmpeg CPU,
# 그마저 실패하면 OpenCV로 순차 폴백 (GPU_ONLY면 폴백 단계에서 예외)
def read_frame_at_timestamp(
    input_path: str,
    timestamp_sec: float,
    *,
    width: int,
    height: int,
    decode_backend: str | None = None,
):
    resolved_backend, hwaccel = resolve_decode_backend(decode_backend)
    if resolved_backend == "ffmpeg":
        frame = _read_frame_by_timestamp_ffmpeg(input_path, timestamp_sec, width, height, hwaccel)
        if frame is not None:
            return frame
        if _gpu_only_enabled():
            raise RuntimeError(
                f"VLVERIFIER_GPU_ONLY=1 and ffmpeg timestamp decode failed at {timestamp_sec:.6f}s"
            )
    elif _gpu_only_enabled():
        raise RuntimeError("VLVERIFIER_GPU_ONLY=1 requires ffmpeg GPU timestamp decode")
    frame = _read_frame_by_timestamp_ffmpeg(input_path, timestamp_sec, width, height, None)
    if frame is not None:
        return frame
    if _gpu_only_enabled():
        raise RuntimeError(
            f"VLVERIFIER_GPU_ONLY=1 and non-GPU ffmpeg timestamp fallback would be required at {timestamp_sec:.6f}s"
        )
    return _read_frame_by_timestamp_opencv(input_path, timestamp_sec)