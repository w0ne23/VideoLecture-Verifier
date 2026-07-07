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


def _gpu_only_enabled() -> bool:
    return os.getenv("GRAPHLEC_GPU_ONLY", "0").strip().lower() not in {"", "0", "false", "no"}


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


def read_video_metadata(input_path: str) -> dict:
    metadata = _ffprobe_video_metadata(input_path) or _opencv_video_metadata(input_path)
    if metadata is None:
        raise RuntimeError(f"Cannot read video metadata: {input_path}")
    return metadata


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
                "cuda=verilec_cuda",
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


def resolve_decode_backend(preferred_backend: str | None) -> tuple[str, str | None]:
    backend = (preferred_backend or os.getenv("GRAPHLEC_SLIDE_DECODE_BACKEND", "auto")).strip().lower()
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
            raise RuntimeError("GRAPHLEC_GPU_ONLY=1 but ffmpeg-cuda is not available")
        return ("ffmpeg", "cuda") if cuda_usable else ("opencv", None)
    if backend == "ffmpeg-videotoolbox":
        if gpu_only and "videotoolbox" not in hwaccels:
            raise RuntimeError("GRAPHLEC_GPU_ONLY=1 but ffmpeg-videotoolbox is not available")
        return ("ffmpeg", "videotoolbox") if "videotoolbox" in hwaccels else ("opencv", None)
    if backend == "auto":
        if cuda_usable:
            return "ffmpeg", "cuda"
        if system == "darwin" and "videotoolbox" in hwaccels:
            return "ffmpeg", "videotoolbox"
        if gpu_only:
            raise RuntimeError("GRAPHLEC_GPU_ONLY=1 but no GPU decode backend is available")
        return "opencv", None
    if gpu_only and backend == "opencv":
        raise RuntimeError("GRAPHLEC_GPU_ONLY=1 does not allow OpenCV decode backend")
    return "opencv", None


def _iter_frames_opencv(
    input_path: str,
    fps: float,
    sample_every: int,
    sample_fps: float | None = None,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")
    try:
        frame_no = 0
        next_sample_ts = 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            timestamp = frame_no / fps
            if sample_fps is not None and sample_fps > 0:
                if timestamp + 1e-9 < next_sample_ts:
                    continue
                yield frame_no, timestamp, frame
                next_sample_ts += 1.0 / sample_fps
                continue
            if frame_no % sample_every != 0:
                continue
            yield frame_no, timestamp, frame
    finally:
        cap.release()


def _resize_frame_if_needed(frame: np.ndarray, output_width: int | None, output_height: int | None) -> np.ndarray:
    if not output_width or not output_height:
        return frame
    if frame.shape[1] == int(output_width) and frame.shape[0] == int(output_height):
        return frame
    return cv2.resize(frame, (int(output_width), int(output_height)), interpolation=cv2.INTER_AREA)


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
):
    target_width = int(output_width or width)
    target_height = int(output_height or height)
    filters: list[str] = []
    if hwaccel == "cuda" and (target_width != width or target_height != height):
        filters.append(f"scale_cuda={target_width}:{target_height}")
    elif target_width != width or target_height != height:
        filters.append(f"scale={target_width}:{target_height}")

    if sample_fps is not None and sample_fps > 0:
        filters.append(f"fps={sample_fps:.6f}")
        frame_no = 1
        frame_step = max(1, int(round(fps / sample_fps)))
        select_expr = None
    elif sample_every <= 1:
        frame_no = 1
        frame_step = 1
        select_expr = None
    else:
        select_expr = f"select='not(mod(n+1\\,{sample_every}))'"
        frame_no = sample_every
        frame_step = sample_every

    if hwaccel == "cuda":
        # CUDA frames cannot be downloaded directly as bgr24 on many ffmpeg builds.
        # Keep the successful chain:
        # scale_cuda -> hwdownload -> format=nv12 -> format=bgr24 -> select
        filters.append("hwdownload")
        filters.append("format=nv12")
        filters.append("format=bgr24")
        if select_expr:
            filters.append(select_expr)
    else:
        if select_expr:
            filters.append(select_expr)
        filters.append("format=bgr24")
    select_filter = ",".join(filters)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        hwaccel,
    ]
    if hwaccel == "cuda":
        cmd.extend(["-hwaccel_output_format", "cuda"])
    cmd.extend([
        "-i",
        input_path,
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
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10 ** 8,
    )
    frame_size = target_width * target_height * 3
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if not raw:
                break
            if len(raw) != frame_size:
                raise RuntimeError("ffmpeg rawvideo output truncated")
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((target_height, target_width, 3)).copy()
            yield frame_no, frame_no / fps, frame
            frame_no += frame_step
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
                ),
                lambda: (
                    (frame_no, timestamp, _resize_frame_if_needed(frame, output_width, output_height))
                    for frame_no, timestamp, frame in _iter_frames_opencv(
                        input_path,
                        fps,
                        sample_every,
                        sample_fps=sample_fps,
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
            )
        ),
        "opencv",
    )


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
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except Exception:
        return None
    if proc.returncode != 0 or len(proc.stdout) < frame_size:
        return None
    raw = proc.stdout[:frame_size]
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()


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
                f"GRAPHLEC_GPU_ONLY=1 and ffmpeg timestamp decode failed at {timestamp_sec:.6f}s"
            )
    elif _gpu_only_enabled():
        raise RuntimeError("GRAPHLEC_GPU_ONLY=1 requires ffmpeg GPU timestamp decode")
    frame = _read_frame_by_timestamp_ffmpeg(input_path, timestamp_sec, width, height, None)
    if frame is not None:
        return frame
    if _gpu_only_enabled():
        raise RuntimeError(
            f"GRAPHLEC_GPU_ONLY=1 and non-GPU ffmpeg timestamp fallback would be required at {timestamp_sec:.6f}s"
        )
    return _read_frame_by_timestamp_opencv(input_path, timestamp_sec)
