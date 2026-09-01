"""Media primitives: probe, audio extraction, frame sampling, cut detection.

No system ffmpeg is required — `imageio-ffmpeg` ships a static binary, and PyAV
(pulled in by faster-whisper) handles container probing and frame seeking. A
system ffmpeg is used automatically if one is on PATH.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import wave
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..logging_setup import get_logger

log = get_logger(__name__)


class MediaError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# binary resolution
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def ffmpeg_bin() -> str:
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - only if the wheel is broken
        raise MediaError(
            "No ffmpeg available. `pip install imageio-ffmpeg` or install ffmpeg."
        ) from exc


def _run(args: Sequence[str], *, timeout: int = 900) -> subprocess.CompletedProcess:
    log.debug("ffmpeg: %s", " ".join(str(a) for a in args[:12]))
    return subprocess.run(
        [str(a) for a in args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def probe(video_path: Path) -> Dict[str, Optional[float]]:
    """Container facts: duration, dimensions, fps, audio presence."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise MediaError(f"missing video file: {video_path}")

    try:
        import av

        with av.open(str(video_path)) as container:
            vstreams = container.streams.video
            astreams = container.streams.audio
            duration = None
            if container.duration:
                duration = float(container.duration) / av.time_base
            info: Dict[str, Optional[float]] = {
                "duration_sec": duration,
                "width": None,
                "height": None,
                "fps": None,
                "has_audio_stream": 1 if astreams else 0,
            }
            if vstreams:
                v = vstreams[0]
                info["width"] = int(v.codec_context.width or 0) or None
                info["height"] = int(v.codec_context.height or 0) or None
                try:
                    info["fps"] = float(v.average_rate) if v.average_rate else None
                except (TypeError, ZeroDivisionError):
                    info["fps"] = None
                if not info["duration_sec"] and v.duration and v.time_base:
                    info["duration_sec"] = float(v.duration * v.time_base)
            return info
    except Exception as exc:
        log.debug("PyAV probe failed for %s (%s); falling back to ffmpeg", video_path.name, exc)

    # Fallback: parse ffmpeg's own stderr banner.
    proc = _run([ffmpeg_bin(), "-hide_banner", "-i", str(video_path)])
    text = proc.stderr.decode("utf-8", errors="replace")
    info = {"duration_sec": None, "width": None, "height": None, "fps": None, "has_audio_stream": 0}
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", text)
    if m:
        h, mnt, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        info["duration_sec"] = h * 3600 + mnt * 60 + s
    m = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", text)
    if m:
        info["width"], info["height"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", text)
    if m:
        info["fps"] = float(m.group(1))
    info["has_audio_stream"] = 1 if re.search(r"Stream #\d+:\d+.*?: Audio:", text) else 0
    if info["duration_sec"] is None:
        raise MediaError(f"could not determine duration for {video_path}")
    return info


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------
def extract_audio(video_path: Path, out_path: Path, *, sample_rate: int = 16000) -> Path:
    """16 kHz mono PCM WAV — what Whisper wants, and easy to analyse with numpy."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 1024:
        return out_path
    proc = _run(
        [
            ffmpeg_bin(), "-nostdin", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-c:a", "pcm_s16le", str(out_path),
        ]
    )
    if proc.returncode != 0 or not out_path.exists():
        raise MediaError(
            f"audio extraction failed for {video_path.name}: "
            f"{proc.stderr.decode('utf-8', errors='replace')[-400:]}"
        )
    return out_path


def audio_envelope(wav_path: Path, frame_ms: int = 50) -> Tuple[np.ndarray, np.ndarray, int]:
    """RMS envelope of a mono PCM WAV.

    Returns (times_sec, rms, sample_rate). Used for dead-air measurement and
    the background-music heuristic (energy present where no words are spoken).
    """
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        width = wf.getsampwidth()
    if width != 2:
        raise MediaError(f"expected 16-bit PCM, got {width * 8}-bit")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size == 0:
        return np.zeros(0), np.zeros(0), sample_rate

    hop = max(1, int(sample_rate * frame_ms / 1000))
    n_hops = int(np.ceil(samples.size / hop))
    padded = np.pad(samples, (0, n_hops * hop - samples.size))
    blocks = padded.reshape(n_hops, hop)
    rms = np.sqrt((blocks ** 2).mean(axis=1))
    times = np.arange(n_hops) * (hop / sample_rate)
    return times, rms, sample_rate


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------
def extract_frames(
    video_path: Path,
    timestamps: Sequence[float],
    out_dir: Path,
    *,
    prefix: str = "f",
    max_width: int = 720,
    overwrite: bool = False,
) -> List[Dict[str, object]]:
    """Grab one JPEG per requested timestamp.

    Frames are written to disk and kept, so vision analysis can be re-run later
    against the exact same source material without re-downloading anything.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []

    for ts in timestamps:
        name = f"{prefix}_{ts:07.2f}.jpg".replace(".", "_", 1)
        out_path = out_dir / name
        if out_path.exists() and not overwrite:
            results.append({"t_sec": round(float(ts), 2), "path": str(out_path)})
            continue
        proc = _run(
            [
                ffmpeg_bin(), "-nostdin", "-y", "-loglevel", "error",
                "-ss", f"{max(0.0, float(ts)):.3f}",
                "-i", str(video_path),
                "-frames:v", "1",
                "-vf", f"scale='min({max_width},iw)':-2",
                "-q:v", "3",
                str(out_path),
            ],
            timeout=120,
        )
        if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            results.append({"t_sec": round(float(ts), 2), "path": str(out_path)})
        else:
            log.debug("frame extraction failed at %.2fs for %s", ts, video_path.name)

    # Attach real dimensions (cheap; Pillow only reads the header).
    try:
        from PIL import Image

        for r in results:
            try:
                with Image.open(r["path"]) as im:
                    r["width"], r["height"] = im.size
            except Exception:
                pass
    except ImportError:
        pass
    return results


def plan_frame_timestamps(
    duration: float,
    *,
    hook_window: float = 5.0,
    hook_interval: float = 0.5,
    body_interval: float = 3.0,
    max_frames: int = 40,
) -> List[Tuple[float, str]]:
    """Dense sampling across the hook, sparse across the body.

    Returns [(t_sec, phase)] where phase is 'hook' or 'body'.
    """
    duration = max(0.0, float(duration or 0.0))
    stamps: List[Tuple[float, str]] = []
    t = 0.0
    while t < min(hook_window, duration) + 1e-6:
        stamps.append((round(t, 2), "hook"))
        t += hook_interval
    t = max(hook_window, hook_interval)
    while t < duration - 0.2:
        stamps.append((round(t, 2), "body"))
        t += body_interval
    if duration > 1.0:
        stamps.append((round(max(0.0, duration - 0.5), 2), "body"))

    seen, out = set(), []
    for ts, phase in stamps:
        if ts in seen or ts > duration:
            continue
        seen.add(ts)
        out.append((ts, phase))
    if len(out) > max_frames:  # keep all hook frames, thin the body
        hook = [x for x in out if x[1] == "hook"]
        body = [x for x in out if x[1] == "body"]
        keep = max(1, max_frames - len(hook))
        step = max(1, len(body) // keep)
        out = hook + body[::step][:keep]
        out.sort()
    return out


# ---------------------------------------------------------------------------
# cuts / scene changes
# ---------------------------------------------------------------------------
_SHOWINFO_PTS = re.compile(r"pts_time:([0-9.]+)")


def detect_scene_changes(video_path: Path, threshold: float = 0.30) -> List[float]:
    """Timestamps where the frame content changes abruptly — i.e. cuts.

    ffmpeg's `scene` score is a normalised inter-frame difference; a hard cut
    scores high, a pan or a zoom scores low. Threshold is tunable in .env.
    """
    proc = _run(
        [
            ffmpeg_bin(), "-nostdin", "-loglevel", "info",
            "-i", str(video_path),
            "-filter:v", f"select='gt(scene,{threshold})',showinfo",
            "-f", "null", "-",
        ],
        timeout=600,
    )
    text = proc.stderr.decode("utf-8", errors="replace")
    stamps = [float(m) for m in _SHOWINFO_PTS.findall(text)]
    # showinfo also prints the first selected frame at ~0.0; a cut at t=0 is
    # just "the video started", not an edit.
    return sorted(t for t in stamps if t > 0.15)


def detect_silence(
    video_path: Path, *, noise_db: float = -32.0, min_dur: float = 0.30
) -> List[Tuple[float, float]]:
    """Silence spans as (start, end) via ffmpeg's silencedetect."""
    proc = _run(
        [
            ffmpeg_bin(), "-nostdin", "-loglevel", "info",
            "-i", str(video_path),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
            "-f", "null", "-",
        ],
        timeout=600,
    )
    text = proc.stderr.decode("utf-8", errors="replace")
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[0-9.]+)", text)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", text)]
    spans: List[Tuple[float, float]] = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else None
        spans.append((max(0.0, s), e if e is not None else s))
    return spans
