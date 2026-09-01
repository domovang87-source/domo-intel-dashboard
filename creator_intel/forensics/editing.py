"""RAW editing measurements — deterministic, no model in the loop.

Everything here is derived from the video file itself plus the word-level
transcript: cuts, dead air, pacing, silence, and a background-music heuristic
based on audio energy present where nobody is speaking.
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import settings
from ..logging_setup import get_logger
from ..media.ffmpeg import audio_envelope, detect_scene_changes, detect_silence, extract_audio
from ..utils import from_json, safe_div

log = get_logger(__name__)

JUMP_CUT_MAX_GAP = 1.2   # shots shorter than this read as rapid-fire jump cutting
MUSIC_RMS_FLOOR = 0.012  # ~-38 dBFS; below this a "gap" is genuine silence


def pace_bucket(wpm: Optional[float]) -> Optional[str]:
    if not wpm:
        return None
    if wpm < 110:
        return "slow"
    if wpm < 165:
        return "measured"
    if wpm < 215:
        return "fast"
    return "rapid"


def shot_stats(cuts: Sequence[float], duration: float) -> Dict[str, Any]:
    """Shot lengths implied by cut timestamps."""
    boundaries = [0.0] + sorted(float(c) for c in cuts) + [float(duration or 0.0)]
    shots = [b - a for a, b in zip(boundaries, boundaries[1:]) if b > a]
    if not shots:
        return {"avg_shot_sec": duration, "median_shot_sec": duration, "min_shot_sec": duration,
                "jump_cut_count": 0, "has_jump_cuts": 0}
    jump_cuts = sum(1 for s in shots if s < JUMP_CUT_MAX_GAP)
    return {
        "avg_shot_sec": round(statistics.fmean(shots), 3),
        "median_shot_sec": round(statistics.median(shots), 3),
        "min_shot_sec": round(min(shots), 3),
        "jump_cut_count": jump_cuts,
        "has_jump_cuts": 1 if jump_cuts >= 2 else 0,
    }


def speech_spans(words: Sequence[Dict[str, Any]], pad: float = 0.15) -> List[Tuple[float, float]]:
    """Merged [start, end] spans where speech is happening."""
    spans: List[Tuple[float, float]] = []
    for w in words:
        try:
            s, e = float(w["start"]) - pad, float(w["end"]) + pad
        except (KeyError, TypeError, ValueError):
            continue
        s = max(0.0, s)
        if spans and s <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], e))
        else:
            spans.append((s, e))
    return spans


def music_heuristic(
    times: np.ndarray, rms: np.ndarray, spans: Sequence[Tuple[float, float]], duration: float
) -> Dict[str, Any]:
    """Is there audio energy where there are no words?

    Speech-only audio drops to the noise floor between phrases. A bed of music
    keeps the floor lifted, so the ratio of non-speech energy to speech energy
    separates the two reasonably well without a classifier.
    """
    if times.size == 0:
        return {"music_detected": None, "music_confidence": None}

    mask = np.ones(times.shape, dtype=bool)
    for s, e in spans:
        mask &= ~((times >= s) & (times <= e))
    # Ignore the very end, where fades and encoder tails distort the floor.
    if duration:
        mask &= times < max(0.0, duration - 0.4)

    nonspeech = rms[mask]
    speech = rms[~mask]
    if nonspeech.size < 8 or speech.size < 8:
        return {"music_detected": None, "music_confidence": None,
                "nonspeech_frames": int(nonspeech.size)}

    ns_level = float(np.median(nonspeech))
    sp_level = float(np.median(speech)) or 1e-6
    ratio = ns_level / sp_level
    detected = bool(ns_level > MUSIC_RMS_FLOOR and ratio > 0.18)
    return {
        "music_detected": 1 if detected else 0,
        "music_confidence": round(min(1.0, ratio / 0.5), 3),
        "nonspeech_rms": round(ns_level, 5),
        "speech_rms": round(sp_level, 5),
        "nonspeech_frames": int(nonspeech.size),
    }


def compute_editing(
    video_path: Path,
    duration: Optional[float],
    transcript_row: Optional[Dict[str, Any]],
    *,
    scene_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    video_path = Path(video_path)
    scene_threshold = scene_threshold if scene_threshold is not None else settings.scene_threshold

    cuts = detect_scene_changes(video_path, threshold=scene_threshold)
    duration = float(duration or 0.0)
    data: Dict[str, Any] = {
        "duration_sec": round(duration, 3) if duration else None,
        "cut_count": len(cuts),
        "cuts_first_5s": sum(1 for c in cuts if c <= 5.0),
        "cuts_per_minute": round(safe_div(len(cuts) * 60.0, duration), 2) if duration else None,
        "scene_changes_json": None,
    }
    from ..utils import to_json

    data["scene_changes_json"] = to_json([round(c, 3) for c in cuts])
    data.update(shot_stats(cuts, duration))

    # --- speech-derived pacing --------------------------------------------
    words = from_json(transcript_row.get("words_json") if transcript_row else None, []) or []
    word_count = len(words) or (transcript_row or {}).get("word_count") or 0
    if duration:
        wpm = round(safe_div(word_count * 60.0, duration), 2)
        data["words_per_minute"] = wpm
        data["speaking_pace"] = pace_bucket(wpm)

    if words:
        data["dead_air_before_speech"] = round(float(words[0].get("start") or 0.0), 3)
        gaps = [
            float(b.get("start") or 0) - float(a.get("end") or 0)
            for a, b in zip(words, words[1:])
        ]
        data["longest_pause_sec"] = round(max(gaps), 3) if gaps else 0.0
    else:
        data["dead_air_before_speech"] = None
        data["longest_pause_sec"] = None

    spans = speech_spans(words)
    speech_total = sum(e - s for s, e in spans)
    if duration:
        data["speech_ratio"] = round(min(1.0, safe_div(speech_total, duration)), 4)
        data["silence_ratio"] = round(max(0.0, 1.0 - data["speech_ratio"]), 4)

    # --- audio: silence spans + music heuristic ---------------------------
    audio_json: Dict[str, Any] = {"scene_threshold": scene_threshold}
    try:
        wav = settings.audio_dir / f"{video_path.stem}.wav"
        extract_audio(video_path, wav)
        times, rms, sample_rate = audio_envelope(wav)
        audio_json["sample_rate"] = sample_rate
        audio_json["peak_rms"] = round(float(rms.max()), 5) if rms.size else None
        data.update({k: v for k, v in music_heuristic(times, rms, spans, duration).items()
                     if k in ("music_detected", "music_confidence")})
        audio_json["music"] = music_heuristic(times, rms, spans, duration)
    except Exception as exc:  # noqa: BLE001 — audio analysis is a nice-to-have
        log.warning("audio analysis failed for %s: %s", video_path.name, exc)
        audio_json["error"] = str(exc)

    try:
        silences = detect_silence(video_path)
        audio_json["silence_spans"] = [[round(s, 3), round(e, 3)] for s, e in silences]
    except Exception as exc:  # noqa: BLE001
        audio_json["silence_error"] = str(exc)

    data["audio_json"] = to_json(audio_json)
    return data


def compute_and_save_editing(conn, reel_row, transcript_row) -> Dict[str, Any]:
    from ..db import database as db

    data = compute_editing(
        Path(reel_row["local_video_path"]),
        reel_row["duration_sec"],
        dict(transcript_row) if transcript_row else None,
    )
    data["shortcode"] = reel_row["shortcode"]
    db.save_editing(conn, data)
    return data


def update_editing_from_vision(conn, shortcode: str) -> Dict[str, Any]:
    """Second pass: fold OCR/vision findings into the editing row.

    Subtitles, text overlays and zooms can only be measured once we know what
    was on screen, so they land here rather than in the deterministic pass.
    """
    from ..db import database as db

    overlays = [
        dict(r) for r in conn.execute(
            "SELECT * FROM raw_overlay_text WHERE shortcode = ?", (shortcode,)
        )
    ]
    # Denominator must be frames the vision model actually looked at, not all
    # extracted frames — otherwise the ratio is systematically underestimated.
    frames_seen = conn.execute(
        "SELECT COUNT(DISTINCT t_sec) c FROM raw_visual_frames WHERE shortcode = ?", (shortcode,)
    ).fetchone()["c"]

    subtitle_rows = [o for o in overlays if o.get("is_subtitle")]
    overlay_rows = [o for o in overlays if not o.get("is_subtitle")]
    subtitle_frames = sum(o.get("frame_count") or 0 for o in subtitle_rows)

    data: Dict[str, Any] = {
        "shortcode": shortcode,
        "has_text_overlays": 1 if overlay_rows else 0,
        "text_overlay_count": len(overlay_rows),
        "subtitle_frame_ratio": (
            round(min(1.0, safe_div(subtitle_frames, frames_seen)), 3) if frames_seen else None
        ),
    }
    # Word-chunk captions produce MANY short-lived rows; a row count is a more
    # robust signal than frame coverage alone.
    if frames_seen:
        data["has_subtitles"] = 1 if (
            len(subtitle_rows) >= 5 or (data["subtitle_frame_ratio"] or 0) >= 0.35
        ) else 0
    else:
        data["has_subtitles"] = None

    # Zoom proxy: shot-scale changes between adjacent sampled frames that are
    # NOT explained by a hard cut. Approximate by construction; documented as such.
    vis_frames = [
        dict(r) for r in conn.execute(
            "SELECT t_sec, labels_json FROM raw_visual_frames WHERE shortcode = ? ORDER BY t_sec",
            (shortcode,),
        )
    ]
    editing_row = conn.execute(
        "SELECT scene_changes_json FROM raw_editing WHERE shortcode = ?", (shortcode,)
    ).fetchone()
    cuts = from_json(editing_row["scene_changes_json"] if editing_row else None, []) or []

    zooms = 0
    prev_shot, prev_t = None, None
    for f in vis_frames:
        labels = from_json(f.get("labels_json"), {}) or {}
        shot = (labels.get("scene") or {}).get("shot_type") or labels.get("shot_type")
        t = float(f.get("t_sec") or 0.0)
        if prev_shot and shot and shot != prev_shot:
            crossed_cut = any(prev_t < c <= t for c in cuts)
            if not crossed_cut:
                zooms += 1
        prev_shot, prev_t = shot or prev_shot, t
    data["zoom_events"] = zooms
    data["has_zooms"] = 1 if zooms >= 1 else 0

    db.save_editing(conn, data)
    return data
