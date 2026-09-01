"""Transcription stage: video -> 16kHz wav -> verbatim transcript in SQLite."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from ..config import settings
from ..logging_setup import get_logger
from .base import Transcriber, TranscriptResult  # noqa: F401

log = get_logger(__name__)


def get_transcriber(backend: Optional[str] = None) -> Transcriber:
    backend = (backend or settings.transcribe_backend).lower()
    if backend in ("faster_whisper", "faster-whisper", "local"):
        from .faster_whisper_backend import FasterWhisperTranscriber

        return FasterWhisperTranscriber()
    if backend in ("openai", "api", "whisper-1"):
        from .openai_backend import OpenAIWhisperTranscriber

        return OpenAIWhisperTranscriber()
    raise ValueError(f"unknown transcription backend: {backend}")


def transcribe_reels(
    conn,
    *,
    limit: Optional[int] = None,
    force: bool = False,
    backend: Optional[str] = None,
    shortcode: Optional[str] = None,
) -> Dict[str, int]:
    from ..db import database as db
    from ..media.ffmpeg import extract_audio

    if shortcode:
        row = db.get_reel(conn, shortcode)
        rows = [row] if row else []
    else:
        rows = db.reels_needing(conn, "transcribe", limit=limit, force=force)

    if not rows:
        log.info("nothing to transcribe")
        return {"done": 0, "skipped": 0, "errors": 0}

    transcriber = get_transcriber(backend)
    log.info("transcribing %s reel(s) with %s", len(rows), transcriber.name)
    stats = {"done": 0, "skipped": 0, "errors": 0}

    for i, row in enumerate(rows, 1):
        sc = row["shortcode"]
        video_path = row["local_video_path"]
        if not video_path or not Path(video_path).exists():
            log.warning("[%s/%s] %s has no local video; skipping", i, len(rows), sc)
            db.set_job(conn, sc, "transcribe", "skipped", error="no local video")
            stats["skipped"] += 1
            conn.commit()
            continue

        try:
            db.set_job(conn, sc, "transcribe", "running")
            conn.commit()

            audio_path = settings.audio_dir / f"{sc}.wav"
            extract_audio(Path(video_path), audio_path)

            result = transcriber.transcribe(audio_path)
            written = db.save_transcript(conn, result.to_row(sc), force=force)

            db.upsert_reel(conn, {"shortcode": sc, "local_audio_path": str(audio_path)})
            db.sync_fts(conn, sc)
            db.set_job(
                conn, sc, "transcribe", "done",
                detail={"words": len(result.words), "seconds": result.duration_of_run},
            )
            conn.commit()

            if written:
                stats["done"] += 1
                log.info(
                    "[%s/%s] %s — %s words, %.1fs of audio, %.1fs to transcribe",
                    i, len(rows), sc, len(result.words),
                    result.audio_duration or 0.0, result.duration_of_run or 0.0,
                )
            else:
                stats["skipped"] += 1
        except Exception as exc:  # noqa: BLE001
            log.exception("[%s/%s] transcription failed for %s", i, len(rows), sc)
            db.set_job(conn, sc, "transcribe", "error", error=str(exc))
            conn.commit()
            stats["errors"] += 1

    log.info("transcription: %s", stats)
    return stats
