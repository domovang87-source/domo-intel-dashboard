"""Fallback transcription via the OpenAI audio API.

Only used when TRANSCRIBE_BACKEND=openai. Word-level timestamps are requested
so the hook-window slicing still works; `whisper-1` supports them.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from ..config import settings
from ..logging_setup import get_logger
from .base import Segment, TranscriptResult, Word

log = get_logger(__name__)


class OpenAIWhisperTranscriber:
    name = "openai"

    def __init__(self, model: str = "whisper-1", language: Optional[str] = None):
        self.model = model
        self.language = language if language is not None else settings.whisper_language

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        client = OpenAI(api_key=settings.openai_api_key)
        started = time.time()

        kwargs = {
            "model": self.model,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment", "word"],
        }
        if self.language:
            kwargs["language"] = self.language

        with open(audio_path, "rb") as fh:
            resp = client.audio.transcriptions.create(file=fh, **kwargs)

        data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        api_words = data.get("words") or []
        segments = []
        for idx, seg in enumerate(data.get("segments") or []):
            start, end = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
            words = [
                Word(word=w.get("word", ""), start=float(w.get("start", 0.0)), end=float(w.get("end", 0.0)))
                for w in api_words
                if start - 1e-6 <= float(w.get("start", -1)) <= end + 1e-6
            ]
            segments.append(
                Segment(
                    id=idx,
                    start=start,
                    end=end,
                    text=seg.get("text", ""),
                    words=words,
                    avg_logprob=seg.get("avg_logprob"),
                    no_speech_prob=seg.get("no_speech_prob"),
                )
            )

        if not segments and data.get("text"):
            segments = [Segment(id=0, start=0.0, end=float(data.get("duration") or 0.0),
                                text=data["text"],
                                words=[Word(w.get("word", ""), float(w.get("start", 0.0)),
                                            float(w.get("end", 0.0))) for w in api_words])]

        return TranscriptResult(
            engine=self.name,
            model=self.model,
            compute_type="api",
            language=data.get("language"),
            language_prob=None,
            segments=segments,
            audio_duration=data.get("duration"),
            duration_of_run=round(time.time() - started, 2),
        )
