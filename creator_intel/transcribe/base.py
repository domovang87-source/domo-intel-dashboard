"""Transcription contract + the verbatim result container."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from ..utils import fmt_ts, safe_div


@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "probability": round(self.probability, 4) if self.probability is not None else None,
        }


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "avg_logprob": self.avg_logprob,
            "no_speech_prob": self.no_speech_prob,
            "words": [w.as_dict() for w in self.words],
        }


@dataclass
class TranscriptResult:
    """The verbatim record of what was said.

    `text` is exactly what the model emitted — filler words, false starts,
    repeats, slang and profanity included. Nothing downstream is allowed to
    rewrite it; interpretations live in the `analysis` table instead.
    """

    engine: str
    model: str
    language: Optional[str]
    language_prob: Optional[float]
    segments: List[Segment]
    audio_duration: Optional[float] = None
    compute_type: Optional[str] = None
    duration_of_run: Optional[float] = None

    @property
    def words(self) -> List[Word]:
        return [w for seg in self.segments for w in seg.words]

    @property
    def text(self) -> str:
        return "".join(seg.text for seg in self.segments).strip()

    @property
    def timestamped_text(self) -> str:
        return "\n".join(
            f"[{fmt_ts(s.start)} -> {fmt_ts(s.end)}] {s.text.strip()}" for s in self.segments
        )

    def to_row(self, shortcode: str) -> Dict[str, Any]:
        from ..utils import to_json

        words = self.words
        speech_duration = sum(max(0.0, s.end - s.start) for s in self.segments)
        word_count = len(words) or len(self.text.split())
        confs = [w.probability for w in words if w.probability is not None]
        return {
            "shortcode": shortcode,
            "engine": self.engine,
            "model": self.model,
            "compute_type": self.compute_type,
            "language": self.language,
            "language_prob": self.language_prob,
            "text": self.text,
            "timestamped_text": self.timestamped_text,
            "segments_json": to_json([s.as_dict() for s in self.segments]),
            "words_json": to_json([w.as_dict() for w in words]),
            "word_count": word_count,
            "segment_count": len(self.segments),
            "speech_duration": round(speech_duration, 3),
            "audio_duration": self.audio_duration,
            "wpm": round(safe_div(word_count * 60.0, speech_duration), 2),
            "avg_word_conf": round(sum(confs) / len(confs), 4) if confs else None,
            "duration_of_run": self.duration_of_run,
        }


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        ...
