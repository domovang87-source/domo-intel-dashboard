"""Local transcription with faster-whisper (CTranslate2). No API, no torch.

Settings are tuned for VERBATIM fidelity rather than readability:
  * word-level timestamps (required for the 0-2s / 0-5s hook windows)
  * VAD off by default, so silence and dead air stay measurable
  * condition_on_previous_text off, which stops Whisper from "tidying" later
    sentences to match earlier ones and cuts repetition loops
  * an initial_prompt full of filler words, which biases the decoder toward
    transcribing "um / uh / like / y'know" instead of silently dropping them
"""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..config import settings
from ..logging_setup import get_logger
from .base import Segment, TranscriptResult, Word

log = get_logger(__name__)

# Whisper conditions on this text, so writing it in a disfluent, casual,
# profanity-tolerant register makes it transcribe that register more faithfully.
VERBATIM_PROMPT = (
    "Um, so like, here's the thing, right? I mean, uh, honestly... you know what I'm saying? "
    "Yeah, no, listen — that's crazy. Bro. Shit. Damn."
)


@lru_cache(maxsize=2)
def _load_model(model_size: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    log.info(
        "loading faster-whisper model=%s device=%s compute_type=%s "
        "(first run downloads weights to ~/.cache/huggingface)",
        model_size, device, compute_type,
    )
    return WhisperModel(model_size, device=device, compute_type=compute_type)


class FasterWhisperTranscriber:
    name = "faster_whisper"

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
        vad_filter: Optional[bool] = None,
    ):
        self.model_size = model_size or settings.whisper_model
        self.device = device or settings.whisper_device
        self.compute_type = compute_type or settings.whisper_compute_type
        self.language = language if language is not None else settings.whisper_language
        self.vad_filter = settings.whisper_vad_filter if vad_filter is None else vad_filter

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        model = _load_model(self.model_size, self.device, self.compute_type)
        started = time.time()

        segments_iter, info = model.transcribe(
            str(audio_path),
            language=self.language,
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=self.vad_filter,
            initial_prompt=VERBATIM_PROMPT,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

        segments = []
        for idx, seg in enumerate(segments_iter):  # generator: work happens here
            words = [
                Word(
                    word=w.word,
                    start=float(w.start),
                    end=float(w.end),
                    probability=float(w.probability) if w.probability is not None else None,
                )
                for w in (seg.words or [])
                if w.start is not None and w.end is not None
            ]
            segments.append(
                Segment(
                    id=idx,
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text,  # verbatim, including Whisper's leading space
                    words=words,
                    avg_logprob=getattr(seg, "avg_logprob", None),
                    no_speech_prob=getattr(seg, "no_speech_prob", None),
                )
            )

        return TranscriptResult(
            engine=self.name,
            model=self.model_size,
            compute_type=self.compute_type,
            language=getattr(info, "language", None),
            language_prob=getattr(info, "language_probability", None),
            segments=segments,
            audio_duration=getattr(info, "duration", None),
            duration_of_run=round(time.time() - started, 2),
        )
