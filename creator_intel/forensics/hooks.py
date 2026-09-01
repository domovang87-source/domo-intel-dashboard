"""RAW hook extraction.

This module produces no opinions. It slices the verbatim word-level transcript
by wall-clock time and records exactly what was said and shown in the opening
seconds, so that later you can compare the literal wording

    "Guys, this is super manipulative"
    "Guys, this is a big secret"

against performance, instead of comparing two rows that both say
hook_type = "curiosity".
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..logging_setup import get_logger
from ..utils import from_json, safe_div

log = get_logger(__name__)

_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*$")
_TOKEN = re.compile(r"[a-z0-9']+")


# ---------------------------------------------------------------------------
# word-window slicing
# ---------------------------------------------------------------------------
def words_in_window(words: Sequence[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    """Words whose onset falls inside [start, end).

    Onset (not overlap) is the right rule for a hook: a word that *starts* at
    4.9s belongs to the first five seconds even if it finishes at 5.2s, because
    the viewer has already begun hearing it.
    """
    out = []
    for w in words:
        try:
            ws = float(w.get("start"))
        except (TypeError, ValueError):
            continue
        if start - 1e-9 <= ws < end:
            out.append(w)
    return out


def join_words(words: Sequence[Dict[str, Any]]) -> str:
    """Reassemble verbatim text. Whisper words carry their own leading space."""
    if not words:
        return ""
    parts = []
    for i, w in enumerate(words):
        token = w.get("word", "")
        if i > 0 and token and not token[0].isspace() and not token[0] in ",.!?;:'":
            token = " " + token
        parts.append(token)
    return "".join(parts).strip()


def first_sentence(words: Sequence[Dict[str, Any]], max_words: int = 60) -> Tuple[str, Optional[float]]:
    """The exact first spoken sentence and the time it ends."""
    if not words:
        return "", None
    acc: List[Dict[str, Any]] = []
    for w in words[:max_words]:
        acc.append(w)
        token = (w.get("word") or "").strip()
        if _SENTENCE_END.search(token) and len(acc) >= 2:
            return join_words(acc), float(w.get("end") or 0.0)
    # No terminator found (Whisper sometimes omits final punctuation).
    return join_words(acc), float(acc[-1].get("end") or 0.0) if acc else (join_words(acc), None)


def token_similarity(a: str, b: str) -> float:
    """Jaccard overlap of word tokens — a cheap, deterministic 'are these the
    same message?' score for spoken vs on-screen hooks."""
    ta = set(_TOKEN.findall((a or "").lower()))
    tb = set(_TOKEN.findall((b or "").lower()))
    if not ta or not tb:
        return 0.0
    return round(len(ta & tb) / len(ta | tb), 4)


# ---------------------------------------------------------------------------
# on-screen side
# ---------------------------------------------------------------------------
def onscreen_in_window(overlays: Sequence[Dict[str, Any]], end: float) -> List[str]:
    """Distinct overlay strings first visible before `end` seconds."""
    seen, out = set(), []
    for o in sorted(overlays, key=lambda x: (x.get("t_start") or 0.0)):
        t_start = o.get("t_start")
        if t_start is None or float(t_start) >= end:
            continue
        text = (o.get("text") or "").strip()
        key = " ".join(text.lower().split())
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def compute_hook(
    transcript_row: Optional[Dict[str, Any]],
    overlays: Sequence[Dict[str, Any]],
    *,
    hook_window: float = 5.0,
) -> Dict[str, Any]:
    words: List[Dict[str, Any]] = from_json(
        transcript_row.get("words_json") if transcript_row else None, []
    ) or []
    # Some backends return segments without word timings; degrade gracefully.
    if not words and transcript_row:
        segments = from_json(transcript_row.get("segments_json"), []) or []
        words = [
            {"word": " " + s.get("text", "").strip(), "start": s.get("start"), "end": s.get("end")}
            for s in segments
        ]

    result: Dict[str, Any] = {}

    if words:
        w0 = words[0]
        result["time_to_first_word"] = round(float(w0.get("start") or 0.0), 3)
        result["first_word"] = (w0.get("word") or "").strip()
        result["first_word_start"] = round(float(w0.get("start") or 0.0), 3)

        sentence, sentence_end = first_sentence(words)
        result["spoken_first_sentence"] = sentence
        result["spoken_first_sentence_end"] = round(sentence_end, 3) if sentence_end else None

        for label, end in (("spoken_0_2", 2.0), ("spoken_0_3", 3.0),
                           ("spoken_0_5", 5.0), ("spoken_0_10", 10.0)):
            result[label] = join_words(words_in_window(words, 0.0, end))

        result["spoken_first_15_words"] = join_words(words[:15])

        # The hook span: the first sentence when it lands quickly, otherwise
        # whatever was said inside the configured hook window.
        if sentence and sentence_end and sentence_end <= hook_window + 2.0:
            result["spoken_hook"] = sentence
            result["spoken_hook_end"] = round(sentence_end, 3)
        else:
            result["spoken_hook"] = join_words(words_in_window(words, 0.0, hook_window))
            result["spoken_hook_end"] = hook_window
        result["spoken_hook_word_count"] = len((result.get("spoken_hook") or "").split())

        n2 = len(words_in_window(words, 0.0, 2.0))
        n5 = len(words_in_window(words, 0.0, 5.0))
        result["words_in_first_2s"] = n2
        result["words_in_first_5s"] = n5
        result["wps_first_5s"] = round(safe_div(n5, 5.0), 3)
    else:
        result.update(
            {
                "time_to_first_word": None,
                "first_word": None,
                "spoken_first_sentence": "",
                "spoken_0_2": "", "spoken_0_3": "", "spoken_0_5": "", "spoken_0_10": "",
                "spoken_hook": "", "spoken_hook_word_count": 0,
                "words_in_first_2s": 0, "words_in_first_5s": 0, "wps_first_5s": 0.0,
            }
        )

    # --- on-screen ---------------------------------------------------------
    if overlays:
        for label, end in (("onscreen_0_2", 2.0), ("onscreen_0_3", 3.0), ("onscreen_0_5", 5.0)):
            result[label] = " | ".join(onscreen_in_window(overlays, end))
        hook_overlays = [
            o for o in overlays
            if o.get("t_start") is not None and float(o["t_start"]) < hook_window
            and not o.get("is_subtitle")
        ]
        hook_overlays.sort(key=lambda x: float(x.get("t_start") or 0.0))
        if hook_overlays:
            result["onscreen_hook"] = hook_overlays[0].get("text")
            result["onscreen_hook_first_seen"] = round(float(hook_overlays[0].get("t_start") or 0.0), 3)
        result["onscreen_hook_count"] = len(
            {" ".join((o.get("text") or "").lower().split()) for o in hook_overlays}
        )
        result["onscreen_all"] = "\n".join(
            dict.fromkeys((o.get("text") or "").strip() for o in overlays if o.get("text"))
        )

    spoken_hook = result.get("spoken_hook") or ""
    onscreen_hook = result.get("onscreen_hook") or ""
    if spoken_hook and onscreen_hook:
        sim = token_similarity(spoken_hook, onscreen_hook)
        result["hook_text_similarity"] = sim
        result["hooks_differ"] = 1 if sim < 0.6 else 0
    elif onscreen_hook and not spoken_hook:
        result["hook_text_similarity"] = 0.0
        result["hooks_differ"] = 1
    else:
        result["hook_text_similarity"] = None
        result["hooks_differ"] = None

    return result


def compute_and_save_hook(conn, shortcode: str, hook_window: float = 5.0) -> Dict[str, Any]:
    from ..db import database as db

    transcript = conn.execute(
        "SELECT * FROM transcripts WHERE shortcode = ?", (shortcode,)
    ).fetchone()
    overlays = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM raw_overlay_text WHERE shortcode = ? ORDER BY t_start", (shortcode,)
        )
    ]
    data = compute_hook(dict(transcript) if transcript else None, overlays, hook_window=hook_window)
    data["shortcode"] = shortcode
    db.save_hook(conn, data)
    return data
