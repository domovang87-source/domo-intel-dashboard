"""Small shared helpers."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Optional


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_json(value: Any) -> Optional[str]:
    """Serialise for a SQLite JSON column. None stays NULL."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=str)


def from_json(raw: Any, default: Any = None) -> Any:
    if raw is None or raw == "":
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def file_sha1(path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


_HASHTAG_RE = re.compile(r"#(\w+)", re.UNICODE)
_MENTION_RE = re.compile(r"@([A-Za-z0-9._]+)")


def extract_hashtags(text: str) -> list:
    return _HASHTAG_RE.findall(text or "")


def extract_mentions(text: str) -> list:
    return _MENTION_RE.findall(text or "")


def fmt_ts(seconds: float) -> str:
    """0 -> '00:00.00', used for the human-readable timestamped transcript."""
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    try:
        if not denominator:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def json_from_llm(raw: str) -> Any:
    """Parse JSON out of an LLM response that may be fenced or prefixed."""
    if raw is None:
        raise ValueError("empty LLM response")
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the outermost {...} / [...] span.
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = text.find(opener), text.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        raise
