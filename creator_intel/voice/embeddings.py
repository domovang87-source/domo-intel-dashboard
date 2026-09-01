"""Creator-voice foundation: embed each Reel so it can be retrieved later.

Vectors live in the `embeddings` table as float32 blobs and are searched with
numpy in-process. At a few thousand Reels this is instantaneous and has zero
operational cost; if the library ever outgrows it, swapping in sqlite-vec or a
vector DB means changing only `search.py`, because nothing else knows how the
lookup is done.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..config import settings
from ..logging_setup import get_logger
from ..utils import sha1, truncate

log = get_logger(__name__)

MAX_CHARS = 6000  # keep well inside the embedding model's context


def build_source_text(row) -> str:
    """What actually represents a video for similarity purposes.

    The hook is repeated at the top on purpose: when you later ask "find the 20
    videos most like this idea", you want openings to dominate the match, since
    those are the examples you will feed back to an LLM as voice references.
    """
    get = (lambda k: row[k]) if hasattr(row, "keys") else row.get
    parts = []

    hook = get("spoken_hook") if "spoken_hook" in row.keys() else None
    if hook:
        parts.append(f"HOOK: {hook}")
    topic = get("main_topic") if "main_topic" in row.keys() else None
    if topic:
        subtopic = get("subtopic") if "subtopic" in row.keys() else None
        parts.append(f"TOPIC: {topic}{' / ' + subtopic if subtopic else ''}")
    caption = get("caption")
    if caption:
        parts.append(f"CAPTION: {truncate(caption, 500)}")
    transcript = get("transcript") if "transcript" in row.keys() else get("text")
    if transcript:
        parts.append(f"TRANSCRIPT: {transcript}")

    return truncate("\n".join(parts), MAX_CHARS)


def to_blob(vector: List[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def embed_reels(
    conn,
    *,
    limit: Optional[int] = None,
    force: bool = False,
    batch_size: int = 32,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, int]:
    from ..analysis.providers import get_provider
    from ..db import database as db

    model = model or settings.embedding_model
    rows = conn.execute(
        """
        SELECT r.shortcode, r.caption, t.text AS transcript,
               h.spoken_hook, a.main_topic, a.subtopic,
               e.source_hash AS existing_hash
        FROM reels r
        JOIN transcripts t ON t.shortcode = r.shortcode
        LEFT JOIN raw_hook h ON h.shortcode = r.shortcode
        LEFT JOIN analysis a ON a.shortcode = r.shortcode
        LEFT JOIN embeddings e ON e.shortcode = r.shortcode AND e.kind = 'transcript' AND e.model = ?
        ORDER BY r.taken_at_ts DESC
        """,
        (model,),
    ).fetchall()

    pending = []
    for row in rows:
        text = build_source_text(row)
        if not text.strip():
            continue
        digest = sha1(text)
        if not force and row["existing_hash"] == digest:
            continue  # unchanged since last time
        pending.append((row["shortcode"], text, digest))
        if limit and len(pending) >= limit:
            break

    if not pending:
        log.info("embeddings are up to date")
        return {"embedded": 0, "skipped": len(rows)}

    provider = get_provider(provider_name)
    log.info("embedding %s reel(s) with %s", len(pending), model)
    done = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = provider.embed([t for _, t, _ in batch], model=model)
        for (shortcode, text, digest), vector in zip(batch, vectors):
            db.save_embedding(
                conn, shortcode, "transcript", model, to_blob(vector), len(vector), text, digest
            )
            db.set_job(conn, shortcode, "embed", "done")
            done += 1
        conn.commit()
        log.info("  embedded %s/%s", min(start + batch_size, len(pending)), len(pending))

    return {"embedded": done, "skipped": len(rows) - len(pending)}
