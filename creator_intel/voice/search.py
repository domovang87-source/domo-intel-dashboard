"""Semantic retrieval over your own back catalogue.

    "Find the 20 videos most similar to this idea"  -> semantic_search()
    "...now give me those as writing references"    -> voice_pack()

voice_pack() is the seam where a future script generator plugs in: it returns
the verbatim hooks and transcripts of your closest past videos, which is the
context an LLM needs to write something that sounds like you.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..config import settings
from ..logging_setup import get_logger
from .embeddings import from_blob

log = get_logger(__name__)


def _matrix(rows) -> tuple:
    vectors, shortcodes = [], []
    dim = None
    for r in rows:
        vec = from_blob(r["vector"])
        if dim is None:
            dim = vec.size
        if vec.size != dim:
            continue  # a model change left a stale row; ignore it
        vectors.append(vec)
        shortcodes.append(r["shortcode"])
    if not vectors:
        return np.zeros((0, 0)), []
    matrix = np.vstack(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms, shortcodes


def _hydrate(conn, ranked: List[tuple]) -> List[Dict[str, Any]]:
    if not ranked:
        return []
    placeholders = ",".join("?" for _ in ranked)
    rows = {
        r["shortcode"]: dict(r)
        for r in conn.execute(
            f"SELECT * FROM v_reel_full WHERE shortcode IN ({placeholders})",
            [sc for sc, _ in ranked],
        )
    }
    out = []
    for shortcode, score in ranked:
        row = rows.get(shortcode)
        if row:
            row["similarity"] = round(float(score), 4)
            out.append(row)
    return out


def semantic_search(
    conn,
    query: str,
    *,
    k: int = 20,
    model: Optional[str] = None,
    provider_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from ..analysis.providers import get_provider
    from ..db import database as db

    model = model or settings.embedding_model
    rows = db.load_embeddings(conn, kind="transcript", model=model)
    if not rows:
        log.warning("no embeddings stored for model %s — run `embed` first", model)
        return []

    matrix, shortcodes = _matrix(rows)
    if matrix.size == 0:
        return []

    provider = get_provider(provider_name)
    q = np.asarray(provider.embed([query], model=model)[0], dtype=np.float32)
    q /= np.linalg.norm(q) or 1.0

    scores = matrix @ q
    top = np.argsort(-scores)[: min(k, len(shortcodes))]
    return _hydrate(conn, [(shortcodes[i], scores[i]) for i in top])


def similar_to_reel(
    conn, shortcode: str, *, k: int = 20, model: Optional[str] = None
) -> List[Dict[str, Any]]:
    from ..db import database as db

    model = model or settings.embedding_model
    rows = db.load_embeddings(conn, kind="transcript", model=model)
    matrix, shortcodes = _matrix(rows)
    if matrix.size == 0 or shortcode not in shortcodes:
        return []
    idx = shortcodes.index(shortcode)
    scores = matrix @ matrix[idx]
    order = [i for i in np.argsort(-scores) if i != idx][:k]
    return _hydrate(conn, [(shortcodes[i], scores[i]) for i in order])


def voice_pack(conn, idea: str, *, k: int = 20, model: Optional[str] = None) -> Dict[str, Any]:
    """Retrieval bundle for future script generation.

    Returns the closest past videos with their verbatim hooks and transcripts,
    plus the performance numbers, so a generator can be told: write like these,
    and prefer the patterns from the ones that actually worked.
    """
    matches = semantic_search(conn, idea, k=k, model=model)
    examples = []
    for m in matches:
        examples.append(
            {
                "shortcode": m["shortcode"],
                "url": m.get("url"),
                "similarity": m.get("similarity"),
                "posted": m.get("taken_at_utc"),
                "views": m.get("view_count"),
                "likes": m.get("like_count"),
                "topic": m.get("main_topic"),
                "series": m.get("series"),
                "hook_type": m.get("hook_primary_type"),
                "spoken_hook": m.get("spoken_hook"),
                "spoken_0_5": m.get("spoken_0_5"),
                "onscreen_hook": m.get("onscreen_hook"),
                "transcript": m.get("transcript"),
            }
        )
    with_views = [e for e in examples if e.get("views")]
    return {
        "idea": idea,
        "matches": len(examples),
        "median_views_of_matches": float(np.median([e["views"] for e in with_views])) if with_views else None,
        "examples": examples,
    }
