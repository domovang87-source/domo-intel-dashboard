"""SQLite access layer.

Everything that touches the database goes through here so the storage engine
stays swappable and so the "never silently overwrite" rule has one place to
live. Writes are additive-by-default: existing transcripts/analysis rows are
only replaced when the caller explicitly passes force=True, and even then the
old row is archived by a trigger first.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..config import settings
from ..logging_setup import get_logger
from ..utils import to_json, utcnow_iso

log = get_logger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Stages tracked in job_state, in pipeline order.
STAGES = ("ingest", "media", "transcribe", "forensics", "vision", "analyze", "embed")


# ---------------------------------------------------------------------------
# connection
# ---------------------------------------------------------------------------
def connect(db_path: Optional[Path] = None, *, read_only: bool = False) -> sqlite3.Connection:
    path = Path(db_path or settings.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if read_only and path.exists():
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0)
    else:
        conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if not read_only:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created.

    SQLite's CREATE TABLE IF NOT EXISTS ignores new columns on existing
    tables, so late additions are ALTERed in here. Purely additive.
    """
    existing = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='reels'"
    ).fetchone()
    if not existing:
        return
    have = {r["name"] for r in conn.execute("PRAGMA table_info(reels)")}
    for col, decl in [
        ("share_count", "INTEGER"),
        ("save_count", "INTEGER"),
        ("reach_count", "INTEGER"),
        ("follows_count", "INTEGER"),
        ("insights_imported_at", "TEXT"),
    ]:
        if col not in have:
            conn.execute(f"ALTER TABLE reels ADD COLUMN {col} {decl}")
            log.info("migration: added reels.%s", col)


def init_db(db_path: Optional[Path] = None) -> Path:
    """Create the schema. Safe to run repeatedly; never drops data."""
    settings.ensure_dirs()
    path = Path(db_path or settings.db_path)
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect(path) as conn:
        _migrate(conn)
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_initialized_at', ?) "
            "ON CONFLICT(key) DO NOTHING",
            (utcnow_iso(),),
        )
        conn.commit()
    log.info("database ready at %s", path)
    return path


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------
def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]


def _upsert(
    conn: sqlite3.Connection,
    table: str,
    pk: Sequence[str],
    data: Dict[str, Any],
    *,
    update: bool = True,
) -> None:
    """INSERT ... ON CONFLICT DO UPDATE, restricted to real columns.

    Uses ON CONFLICT (not INSERT OR REPLACE) precisely so that the archive
    triggers fire and old rows are preserved.
    """
    cols = [c for c in _table_columns(conn, table) if c in data]
    if not cols:
        raise ValueError(f"no matching columns for {table}: {sorted(data)[:8]}")
    placeholders = ", ".join("?" for _ in cols)
    collist = ", ".join(cols)
    sql = f"INSERT INTO {table} ({collist}) VALUES ({placeholders})"
    if update:
        setters = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in pk)
        if setters:
            sql += f" ON CONFLICT({', '.join(pk)}) DO UPDATE SET {setters}"
        else:
            sql += f" ON CONFLICT({', '.join(pk)}) DO NOTHING"
    else:
        sql += f" ON CONFLICT({', '.join(pk)}) DO NOTHING"
    conn.execute(sql, [data[c] for c in cols])


def _row_exists(conn: sqlite3.Connection, table: str, shortcode: str) -> bool:
    cur = conn.execute(f"SELECT 1 FROM {table} WHERE shortcode = ? LIMIT 1", (shortcode,))
    return cur.fetchone() is not None


# ---------------------------------------------------------------------------
# reels
# ---------------------------------------------------------------------------
def upsert_reel(conn: sqlite3.Connection, data: Dict[str, Any]) -> bool:
    """Insert or refresh a reel row. Returns True if this reel is new."""
    shortcode = data["shortcode"]
    now = utcnow_iso()
    existing = conn.execute(
        "SELECT shortcode, first_seen_at, local_video_path FROM reels WHERE shortcode = ?",
        (shortcode,),
    ).fetchone()
    is_new = existing is None

    payload = dict(data)
    payload["updated_at"] = now
    payload.setdefault("metadata_updated_at", now)
    if is_new:
        payload["first_seen_at"] = now
    else:
        payload.pop("first_seen_at", None)
        # Never blank out a known local file with a None from a metadata-only refresh.
        if not payload.get("local_video_path") and existing["local_video_path"]:
            payload.pop("local_video_path", None)

    payload = {k: v for k, v in payload.items() if v is not None or is_new}
    _upsert(conn, "reels", ["shortcode"], payload)
    return is_new


def record_metrics_snapshot(conn: sqlite3.Connection, shortcode: str, metrics: Dict[str, Any]) -> None:
    """Append a performance snapshot, but only when a counter actually moved."""
    last = conn.execute(
        "SELECT view_count, play_count, like_count, comment_count FROM metrics_history "
        "WHERE shortcode = ? ORDER BY id DESC LIMIT 1",
        (shortcode,),
    ).fetchone()
    keys = ("view_count", "play_count", "like_count", "comment_count")
    current = tuple(metrics.get(k) for k in keys)
    if last is not None and tuple(last[k] for k in keys) == current:
        return
    conn.execute(
        "INSERT INTO metrics_history (shortcode, captured_at, view_count, play_count, "
        "like_count, comment_count, metrics_json) VALUES (?,?,?,?,?,?,?)",
        (
            shortcode,
            utcnow_iso(),
            metrics.get("view_count"),
            metrics.get("play_count"),
            metrics.get("like_count"),
            metrics.get("comment_count"),
            to_json(metrics.get("extra")),
        ),
    )


def get_reel(conn: sqlite3.Connection, shortcode: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM reels WHERE shortcode = ?", (shortcode,)).fetchone()


def iter_reels(conn: sqlite3.Connection, limit: Optional[int] = None) -> List[sqlite3.Row]:
    sql = "SELECT * FROM reels ORDER BY taken_at_ts DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return list(conn.execute(sql))


def reels_needing(
    conn: sqlite3.Connection,
    stage: str,
    *,
    limit: Optional[int] = None,
    force: bool = False,
    require_video: bool = True,
) -> List[sqlite3.Row]:
    """Reels that still need `stage`, i.e. the resumability query.

    force=True returns everything eligible regardless of prior completion.
    """
    table_for_stage = {
        "transcribe": "transcripts",
        "forensics": "raw_hook",
        "vision": "raw_visual",
        "analyze": "analysis",
    }
    where = ["1=1"]
    if require_video:
        where.append("r.local_video_path IS NOT NULL AND r.local_video_path <> ''")
    # Highest-viewed reels first (they're the most valuable to analyse), then
    # unknown-view reels newest-first.
    order = " ORDER BY (r.view_count IS NULL), r.view_count DESC, r.taken_at_ts DESC"
    if stage == "embed":
        # needs a transcript, and either no embedding or a stale one
        sql = (
            "SELECT r.* FROM reels r JOIN transcripts t ON t.shortcode = r.shortcode "
            "LEFT JOIN embeddings e ON e.shortcode = r.shortcode AND e.kind='transcript' "
            "WHERE " + " AND ".join(where)
        )
        if not force:
            sql += " AND e.shortcode IS NULL"
        sql += order
    elif stage in table_for_stage:
        target = table_for_stage[stage]
        sql = f"SELECT r.* FROM reels r LEFT JOIN {target} x ON x.shortcode = r.shortcode WHERE " + " AND ".join(where)
        if stage in ("analyze",) :
            sql += " AND EXISTS (SELECT 1 FROM transcripts t WHERE t.shortcode = r.shortcode)"
        if stage in ("forensics",):
            sql += " AND EXISTS (SELECT 1 FROM transcripts t WHERE t.shortcode = r.shortcode)"
        if not force:
            sql += " AND x.shortcode IS NULL"
        sql += order
    else:
        sql = "SELECT r.* FROM reels r WHERE " + " AND ".join(where) + order
    if limit:
        sql += f" LIMIT {int(limit)}"
    return list(conn.execute(sql))


# ---------------------------------------------------------------------------
# transcripts
# ---------------------------------------------------------------------------
def save_transcript(conn: sqlite3.Connection, data: Dict[str, Any], *, force: bool = False) -> bool:
    sc = data["shortcode"]
    if _row_exists(conn, "transcripts", sc) and not force:
        log.debug("transcript already present for %s; skipping (use --force to redo)", sc)
        return False
    data = dict(data)
    data["transcribed_at"] = data.get("transcribed_at") or utcnow_iso()
    _upsert(conn, "transcripts", ["shortcode"], data)
    return True


def get_transcript(conn: sqlite3.Connection, shortcode: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM transcripts WHERE shortcode = ?", (shortcode,)).fetchone()


# ---------------------------------------------------------------------------
# raw forensics
# ---------------------------------------------------------------------------
def save_hook(conn: sqlite3.Connection, data: Dict[str, Any]) -> None:
    data = dict(data)
    data["computed_at"] = utcnow_iso()
    _upsert(conn, "raw_hook", ["shortcode"], data)


def save_frames(conn: sqlite3.Connection, shortcode: str, frames: Iterable[Dict[str, Any]]) -> None:
    now = utcnow_iso()
    for f in frames:
        payload = dict(f)
        payload["shortcode"] = shortcode
        payload["extracted_at"] = now
        _upsert(conn, "raw_frames", ["shortcode", "t_sec"], payload)


def replace_overlay_text(
    conn: sqlite3.Connection, shortcode: str, rows: Iterable[Dict[str, Any]]
) -> int:
    """Overlay OCR is re-derivable from the preserved frames, so a rerun
    replaces the previous pass wholesale rather than accumulating duplicates."""
    conn.execute("DELETE FROM raw_overlay_text WHERE shortcode = ?", (shortcode,))
    now = utcnow_iso()
    count = 0
    for r in rows:
        payload = dict(r)
        payload["shortcode"] = shortcode
        payload["created_at"] = now
        cols = [c for c in _table_columns(conn, "raw_overlay_text") if c in payload]
        conn.execute(
            f"INSERT INTO raw_overlay_text ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
            [payload[c] for c in cols],
        )
        count += 1
    return count


def save_visual_frames(conn: sqlite3.Connection, shortcode: str, rows: Iterable[Dict[str, Any]]) -> None:
    now = utcnow_iso()
    for r in rows:
        payload = dict(r)
        payload["shortcode"] = shortcode
        payload["created_at"] = now
        _upsert(conn, "raw_visual_frames", ["shortcode", "t_sec", "model"], payload)


def save_visual(conn: sqlite3.Connection, data: Dict[str, Any]) -> None:
    data = dict(data)
    data["created_at"] = utcnow_iso()
    _upsert(conn, "raw_visual", ["shortcode"], data)


def save_editing(conn: sqlite3.Connection, data: Dict[str, Any]) -> None:
    data = dict(data)
    data["created_at"] = utcnow_iso()
    _upsert(conn, "raw_editing", ["shortcode"], data)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def save_analysis(conn: sqlite3.Connection, data: Dict[str, Any], *, force: bool = False) -> bool:
    sc = data["shortcode"]
    if _row_exists(conn, "analysis", sc) and not force:
        log.debug("analysis already present for %s; skipping", sc)
        return False
    data = dict(data)
    data["created_at"] = utcnow_iso()
    _upsert(conn, "analysis", ["shortcode"], data)
    return True


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------
def save_embedding(
    conn: sqlite3.Connection,
    shortcode: str,
    kind: str,
    model: str,
    vector_bytes: bytes,
    dim: int,
    source_text: str,
    source_hash: str,
) -> None:
    _upsert(
        conn,
        "embeddings",
        ["shortcode", "kind", "model"],
        {
            "shortcode": shortcode,
            "kind": kind,
            "model": model,
            "dim": dim,
            "vector": sqlite3.Binary(vector_bytes),
            "source_text": source_text,
            "source_hash": source_hash,
            "created_at": utcnow_iso(),
        },
    )


def load_embeddings(conn: sqlite3.Connection, kind: str = "transcript", model: Optional[str] = None):
    sql = "SELECT shortcode, model, dim, vector, source_text FROM embeddings WHERE kind = ?"
    params: List[Any] = [kind]
    if model:
        sql += " AND model = ?"
        params.append(model)
    return list(conn.execute(sql, params))


# ---------------------------------------------------------------------------
# job state
# ---------------------------------------------------------------------------
def set_job(
    conn: sqlite3.Connection,
    shortcode: str,
    stage: str,
    status: str,
    *,
    error: Optional[str] = None,
    detail: Any = None,
) -> None:
    now = utcnow_iso()
    conn.execute(
        """
        INSERT INTO job_state (shortcode, stage, status, attempts, error, detail_json, started_at, updated_at)
        VALUES (?, ?, ?, 1, ?, ?, ?, ?)
        ON CONFLICT(shortcode, stage) DO UPDATE SET
            status = excluded.status,
            attempts = job_state.attempts + 1,
            error = excluded.error,
            detail_json = excluded.detail_json,
            updated_at = excluded.updated_at
        """,
        (shortcode, stage, status, error, to_json(detail), now, now),
    )


def get_job(conn: sqlite3.Connection, shortcode: str, stage: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM job_state WHERE shortcode = ? AND stage = ?", (shortcode, stage)
    ).fetchone()


# ---------------------------------------------------------------------------
# full-text search
# ---------------------------------------------------------------------------
def sync_fts(conn: sqlite3.Connection, shortcode: Optional[str] = None) -> int:
    """Rebuild FTS rows (all, or for one reel). Cheap enough to just redo."""
    if shortcode:
        conn.execute("DELETE FROM transcript_fts WHERE shortcode = ?", (shortcode,))
        where, params = "WHERE r.shortcode = ?", (shortcode,)
    else:
        conn.execute("DELETE FROM transcript_fts")
        where, params = "", ()
    rows = conn.execute(
        f"""
        SELECT r.shortcode,
               COALESCE(t.text, '')    AS text,
               COALESCE(r.caption, '') AS caption,
               COALESCE((SELECT group_concat(o.text, ' | ') FROM raw_overlay_text o
                          WHERE o.shortcode = r.shortcode), '') AS overlay_text
        FROM reels r LEFT JOIN transcripts t ON t.shortcode = r.shortcode
        {where}
        """,
        params,
    ).fetchall()
    conn.executemany(
        "INSERT INTO transcript_fts (shortcode, text, caption, overlay_text) VALUES (?,?,?,?)",
        [(r["shortcode"], r["text"], r["caption"], r["overlay_text"]) for r in rows],
    )
    return len(rows)


def search_fts(conn: sqlite3.Connection, query: str, limit: int = 50) -> List[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT f.shortcode,
                   snippet(transcript_fts, 1, '[', ']', ' … ', 24) AS snippet,
                   bm25(transcript_fts) AS score
            FROM transcript_fts f
            WHERE transcript_fts MATCH ?
            ORDER BY score LIMIT ?
            """,
            (query, limit),
        )
    )


# ---------------------------------------------------------------------------
# stats / pandas bridge
# ---------------------------------------------------------------------------
def db_stats(conn: sqlite3.Connection) -> Dict[str, int]:
    def count(table: str) -> int:
        try:
            return conn.execute(f"SELECT COUNT(*) c FROM {table}").fetchone()["c"]
        except sqlite3.Error:
            return 0

    return {
        "reels": count("reels"),
        "downloaded": conn.execute(
            "SELECT COUNT(*) c FROM reels WHERE local_video_path IS NOT NULL AND local_video_path <> ''"
        ).fetchone()["c"],
        "transcripts": count("transcripts"),
        "hooks": count("raw_hook"),
        "overlay_text_rows": count("raw_overlay_text"),
        "frames": count("raw_frames"),
        "visual": count("raw_visual"),
        "editing": count("raw_editing"),
        "analysis": count("analysis"),
        "embeddings": count("embeddings"),
        "errors": conn.execute(
            "SELECT COUNT(*) c FROM job_state WHERE status = 'error'"
        ).fetchone()["c"],
    }


def query_df(sql: str, params: Sequence[Any] = (), db_path: Optional[Path] = None):
    """pandas DataFrame from a read-only connection (used by the dashboard)."""
    import pandas as pd

    with connect(db_path, read_only=True) as conn:
        return pd.read_sql_query(sql, conn, params=list(params))
