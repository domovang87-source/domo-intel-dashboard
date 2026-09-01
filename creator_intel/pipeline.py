"""Full-pipeline orchestration.

    ingest -> probe -> transcribe -> forensics -> vision -> analyze -> embed

Each stage is independently resumable and records its own job_state rows, so
killing the process at any point and re-running `run` picks up where it stopped.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .config import settings
from .db import database as db
from .logging_setup import get_logger
from .utils import to_json, utcnow_iso

log = get_logger(__name__)

STAGE_ORDER = ["ingest", "probe", "transcribe", "forensics", "vision", "analyze", "embed"]


def _log_run(conn, command: str, stats: Dict[str, Any], started: str, ok: bool, error=None) -> None:
    conn.execute(
        "INSERT INTO runs (command, started_at, ended_at, ok, stats_json, error) VALUES (?,?,?,?,?,?)",
        (command, started, utcnow_iso(), 1 if ok else 0, to_json(stats), error),
    )
    conn.commit()


def run_pipeline(
    conn,
    *,
    limit: Optional[int] = None,
    force: bool = False,
    skip: Optional[list] = None,
    only: Optional[list] = None,
    username: Optional[str] = None,
    kind: str = "reels",
    use_tesseract: bool = False,
    stop_after_known: int = 0,
) -> Dict[str, Any]:
    """Run every stage in order. Failures in one stage do not stop the rest."""
    from .analysis.classify import analyze_reels
    from .forensics import run_forensics, run_vision
    from .ingest.instagram import ingest, probe_and_store_media_facts
    from .transcribe import transcribe_reels
    from .voice.embeddings import embed_reels

    skip = set(skip or [])
    only = set(only or [])
    stages = [s for s in STAGE_ORDER if s not in skip and (not only or s in only)]

    started = utcnow_iso()
    t0 = time.time()
    stats: Dict[str, Any] = {"stages": {}}
    log.info("pipeline starting: %s", " -> ".join(stages))

    for stage in stages:
        stage_start = time.time()
        try:
            if stage == "ingest":
                result = ingest(
                    conn, username=username, limit=limit, kind=kind,
                    stop_after_known=stop_after_known,
                ).as_dict()
            elif stage == "probe":
                result = {"probed": probe_and_store_media_facts(conn, limit=limit, force=force)}
            elif stage == "transcribe":
                result = transcribe_reels(conn, limit=limit, force=force)
            elif stage == "forensics":
                result = run_forensics(conn, limit=limit, force=force)
            elif stage == "vision":
                result = run_vision(conn, limit=limit, force=force, use_tesseract=use_tesseract)
            elif stage == "analyze":
                result = analyze_reels(conn, limit=limit, force=force)
            elif stage == "embed":
                result = embed_reels(conn, limit=limit, force=force)
            else:
                continue
            result["seconds"] = round(time.time() - stage_start, 1)
            stats["stages"][stage] = result
            log.info("stage %-10s done in %ss: %s", stage, result["seconds"], result)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 — a broken stage should not abort the pipeline
            log.exception("stage %s failed", stage)
            stats["stages"][stage] = {"error": str(exc), "seconds": round(time.time() - stage_start, 1)}

    db.sync_fts(conn)
    conn.commit()

    stats["total_seconds"] = round(time.time() - t0, 1)
    stats["db"] = db.db_stats(conn)
    _log_run(conn, "run", stats, started, ok=True)
    log.info("pipeline finished in %ss — %s", stats["total_seconds"], stats["db"])
    return stats


def run_batched(
    conn,
    *,
    batch: int = 20,
    max_rounds: int = 200,
    use_tesseract: bool = False,
) -> Dict[str, Any]:
    """Interleaved processing: push BATCHES of reels through every stage.

    Instead of transcribing the whole backlog before anything gets analysed,
    each round takes the `batch` newest unprocessed reels through
    transcribe -> forensics -> vision -> analyze -> embed, so finished,
    fully-labelled reels appear in the dashboard continuously. Safe to run
    while a separate ingest/download process is still adding reels — each
    round re-queries the database and picks up whatever has arrived.
    """
    from .analysis.classify import analyze_reels
    from .forensics import run_forensics, run_vision
    from .ingest.instagram import probe_and_store_media_facts
    from .transcribe import transcribe_reels
    from .voice.embeddings import embed_reels

    started = utcnow_iso()
    t0 = time.time()
    totals: Dict[str, int] = {"rounds": 0, "transcribed": 0, "analyzed": 0, "vision": 0, "embedded": 0}

    for round_no in range(1, max_rounds + 1):
        round_work = 0
        try:
            probe_and_store_media_facts(conn)

            tr = transcribe_reels(conn, limit=batch)
            round_work += tr.get("done", 0)
            totals["transcribed"] += tr.get("done", 0)

            # forensics is fast + local: always fully catch up
            run_forensics(conn)

            vi = run_vision(conn, limit=batch, use_tesseract=use_tesseract)
            round_work += vi.get("done", 0)
            totals["vision"] += vi.get("done", 0)

            an = analyze_reels(conn)
            round_work += an.get("done", 0)
            totals["analyzed"] += an.get("done", 0)

            em = embed_reels(conn)
            totals["embedded"] += em.get("embedded", 0)

            if an.get("done"):
                try:
                    from .analysis.topics import merge_topics

                    merge_topics(conn)
                except Exception as exc:  # noqa: BLE001 — merging is cosmetic
                    log.warning("topic merge skipped: %s", exc)

            db.sync_fts(conn)
            conn.commit()
        except Exception:  # noqa: BLE001 — a bad round must not end the loop
            log.exception("batch round %s hit an error; continuing", round_no)

        totals["rounds"] = round_no
        stats_now = db.db_stats(conn)
        log.info(
            "=== batch round %s done: %s reels advanced | library: %s/%s transcribed, %s analyzed ===",
            round_no, round_work, stats_now["transcripts"], stats_now["reels"], stats_now["analysis"],
        )
        if round_work == 0:
            # Nothing to do right now — but a downloader may still be adding
            # reels. Idle-wait a few rounds before concluding we're done.
            idle_rounds = totals.get("_idle", 0) + 1
            totals["_idle"] = idle_rounds
            if idle_rounds >= 5:
                log.info("no new work after %s idle checks; batch loop finished", idle_rounds)
                break
            log.info("caught up with the downloader; waiting 60s for new reels (%s/5)", idle_rounds)
            time.sleep(60)
        else:
            totals["_idle"] = 0

    totals.pop("_idle", None)
    totals["total_seconds"] = round(time.time() - t0, 1)
    totals["db"] = db.db_stats(conn)
    _log_run(conn, f"run-batched({batch})", totals, started, ok=True)
    return totals


def status(conn) -> Dict[str, Any]:
    """What is in the database and what still needs doing."""
    counts = db.db_stats(conn)
    pending = {
        "transcribe": len(db.reels_needing(conn, "transcribe")),
        "forensics": len(db.reels_needing(conn, "forensics")),
        "vision": len(db.reels_needing(conn, "vision")),
        "analyze": len(db.reels_needing(conn, "analyze")),
        "embed": len(db.reels_needing(conn, "embed")),
    }
    errors = [
        dict(r)
        for r in conn.execute(
            "SELECT shortcode, stage, error, attempts, updated_at FROM job_state "
            "WHERE status = 'error' ORDER BY updated_at DESC LIMIT 20"
        )
    ]
    last_run = conn.execute(
        "SELECT command, started_at, ended_at, ok FROM runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return {
        "counts": counts,
        "pending": pending,
        "recent_errors": errors,
        "last_run": dict(last_run) if last_run else None,
        "paths": {
            "db": str(settings.db_path),
            "media": str(settings.media_dir),
            "frames": str(settings.frames_dir),
        },
    }
