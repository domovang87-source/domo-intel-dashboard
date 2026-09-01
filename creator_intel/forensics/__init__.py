"""Content forensics orchestration.

Two stages, deliberately separate so you can run the free one on everything and
the paid one selectively:

  forensics  — free & local: frame extraction, cut detection, audio analysis,
               pacing, and the exact spoken hook windows.
  vision     — costs API calls: on-screen text OCR and scene labelling, then a
               backfill pass that completes the hook and editing rows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..config import settings
from ..logging_setup import get_logger
from ..media.ffmpeg import extract_frames, plan_frame_timestamps
from ..utils import to_json
from .editing import compute_and_save_editing, update_editing_from_vision  # noqa: F401
from .hooks import compute_and_save_hook, compute_hook  # noqa: F401
from .vision import (  # noqa: F401
    BATCH_SIZE,
    aggregate_overlays,
    analyze_frame_batch,
    ocr_frames_tesseract,
    reclassify_subtitles,
    rollup_visual,
    tesseract_available,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# stage 1 — local, free
# ---------------------------------------------------------------------------
def extract_and_record_frames(conn, reel_row, *, overwrite: bool = False) -> List[Dict]:
    from ..db import database as db

    shortcode = reel_row["shortcode"]
    duration = reel_row["duration_sec"]
    if not duration:
        log.warning("%s has no known duration; probe it first", shortcode)
        return []

    plan = plan_frame_timestamps(
        duration,
        hook_window=settings.hook_window_sec,
        hook_interval=settings.hook_frame_interval,
        body_interval=settings.body_frame_interval,
    )
    out_dir = settings.frames_dir / shortcode
    extracted = extract_frames(
        Path(reel_row["local_video_path"]), [t for t, _ in plan], out_dir,
        prefix=shortcode, overwrite=overwrite,
    )

    phase_by_ts = {round(t, 2): phase for t, phase in plan}
    for f in extracted:
        f["phase"] = phase_by_ts.get(round(float(f["t_sec"]), 2), "body")
    db.save_frames(conn, shortcode, extracted)
    return extracted


def run_forensics(
    conn,
    *,
    limit: Optional[int] = None,
    force: bool = False,
    shortcode: Optional[str] = None,
) -> Dict[str, int]:
    from ..db import database as db

    if shortcode:
        row = db.get_reel(conn, shortcode)
        rows = [row] if row else []
    else:
        rows = db.reels_needing(conn, "forensics", limit=limit, force=force)

    if not rows:
        log.info("nothing needs forensics")
        return {"done": 0, "skipped": 0, "errors": 0}

    log.info("running local forensics on %s reel(s)", len(rows))
    stats = {"done": 0, "skipped": 0, "errors": 0}

    for i, row in enumerate(rows, 1):
        sc = row["shortcode"]
        if not row["local_video_path"] or not Path(row["local_video_path"]).exists():
            stats["skipped"] += 1
            db.set_job(conn, sc, "forensics", "skipped", error="no local video")
            conn.commit()
            continue
        try:
            db.set_job(conn, sc, "forensics", "running")
            conn.commit()

            frames = extract_and_record_frames(conn, row, overwrite=force)
            transcript = db.get_transcript(conn, sc)
            editing = compute_and_save_editing(conn, row, transcript)
            hook = compute_and_save_hook(conn, sc, hook_window=settings.hook_window_sec)

            db.set_job(
                conn, sc, "forensics", "done",
                detail={"frames": len(frames), "cuts": editing.get("cut_count")},
            )
            conn.commit()
            stats["done"] += 1
            log.info(
                '[%s/%s] %s — %s frames, %s cuts, first words @%.2fs: "%s"',
                i, len(rows), sc, len(frames), editing.get("cut_count"),
                hook.get("time_to_first_word") or 0.0,
                (hook.get("spoken_0_2") or "")[:70],
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("[%s/%s] forensics failed for %s", i, len(rows), sc)
            db.set_job(conn, sc, "forensics", "error", error=str(exc))
            conn.commit()
            stats["errors"] += 1

    log.info("forensics: %s", stats)
    return stats


# ---------------------------------------------------------------------------
# stage 2 — vision (API)
# ---------------------------------------------------------------------------
def select_vision_frames(conn, shortcode: str, max_frames: int) -> List[Dict]:
    """Every hook frame first — the opening is what we care most about —
    then an even spread of body frames up to the budget."""
    rows = [
        dict(r)
        for r in conn.execute(
            "SELECT t_sec, path, phase FROM raw_frames WHERE shortcode = ? ORDER BY t_sec",
            (shortcode,),
        )
        if Path(dict(r)["path"]).exists()
    ]
    hook = [r for r in rows if r["phase"] == "hook"]
    body = [r for r in rows if r["phase"] != "hook"]

    if len(hook) > max_frames:
        step = max(1, len(hook) // max_frames)
        hook = hook[::step][:max_frames]
    budget = max(0, max_frames - len(hook))
    if budget and body:
        step = max(1, len(body) // budget)
        body = body[::step][:budget]
    else:
        body = []
    return sorted(hook + body, key=lambda r: r["t_sec"])


def run_vision(
    conn,
    *,
    limit: Optional[int] = None,
    force: bool = False,
    shortcode: Optional[str] = None,
    use_tesseract: bool = False,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, int]:
    from ..db import database as db

    if shortcode:
        row = db.get_reel(conn, shortcode)
        rows = [row] if row else []
    else:
        rows = db.reels_needing(conn, "vision", limit=limit, force=force)

    if not rows:
        log.info("nothing needs vision analysis")
        return {"done": 0, "skipped": 0, "errors": 0}

    model = model or settings.vision_model
    if use_tesseract and not tesseract_available():
        log.warning("tesseract not installed; falling back to the vision model")
        use_tesseract = False

    log.info("running vision on %s reel(s) via %s", len(rows), "tesseract" if use_tesseract else model)
    stats = {"done": 0, "skipped": 0, "errors": 0}

    for i, row in enumerate(rows, 1):
        sc = row["shortcode"]
        try:
            frames = select_vision_frames(conn, sc, settings.max_vision_frames)
            if not frames:
                log.warning("[%s/%s] %s has no extracted frames; run forensics first", i, len(rows), sc)
                db.set_job(conn, sc, "vision", "skipped", error="no frames")
                conn.commit()
                stats["skipped"] += 1
                continue

            db.set_job(conn, sc, "vision", "running")
            conn.commit()

            if use_tesseract:
                results = ocr_frames_tesseract(frames)
                source_model = "tesseract"
            else:
                results = []
                for start in range(0, len(frames), BATCH_SIZE):
                    batch = frames[start : start + BATCH_SIZE]
                    results.extend(analyze_frame_batch(batch, provider_name=provider_name, model=model))
                source_model = model

            db.save_visual_frames(
                conn,
                sc,
                [
                    {
                        "t_sec": r.get("t_sec"),
                        "frame_path": r.get("frame_path"),
                        "labels_json": to_json({"texts": r.get("texts"), "scene": r.get("scene")}),
                        "model": source_model,
                    }
                    for r in results
                ],
            )

            overlays = aggregate_overlays(results, source_model)
            transcript_row = db.get_transcript(conn, sc)
            overlays = reclassify_subtitles(
                overlays, transcript_row["text"] if transcript_row else None
            )
            db.replace_overlay_text(conn, sc, overlays)

            if not use_tesseract:
                visual = rollup_visual(results, source_model)
                visual["shortcode"] = sc
                db.save_visual(conn, visual)

            # Backfill the parts of hook/editing that need on-screen knowledge.
            compute_and_save_hook(conn, sc, hook_window=settings.hook_window_sec)
            update_editing_from_vision(conn, sc)
            db.sync_fts(conn, sc)

            db.set_job(conn, sc, "vision", "done", detail={"frames": len(frames), "overlays": len(overlays)})
            conn.commit()
            stats["done"] += 1
            first_overlay = overlays[0]["text"] if overlays else ""
            log.info(
                '[%s/%s] %s — %s frames, %s overlay string(s), first on screen: "%s"',
                i, len(rows), sc, len(frames), len(overlays), first_overlay[:60],
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("[%s/%s] vision failed for %s", i, len(rows), sc)
            db.set_job(conn, sc, "vision", "error", error=str(exc))
            conn.commit()
            stats["errors"] += 1

    log.info("vision: %s", stats)
    return stats
