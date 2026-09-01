"""Import Instagram Insights exports (views, shares, saves, reach).

Instagram's web API no longer returns view counts for Reels, and it has never
exposed shares or saves. Those numbers only exist in Instagram's own analytics
for professional accounts. This module merges them in from the CSV export:

    Meta Business Suite (business.facebook.com) -> Insights -> Content
      -> set the date range -> Export data (CSV)

The export format shifts between Meta releases and languages, so parsing is
defensive: headers are matched fuzzily, numbers are cleaned, and each row is
matched to an archived reel by the permalink's shortcode (falling back to
publish-time proximity). Unmatched rows are reported, never guessed.

    python -m creator_intel import-insights ~/Downloads/content_export.csv
"""
from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..logging_setup import get_logger
from ..utils import to_json, utcnow_iso

log = get_logger(__name__)

_SHORTCODE_RE = re.compile(r"instagram\.com/(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)")

# Fuzzy header recognition: lowercase substrings that identify each field.
# First match wins, checked in order (so "views" won't grab "3-second views"
# before the exact form is tried).
_HEADER_PATTERNS: Dict[str, List[str]] = {
    "permalink": ["permalink", "post url", "link"],
    "post_id": ["post id", "media id"],
    "publish_time": ["publish time", "publish date", "date", "created"],
    "description": ["description", "caption", "post description"],
    "views": ["views", "plays", "video plays", "impressions"],
    "reach": ["reach", "accounts reached"],
    "likes": ["likes", "reactions"],
    "comments": ["comments"],
    "shares": ["shares"],
    "saves": ["saves", "saved"],
    "follows": ["follows", "new follows"],
}


def _clean_int(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    text = str(raw).strip().replace(",", "").replace(" ", "")
    if text in ("", "-", "—", "N/A", "n/a", "null"):
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _map_headers(fieldnames: List[str]) -> Dict[str, str]:
    """{our_field: csv_column} via case-insensitive substring matching."""
    mapping: Dict[str, str] = {}
    lowered = [(name, (name or "").strip().lower()) for name in fieldnames]
    for field, patterns in _HEADER_PATTERNS.items():
        for pattern in patterns:
            hit = next(
                (orig for orig, low in lowered
                 if pattern == low or (pattern in low and orig not in mapping.values())),
                None,
            )
            if hit:
                mapping[field] = hit
                break
    return mapping


def _parse_time(raw: Any) -> Optional[int]:
    if not raw:
        return None
    text = str(raw).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M",
        "%m/%d/%Y %I:%M %p", "%m/%d/%Y", "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(text[: len(datetime.now().strftime(fmt))], fmt)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return None


def _read_rows(csv_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Rows + header mapping, tolerating BOMs and preamble junk lines."""
    text = csv_path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()

    # Some Meta exports start with a few metadata lines before the header row.
    header_idx = 0
    for i, line in enumerate(lines[:10]):
        low = line.lower()
        if any(k in low for k in ("permalink", "publish", "description")) and ("," in line or ";" in line or "\t" in line):
            header_idx = i
            break

    body = "\n".join(lines[header_idx:])
    try:
        dialect = csv.Sniffer().sniff(body[:4000], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(body.splitlines(), dialect=dialect)
    rows = [r for r in reader if any((v or "").strip() for v in r.values())]
    mapping = _map_headers(list(reader.fieldnames or []))
    return rows, mapping


def import_insights_csv(
    conn,
    csv_path: Path,
    *,
    dry_run: bool = False,
    time_tolerance_sec: int = 90 * 60,
) -> Dict[str, Any]:
    from ..db import database as db

    csv_path = Path(csv_path).expanduser()
    if not csv_path.exists():
        raise SystemExit(f"file not found: {csv_path}")

    db._migrate(conn)
    rows, mapping = _read_rows(csv_path)
    if not rows:
        raise SystemExit(f"no data rows found in {csv_path}")
    if "permalink" not in mapping and "publish_time" not in mapping:
        raise SystemExit(
            "Could not find a Permalink or Publish time column in the CSV.\n"
            f"Columns seen: {list(rows[0].keys())[:12]}\n"
            "Export the per-post 'Content' CSV from Meta Business Suite."
        )
    log.info("parsed %s rows; recognised columns: %s", len(rows), mapping)

    reels = [dict(r) for r in conn.execute(
        "SELECT shortcode, taken_at_ts, view_count, like_count, comment_count FROM reels"
    )]
    by_shortcode = {r["shortcode"]: r for r in reels}

    stats = {"rows": len(rows), "matched": 0, "updated": 0, "unmatched": 0, "unmatched_examples": []}

    for row in rows:
        # --- match the row to an archived reel --------------------------
        # A parseable permalink is authoritative: if its shortcode is not in
        # the archive, the reel simply isn't downloaded yet — falling back to
        # time matching there would attach numbers to the WRONG reel (this
        # account posts near-daily). Time matching is a last resort for rows
        # with no usable permalink, and only within a tight window.
        shortcode = None
        permalink = row.get(mapping.get("permalink", ""), "") or ""
        m = _SHORTCODE_RE.search(permalink)
        if m:
            shortcode = m.group(1) if m.group(1) in by_shortcode else None
        elif "publish_time" in mapping:
            ts = _parse_time(row.get(mapping["publish_time"]))
            if ts:
                near = [
                    r for r in reels
                    if r["taken_at_ts"] and abs(r["taken_at_ts"] - ts) <= time_tolerance_sec
                ]
                if len(near) == 1:  # only trust an unambiguous time match
                    shortcode = near[0]["shortcode"]

        if not shortcode:
            stats["unmatched"] += 1
            if len(stats["unmatched_examples"]) < 5:
                stats["unmatched_examples"].append(
                    (permalink or str(row.get(mapping.get("description", ""), ""))[:60]).strip()
                )
            continue
        stats["matched"] += 1

        # --- collect the numbers ---------------------------------------
        values = {
            "views": _clean_int(row.get(mapping.get("views", ""))),
            "reach": _clean_int(row.get(mapping.get("reach", ""))),
            "likes": _clean_int(row.get(mapping.get("likes", ""))),
            "comments": _clean_int(row.get(mapping.get("comments", ""))),
            "shares": _clean_int(row.get(mapping.get("shares", ""))),
            "saves": _clean_int(row.get(mapping.get("saves", ""))),
            "follows": _clean_int(row.get(mapping.get("follows", ""))),
        }
        if all(v is None for v in values.values()):
            continue

        current = by_shortcode[shortcode]
        update: Dict[str, Any] = {"shortcode": shortcode, "insights_imported_at": utcnow_iso()}
        if values["views"] is not None:
            update["view_count"] = values["views"]
        if values["reach"] is not None:
            update["reach_count"] = values["reach"]
        if values["shares"] is not None:
            update["share_count"] = values["shares"]
        if values["saves"] is not None:
            update["save_count"] = values["saves"]
        if values["follows"] is not None:
            update["follows_count"] = values["follows"]
        # Likes/comments from the live web session are usually fresher than a
        # CSV export, so only fill gaps — never regress a live number.
        if values["likes"] is not None and current.get("like_count") is None:
            update["like_count"] = values["likes"]
        if values["comments"] is not None and current.get("comment_count") is None:
            update["comment_count"] = values["comments"]

        likes = update.get("like_count", current.get("like_count"))
        comments = update.get("comment_count", current.get("comment_count"))
        views = update.get("view_count", current.get("view_count"))
        if likes is not None or comments is not None:
            update["engagement_total"] = (likes or 0) + (comments or 0)
            if views:
                update["engagement_rate"] = round(update["engagement_total"] / views, 6)

        if not dry_run:
            db.upsert_reel(conn, update)
            db.record_metrics_snapshot(
                conn,
                shortcode,
                {
                    "view_count": views,
                    "like_count": likes,
                    "comment_count": comments,
                    "extra": {
                        "source": "insights_csv",
                        "shares": values["shares"],
                        "saves": values["saves"],
                        "reach": values["reach"],
                        "follows": values["follows"],
                    },
                },
            )
        stats["updated"] += 1

    if not dry_run:
        conn.commit()
    log.info(
        "insights import: %s rows, %s matched, %s updated, %s unmatched%s",
        stats["rows"], stats["matched"], stats["updated"], stats["unmatched"],
        " (dry run — nothing written)" if dry_run else "",
    )
    return stats
