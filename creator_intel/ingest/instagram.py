"""Instagram ingestion via Instaloader.

Auth model (deliberately boring):
  1. `python -m creator_intel login` logs in once interactively, handles 2FA,
     and writes a session cookie to data/session/session-<username>.
  2. Every later run loads that cookie. No password is stored or needed.
  3. If IG_PASSWORD is set in .env, unattended login is attempted as a fallback.

Resumability: we consult the DB before touching the network for media, and we
never re-download a file that already exists on disk with a sane size.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import instaloader
from instaloader import Instaloader, Post, Profile
from instaloader.exceptions import (
    BadCredentialsException,
    ConnectionException,
    LoginRequiredException,
    TwoFactorAuthRequiredException,
)

from ..config import settings
from ..logging_setup import get_logger
from ..utils import extract_hashtags, extract_mentions, to_json, utcnow_iso
from .base import IngestResult

log = get_logger(__name__)

# Instagram's `product_type` for a Reel. Older/other video posts use 'feed'
# or 'igtv'. Some responses omit it entirely.
REEL_PRODUCT_TYPES = {"clips", "reels"}


def session_file_for(username: str) -> Path:
    return settings.ig_session_dir / f"session-{username}"


def _safe(getter: Callable[[], Any], default: Any = None) -> Any:
    """Instaloader raises on fields Instagram omitted; treat those as missing."""
    try:
        value = getter()
        return default if value is None else value
    except Exception:  # noqa: BLE001 - instaloader raises a wide variety here
        return default


# ---------------------------------------------------------------------------
# login
# ---------------------------------------------------------------------------
def _new_loader(quiet: bool = True) -> Instaloader:
    settings.ensure_dirs()
    return Instaloader(
        dirname_pattern=str(settings.media_dir),
        filename_pattern="{shortcode}",
        download_videos=True,
        download_video_thumbnails=True,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,          # we persist our own JSON, uncompressed
        compress_json=False,
        post_metadata_txt_pattern="",
        quiet=quiet,
        request_timeout=30.0,
        max_connection_attempts=3,
        sleep=True,
    )


def login_interactive(username: Optional[str] = None) -> Path:
    """Prompt for password/2FA once and persist the session cookie."""
    username = (username or settings.ig_username).strip()
    if not username:
        raise SystemExit("Set IG_USERNAME in .env (or pass --username).")

    loader = _new_loader(quiet=False)
    target = session_file_for(username)
    target.parent.mkdir(parents=True, exist_ok=True)

    password = settings.ig_password
    try:
        if password:
            log.info("logging in as %s using IG_PASSWORD", username)
            loader.login(username, password)
        else:
            # Prompts for the password (and 2FA code) on the terminal.
            loader.interactive_login(username)
    except TwoFactorAuthRequiredException:
        code = input("Instagram 2FA code: ").strip()
        loader.two_factor_login(code)
    except BadCredentialsException as exc:
        raise SystemExit(f"Instagram rejected those credentials: {exc}") from exc

    loader.save_session_to_file(str(target))
    target.chmod(0o600)
    log.info("session saved to %s", target)
    return target


# ---------------------------------------------------------------------------
# source
# ---------------------------------------------------------------------------
class InstagramSource:
    name = "instagram"

    def __init__(self, username: Optional[str] = None, kind: str = "reels"):
        self.username = (username or settings.ig_username).strip()
        self.kind = kind  # reels | videos | all
        self.loader = _new_loader()
        self._authenticated = False
        self._profile: Optional[Profile] = None

    # -- auth --------------------------------------------------------------
    def authenticate(self) -> None:
        if self._authenticated:
            return
        if not self.username:
            raise SystemExit("Set IG_USERNAME in .env (or pass --username).")

        session_path = session_file_for(self.username)
        if session_path.exists():
            try:
                self.loader.load_session_from_file(self.username, str(session_path))
                log.info("loaded Instagram session for %s", self.username)
                self._authenticated = True
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("stored session unusable (%s); trying a fresh login", exc)

        if settings.ig_password:
            try:
                self.loader.login(self.username, settings.ig_password)
                session_path.parent.mkdir(parents=True, exist_ok=True)
                self.loader.save_session_to_file(str(session_path))
                session_path.chmod(0o600)
                self._authenticated = True
                log.info("logged in and saved a new session for %s", self.username)
                return
            except TwoFactorAuthRequiredException as exc:
                raise SystemExit(
                    "This account needs 2FA. Run:  python -m creator_intel login"
                ) from exc
            except Exception as exc:  # noqa: BLE001
                raise SystemExit(f"Instagram login failed: {exc}") from exc

        raise SystemExit(
            "No Instagram session found.\n"
            "Run this once, then re-run ingest:\n\n"
            "    python -m creator_intel login\n"
        )

    @property
    def profile(self) -> Profile:
        if self._profile is None:
            self.authenticate()
            self._profile = Profile.from_username(self.loader.context, self.username)
        return self._profile

    # -- listing -----------------------------------------------------------
    def _is_reel(self, post: Post) -> bool:
        if not _safe(lambda: post.is_video, False):
            return False
        if self.kind == "all":
            return True
        product_type = _safe(lambda: post._node.get("product_type"), None)
        if self.kind == "videos":
            return True
        if product_type is None:
            # Instagram omitted the field; a video post with no product_type is
            # almost always a Reel on a modern account, so keep it and log.
            return True
        return str(product_type).lower() in REEL_PRODUCT_TYPES

    def iter_items(self, limit: Optional[int] = None) -> Iterator[Post]:
        self.authenticate()
        emitted = 0
        walked = 0
        max_walk = settings.ig_max_posts or 0
        for post in self.profile.get_posts():
            walked += 1
            if max_walk and walked > max_walk:
                log.info("hit IG_MAX_POSTS=%s; stopping the walk", max_walk)
                break
            if not self._is_reel(post):
                continue
            yield post
            emitted += 1
            if limit and emitted >= limit:
                break
            time.sleep(settings.ig_request_delay)

    # -- mapping -----------------------------------------------------------
    def to_record(self, post: Post) -> Dict[str, Any]:
        shortcode = post.shortcode
        caption = _safe(lambda: post.caption, "") or ""
        date_utc = _safe(lambda: post.date_utc)
        node = _safe(lambda: post._asdict(), {}) or {}

        views = _safe(lambda: post.video_view_count)
        plays = node.get("video_play_count") or node.get("play_count")
        likes = _safe(lambda: post.likes)
        comments = _safe(lambda: post.comments)

        engagement_total = None
        if likes is not None or comments is not None:
            engagement_total = (likes or 0) + (comments or 0)
        engagement_rate = None
        denom = views or plays
        if engagement_total is not None and denom:
            engagement_rate = round(engagement_total / denom, 6)

        raw_path = settings.raw_dir / f"{shortcode}.json"
        try:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(
                json.dumps(node, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
        except OSError as exc:
            log.warning("could not write raw metadata for %s: %s", shortcode, exc)
            raw_path = None  # type: ignore[assignment]

        location = _safe(lambda: post.location)
        return {
            "shortcode": shortcode,
            "instagram_id": str(_safe(lambda: post.mediaid, "") or ""),
            "owner_username": _safe(lambda: post.owner_username, self.username),
            "owner_id": str(_safe(lambda: post.owner_id, "") or ""),
            "url": f"https://www.instagram.com/reel/{shortcode}/",
            "typename": _safe(lambda: post.typename),
            "is_video": 1 if _safe(lambda: post.is_video, False) else 0,
            "taken_at_utc": date_utc.isoformat() if date_utc else None,
            "taken_at_ts": int(date_utc.timestamp()) if date_utc else None,
            "caption": caption,
            "caption_hashtags": to_json(
                _safe(lambda: list(post.caption_hashtags), None) or extract_hashtags(caption)
            ),
            "caption_mentions": to_json(
                _safe(lambda: list(post.caption_mentions), None) or extract_mentions(caption)
            ),
            "accessibility_caption": _safe(lambda: post.accessibility_caption),
            "location_name": getattr(location, "name", None) if location else None,
            "view_count": views,
            "play_count": plays,
            "like_count": likes,
            "comment_count": comments,
            "engagement_total": engagement_total,
            "engagement_rate": engagement_rate,
            "duration_sec": _safe(lambda: post.video_duration),
            "metrics_json": to_json(
                {
                    "video_view_count": views,
                    "video_play_count": plays,
                    "likes": likes,
                    "comments": comments,
                    "product_type": node.get("product_type"),
                    "is_pinned": node.get("pinned_for_users") is not None,
                }
            ),
            "raw_metadata_json": to_json(node),
            "raw_metadata_path": str(raw_path) if raw_path else None,
            "metadata_updated_at": utcnow_iso(),
        }

    # -- media -------------------------------------------------------------
    def download(self, post: Post, record: Dict[str, Any]) -> Dict[str, Any]:
        """Download video + thumbnail into data/media/. Idempotent."""
        shortcode = post.shortcode
        media_dir = settings.media_dir
        media_dir.mkdir(parents=True, exist_ok=True)

        existing = self._find_local_video(shortcode)
        if existing:
            return {
                "local_video_path": str(existing),
                "local_thumbnail_path": self._find_local_thumb(shortcode),
                "video_bytes": existing.stat().st_size,
            }

        self.authenticate()
        self.loader.download_post(post, target=Path(""))

        video = self._find_local_video(shortcode)
        if not video:
            raise RuntimeError(f"instaloader reported success but no .mp4 landed for {shortcode}")

        self._cleanup_side_files(shortcode)
        return {
            "local_video_path": str(video),
            "local_thumbnail_path": self._find_local_thumb(shortcode),
            "video_bytes": video.stat().st_size,
            "media_downloaded_at": utcnow_iso(),
        }

    # -- local archive helpers --------------------------------------------
    @staticmethod
    def _find_local_video(shortcode: str) -> Optional[Path]:
        for candidate in sorted(settings.media_dir.glob(f"{shortcode}*.mp4")):
            if candidate.stat().st_size > 10_000:
                return candidate
        return None

    @staticmethod
    def _find_local_thumb(shortcode: str) -> Optional[str]:
        for candidate in sorted(settings.media_dir.glob(f"{shortcode}*.jpg")):
            return str(candidate)
        return None

    @staticmethod
    def _cleanup_side_files(shortcode: str) -> None:
        """instaloader drops .txt/.json.xz next to the media; we keep our own."""
        for pattern in (f"{shortcode}*.txt", f"{shortcode}*.json.xz"):
            for path in settings.media_dir.glob(pattern):
                try:
                    path.unlink()
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def ingest(
    conn,
    *,
    username: Optional[str] = None,
    limit: Optional[int] = None,
    kind: str = "reels",
    metadata_only: bool = False,
    refresh_metrics: bool = True,
    stop_after_known: int = 0,
) -> IngestResult:
    """Walk the profile newest-first, archiving anything not already stored.

    stop_after_known > 0 short-circuits the walk once N consecutive already-known
    reels are seen — the fast path for "just get me what's new".
    """
    from ..db import database as db

    source = InstagramSource(username=username, kind=kind)
    source.authenticate()
    result = IngestResult()
    consecutive_known = 0

    for post in source.iter_items(limit=limit):
        shortcode = post.shortcode
        result.total_seen += 1
        try:
            existing = db.get_reel(conn, shortcode)
            has_media = bool(existing and existing["local_video_path"] and Path(existing["local_video_path"]).exists())

            if existing and has_media and not refresh_metrics:
                result.skipped_existing += 1
                consecutive_known += 1
                if stop_after_known and consecutive_known >= stop_after_known:
                    log.info("saw %s known reels in a row; stopping early", consecutive_known)
                    break
                continue

            record = source.to_record(post)

            if not metadata_only:
                if has_media:
                    record["local_video_path"] = existing["local_video_path"]
                    result.skipped_existing += 1
                else:
                    log.info("downloading %s (%s)", shortcode, record.get("taken_at_utc"))
                    record.update(source.download(post, record))
                    result.downloaded += 1

            is_new = db.upsert_reel(conn, record)
            db.record_metrics_snapshot(
                conn,
                shortcode,
                {
                    "view_count": record.get("view_count"),
                    "play_count": record.get("play_count"),
                    "like_count": record.get("like_count"),
                    "comment_count": record.get("comment_count"),
                },
            )
            db.set_job(conn, shortcode, "ingest", "done")
            conn.commit()

            if is_new:
                result.new_reels += 1
                consecutive_known = 0
            else:
                result.updated += 1
                consecutive_known += 1
                if stop_after_known and consecutive_known >= stop_after_known:
                    log.info("saw %s known reels in a row; stopping early", consecutive_known)
                    break

        except (ConnectionException, LoginRequiredException) as exc:
            log.error("Instagram connection problem on %s: %s", shortcode, exc)
            db.set_job(conn, shortcode, "ingest", "error", error=str(exc))
            conn.commit()
            result.errors += 1
            result.error_detail.append((shortcode, str(exc)))
            log.error("backing off for 60s — Instagram is rate-limiting")
            time.sleep(60)
        except Exception as exc:  # noqa: BLE001 — one bad post must not kill the run
            log.exception("failed to ingest %s", shortcode)
            db.set_job(conn, shortcode, "ingest", "error", error=str(exc))
            conn.commit()
            result.errors += 1
            result.error_detail.append((shortcode, str(exc)))

    log.info(
        "ingest complete: seen=%s new=%s updated=%s downloaded=%s skipped=%s errors=%s",
        result.total_seen, result.new_reels, result.updated,
        result.downloaded, result.skipped_existing, result.errors,
    )
    return result


def probe_and_store_media_facts(conn, limit: Optional[int] = None, force: bool = False) -> int:
    """Fill duration/width/height/fps from the actual downloaded files."""
    from ..db import database as db
    from ..media.ffmpeg import probe

    sql = (
        "SELECT * FROM reels WHERE local_video_path IS NOT NULL AND local_video_path <> ''"
    )
    if not force:
        sql += " AND (duration_sec IS NULL OR width IS NULL)"
    sql += " ORDER BY taken_at_ts DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"

    updated = 0
    for row in list(conn.execute(sql)):
        path = Path(row["local_video_path"])
        if not path.exists():
            log.warning("archived file is missing: %s", path)
            db.set_job(conn, row["shortcode"], "media", "error", error="file missing")
            continue
        try:
            facts = probe(path)
            facts["shortcode"] = row["shortcode"]
            facts["video_bytes"] = path.stat().st_size
            db.upsert_reel(conn, facts)
            db.set_job(conn, row["shortcode"], "media", "done", detail=facts)
            updated += 1
        except Exception as exc:  # noqa: BLE001
            log.error("probe failed for %s: %s", row["shortcode"], exc)
            db.set_job(conn, row["shortcode"], "media", "error", error=str(exc))
        conn.commit()
    log.info("probed %s media files", updated)
    return updated
