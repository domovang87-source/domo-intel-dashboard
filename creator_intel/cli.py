"""Command-line interface.

    python -m creator_intel <command> [options]

Every command is safe to interrupt and re-run.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .config import settings
from .logging_setup import get_logger, setup_logging

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# small output helpers
# ---------------------------------------------------------------------------
def _print_json(data) -> None:
    print(json.dumps(data, indent=2, default=str))


def _hr(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_init(args) -> int:
    from .db.database import init_db

    path = init_db()
    settings.ensure_dirs()
    print(f"Database ready: {path}")
    print(f"Media archive:  {settings.media_dir}")
    print(f"Frames:         {settings.frames_dir}")
    return 0


def cmd_login(args) -> int:
    from .ingest.instagram import login_interactive

    path = login_interactive(args.username)
    print(f"Instagram session saved to {path}")
    print("You will not need to log in again unless Instagram invalidates it.")
    return 0


def cmd_doctor(args) -> int:
    """Check that everything this pipeline depends on is actually usable."""
    from .ingest.instagram import session_file_for

    ok = True
    _hr("environment")
    print(f"python              {sys.version.split()[0]}")
    print(f"data dir            {settings.data_dir}")
    print(f"database            {settings.db_path} {'(exists)' if settings.db_path.exists() else '(not created yet)'}")

    _hr("ffmpeg")
    try:
        from .media.ffmpeg import ffmpeg_bin

        print(f"ffmpeg              {ffmpeg_bin()}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"ffmpeg              MISSING ({exc})")

    _hr("instagram")
    if settings.ig_username:
        session = session_file_for(settings.ig_username)
        print(f"username            {settings.ig_username}")
        print(f"session file        {session} {'(found)' if session.exists() else '(MISSING — run: login)'}")
        if not session.exists():
            ok = False
    else:
        ok = False
        print("username            NOT SET — put IG_USERNAME in .env")

    _hr("llm")
    print(f"provider            {settings.llm_provider}")
    print(f"text model          {settings.llm_model}")
    print(f"vision model        {settings.vision_model}")
    print(f"embedding model     {settings.embedding_model}")
    key = settings.api_key_for(settings.llm_provider)
    print(f"api key             {'set (' + key[:7] + '…)' if key else 'MISSING'}")
    if not key:
        ok = False
    try:
        from .analysis.providers import available_providers

        print(f"available providers {', '.join(available_providers())}")
    except Exception as exc:  # noqa: BLE001
        print(f"providers           error: {exc}")

    _hr("transcription")
    print(f"backend             {settings.transcribe_backend}")
    print(f"whisper model       {settings.whisper_model} ({settings.whisper_compute_type}, {settings.whisper_device})")
    try:
        import faster_whisper  # noqa: F401

        print("faster-whisper      installed")
    except ImportError:
        ok = False
        print("faster-whisper      MISSING")

    _hr("ocr")
    from .forensics.vision import tesseract_available

    print(f"tesseract           {'available' if tesseract_available() else 'not installed (vision model will be used)'}")

    print("\n" + ("All good." if ok else "Some things need attention (see MISSING above)."))
    return 0 if ok else 1


def cmd_ingest(args) -> int:
    from .db.database import connect
    from .ingest.instagram import ingest, probe_and_store_media_facts

    with connect() as conn:
        result = ingest(
            conn,
            username=args.username,
            limit=args.limit,
            kind=args.kind,
            metadata_only=args.metadata_only,
            stop_after_known=args.stop_after_known,
        )
        if not args.metadata_only:
            probe_and_store_media_facts(conn, force=args.force)
        _print_json(result.as_dict())
    return 0


def cmd_import_insights(args) -> int:
    from .db.database import connect, init_db
    from .ingest.insights import import_insights_csv

    init_db()
    with connect() as conn:
        stats = import_insights_csv(conn, args.csv, dry_run=args.dry_run)
    _print_json(stats)
    if stats["unmatched"]:
        print(
            f"\n{stats['unmatched']} row(s) didn't match any archived reel "
            "(usually posts that aren't downloaded yet, or Stories/photos in the export)."
        )
    return 0


def cmd_probe(args) -> int:
    from .db.database import connect
    from .ingest.instagram import probe_and_store_media_facts

    with connect() as conn:
        count = probe_and_store_media_facts(conn, limit=args.limit, force=args.force)
    print(f"probed {count} file(s)")
    return 0


def cmd_transcribe(args) -> int:
    from .db.database import connect
    from .transcribe import transcribe_reels

    with connect() as conn:
        _print_json(
            transcribe_reels(
                conn, limit=args.limit, force=args.force,
                backend=args.backend, shortcode=args.shortcode,
            )
        )
    return 0


def cmd_forensics(args) -> int:
    from .db.database import connect
    from .forensics import run_forensics

    with connect() as conn:
        _print_json(run_forensics(conn, limit=args.limit, force=args.force, shortcode=args.shortcode))
    return 0


def cmd_vision(args) -> int:
    from .db.database import connect
    from .forensics import run_vision

    with connect() as conn:
        _print_json(
            run_vision(
                conn, limit=args.limit, force=args.force, shortcode=args.shortcode,
                use_tesseract=args.tesseract, model=args.model,
            )
        )
    return 0


def cmd_analyze(args) -> int:
    from .analysis.classify import analyze_reels
    from .db.database import connect

    with connect() as conn:
        _print_json(
            analyze_reels(
                conn, limit=args.limit, force=args.force, shortcode=args.shortcode,
                provider_name=args.provider, model=args.model,
            )
        )
    return 0


def cmd_embed(args) -> int:
    from .db.database import connect
    from .voice.embeddings import embed_reels

    with connect() as conn:
        _print_json(embed_reels(conn, limit=args.limit, force=args.force))
    return 0


def cmd_merge_topics(args) -> int:
    from .analysis.topics import merge_topics
    from .db.database import connect

    with connect() as conn:
        _print_json(merge_topics(conn, force=args.force))
    return 0


def cmd_run(args) -> int:
    from .db.database import connect, init_db
    from .pipeline import run_batched, run_pipeline

    init_db()
    with connect() as conn:
        if args.batch:
            stats = run_batched(conn, batch=args.batch, use_tesseract=args.tesseract)
            _print_json(stats)
            return 0
        stats = run_pipeline(
            conn,
            limit=args.limit,
            force=args.force,
            skip=args.skip,
            only=args.only,
            username=args.username,
            kind=args.kind,
            use_tesseract=args.tesseract,
            stop_after_known=args.stop_after_known,
        )
    _print_json(stats)
    return 0


def cmd_status(args) -> int:
    from .db.database import connect
    from .pipeline import status

    with connect(read_only=True) as conn:
        info = status(conn)

    _hr("archive")
    for key, value in info["counts"].items():
        print(f"  {key:<20} {value}")
    _hr("still to do")
    for key, value in info["pending"].items():
        print(f"  {key:<20} {value}")
    if info["recent_errors"]:
        _hr("recent errors")
        for err in info["recent_errors"][:10]:
            print(f"  {err['shortcode']:<14} {err['stage']:<10} {str(err['error'])[:80]}")
    _hr("paths")
    for key, value in info["paths"].items():
        print(f"  {key:<20} {value}")
    return 0


def cmd_search(args) -> int:
    from .db.database import connect

    with connect(read_only=True) as conn:
        if args.semantic:
            from .voice.search import semantic_search

            results = semantic_search(conn, args.query, k=args.limit)
            _hr(f"semantic matches for: {args.query}")
            for r in results:
                views = r.get("view_count")
                print(
                    f"  {r['similarity']:.3f}  {r['shortcode']:<14} "
                    f"{('views ' + str(int(views))) if views else 'views ?':<14} "
                    f"{(r.get('main_topic') or '-')[:24]:<26} "
                    f"\"{(r.get('spoken_hook') or '')[:70]}\""
                )
        else:
            from .db.database import search_fts

            results = search_fts(conn, args.query, limit=args.limit)
            _hr(f"transcript matches for: {args.query}")
            for r in results:
                print(f"  {r['shortcode']:<14} {r['snippet']}")
        if not results:
            print("  (no matches)")
    return 0


def cmd_similar(args) -> int:
    from .db.database import connect
    from .voice.search import similar_to_reel

    with connect(read_only=True) as conn:
        results = similar_to_reel(conn, args.shortcode, k=args.limit)
    _hr(f"most similar to {args.shortcode}")
    for r in results:
        print(f"  {r['similarity']:.3f}  {r['shortcode']:<14} \"{(r.get('spoken_hook') or '')[:80]}\"")
    return 0


def cmd_voice_pack(args) -> int:
    from .db.database import connect
    from .voice.search import voice_pack

    with connect(read_only=True) as conn:
        pack = voice_pack(conn, args.idea, k=args.limit)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(pack, fh, indent=2, default=str)
        print(f"wrote {args.out} ({pack['matches']} examples)")
    else:
        _print_json(pack)
    return 0


def cmd_show(args) -> int:
    """Everything the database knows about one reel — raw first, labels second."""
    from .db.database import connect
    from .utils import from_json

    with connect(read_only=True) as conn:
        row = conn.execute("SELECT * FROM v_reel_full WHERE shortcode = ?", (args.shortcode,)).fetchone()
        if row is None:
            print(f"no such reel: {args.shortcode}")
            return 1
        row = dict(row)
        overlays = [
            dict(r) for r in conn.execute(
                "SELECT * FROM raw_overlay_text WHERE shortcode = ? ORDER BY t_start",
                (args.shortcode,),
            )
        ]

    if args.json:
        row["overlays"] = overlays
        _print_json(row)
        return 0

    _hr(f"{row['shortcode']}  —  {row.get('taken_at_utc')}")
    print(f"  url        {row.get('url')}")
    print(f"  duration   {row.get('duration_sec')}s")
    print(f"  views      {row.get('view_count')}   likes {row.get('like_count')}   comments {row.get('comment_count')}")

    _hr("RAW — spoken opening")
    print(f"  time to first word   {row.get('time_to_first_word')}s")
    print(f"  0.0-2.0s             \"{row.get('spoken_0_2') or ''}\"")
    print(f"  0.0-5.0s             \"{row.get('spoken_0_5') or ''}\"")
    print(f"  first sentence       \"{row.get('spoken_first_sentence') or ''}\"")

    _hr("RAW — on screen")
    if overlays:
        for o in overlays:
            flag = "SUB" if o.get("is_subtitle") else "TXT"
            print(f"  [{o['t_start']:>5.1f}-{o['t_end']:>5.1f}s] {flag} {o.get('position') or '?':<12} \"{o['text']}\"")
    else:
        print("  (none detected — run: vision)")

    _hr("RAW — editing")
    print(f"  cuts {row.get('cut_count')} (first 5s: {row.get('cuts_first_5s')})   "
          f"avg shot {row.get('avg_shot_sec')}s   {row.get('words_per_minute')} wpm ({row.get('speaking_pace')})")
    print(f"  subtitles {bool(row.get('has_subtitles'))}   overlays {bool(row.get('has_text_overlays'))}   "
          f"music {bool(row.get('music_detected'))}   zooms {row.get('zoom_events')}")

    _hr("RAW — visual")
    print(f"  {row.get('posture')} / {row.get('setting')} / {row.get('shot_type')} / "
          f"{row.get('outfit_formality')} {row.get('outfit_category')} ({row.get('clothing_color')}) / "
          f"bg {row.get('background')} / other person: {bool(row.get('other_person'))}")

    _hr("INTERPRETATION — ai labels")
    print(f"  topic       {row.get('main_topic')} / {row.get('subtopic')}")
    print(f"  category    {row.get('dating_category')}    series: {row.get('series')}")
    print(f"  format      {row.get('content_format')} / {row.get('content_mode')} / tone {row.get('tone')}")
    print(f"  hook        {row.get('hook_primary_type')} ({row.get('hook_structure')})  "
          f"types={from_json(row.get('hook_types_json'), [])}")
    print(f"  cta         {row.get('cta') or '(none)'}")
    print(f"  summary     {row.get('summary')}")

    _hr("RAW — verbatim transcript")
    print(row.get("transcript") or "(none)")
    return 0


def cmd_dashboard(args) -> int:
    import subprocess
    from pathlib import Path

    app = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port)]
    print(f"launching: {' '.join(cmd)}")
    return subprocess.call(cmd)


def cmd_export(args) -> int:
    """Dump the joined view to CSV/JSON for use elsewhere."""
    from .analytics.queries import load_reels

    df = load_reels()
    if df.empty:
        print("nothing to export yet")
        return 1
    if args.format == "csv":
        df.to_csv(args.out, index=False)
    else:
        df.to_json(args.out, orient="records", indent=2, date_format="iso")
    print(f"wrote {len(df)} rows to {args.out}")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="creator_intel",
        description="Local Instagram creator intelligence: archive, transcribe, dissect, analyse.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python -m creator_intel init
  python -m creator_intel login
  python -m creator_intel run --limit 5
  python -m creator_intel search "situationship" --semantic
  python -m creator_intel show C1a2b3c4d5
  python -m creator_intel dashboard
""",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG | INFO | WARNING | ERROR")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p, *, shortcode=False):
        p.add_argument("--limit", type=int, default=None, help="max items to process")
        p.add_argument("--force", action="store_true", help="redo work that is already done")
        if shortcode:
            p.add_argument("--shortcode", default=None, help="process just this one reel")

    p = sub.add_parser("init", help="create the SQLite database and folders")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("login", help="log into Instagram once and save the session")
    p.add_argument("--username", default=None)
    p.set_defaults(func=cmd_login)

    p = sub.add_parser("doctor", help="check credentials, models and binaries")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("ingest", help="download new Reels + metadata")
    add_common(p)
    p.add_argument("--username", default=None)
    p.add_argument("--kind", choices=["reels", "videos", "all"], default="reels")
    p.add_argument("--metadata-only", action="store_true", help="refresh metrics without downloading media")
    p.add_argument("--stop-after-known", type=int, default=0,
                   help="stop once N consecutive already-archived reels are seen")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("probe", help="read duration/resolution from downloaded files")
    add_common(p)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser(
        "import-insights",
        help="merge views/shares/saves/reach from a Meta Business Suite content CSV export",
    )
    p.add_argument("csv", help="path to the exported .csv file")
    p.add_argument("--dry-run", action="store_true", help="parse and report, write nothing")
    p.set_defaults(func=cmd_import_insights)

    p = sub.add_parser("transcribe", help="verbatim transcription with timestamps")
    add_common(p, shortcode=True)
    p.add_argument("--backend", default=None, choices=["faster_whisper", "openai"])
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("forensics", help="frames, cuts, pacing and raw hook windows (local, free)")
    add_common(p, shortcode=True)
    p.set_defaults(func=cmd_forensics)

    p = sub.add_parser("vision", help="on-screen text OCR + visual scene labels (uses the API)")
    add_common(p, shortcode=True)
    p.add_argument("--tesseract", action="store_true", help="use local tesseract OCR instead")
    p.add_argument("--model", default=None)
    p.set_defaults(func=cmd_vision)

    p = sub.add_parser("analyze", help="LLM classification into structured fields")
    add_common(p, shortcode=True)
    p.add_argument("--provider", default=None, choices=["openai", "anthropic", "gemini"])
    p.add_argument("--model", default=None)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("embed", help="build/refresh embeddings for semantic search")
    add_common(p)
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("merge-topics", help="collapse near-duplicate topic labels into canonical topics")
    p.add_argument("--force", action="store_true", help="rebuild the whole alias map from scratch")
    p.set_defaults(func=cmd_merge_topics)

    p = sub.add_parser("run", help="the whole pipeline, end to end")
    add_common(p)
    p.add_argument("--username", default=None)
    p.add_argument("--kind", choices=["reels", "videos", "all"], default="reels")
    p.add_argument("--skip", nargs="*", default=None, help="stages to skip, e.g. --skip vision analyze")
    p.add_argument("--only", nargs="*", default=None, help="run only these stages")
    p.add_argument("--tesseract", action="store_true")
    p.add_argument("--stop-after-known", type=int, default=0)
    p.add_argument(
        "--batch", type=int, default=0, metavar="N",
        help="interleaved mode: push batches of N reels through ALL stages so "
             "fully-analysed reels appear continuously (skips ingest; safe to "
             "run alongside a downloading process)",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="what is archived and what still needs doing")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("search", help="search transcripts (keyword, or --semantic)")
    p.add_argument("query")
    p.add_argument("--semantic", action="store_true", help="vector similarity instead of keywords")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("similar", help="find reels most like a given reel")
    p.add_argument("shortcode")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_similar)

    p = sub.add_parser("voice-pack", help="retrieve past videos as writing references for an idea")
    p.add_argument("idea")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--out", default=None, help="write JSON to this path")
    p.set_defaults(func=cmd_voice_pack)

    p = sub.add_parser("show", help="everything known about one reel")
    p.add_argument("shortcode")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("export", help="dump the joined view to a file")
    p.add_argument("--format", choices=["csv", "json"], default="csv")
    p.add_argument("--out", default="creator_intel_export.csv")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(func=cmd_dashboard)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        log.warning("interrupted — progress is saved; re-run the same command to resume")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
