"""AI analysis stage: raw evidence -> validated structured interpretation.

The prompt is deliberately built from RAW observations only (verbatim
transcript, exact hook windows, exact on-screen text, deterministic editing
measurements). The model is told, repeatedly, to quote rather than paraphrase,
and every quote is checked against the raw material before the row is saved.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from ..config import settings
from ..logging_setup import get_logger
from ..utils import from_json, json_from_llm, truncate
from .providers import get_provider
from .schemas import (
    PROMPT_VERSION,
    ReelClassification,
    evidence_is_grounded,
    to_db_row,
)

log = get_logger(__name__)

SYSTEM_PROMPT = """You are a content forensics analyst for a single short-form video creator.

Your job is to label ONE Instagram Reel so that thousands of them can later be
compared against performance data. Three hard rules:

1. QUOTE, DO NOT PARAPHRASE. Every `quote` field must be an exact substring of
   the raw material you were given. If you cannot find an exact quote, use null.
2. LABEL WHAT IS THERE, not what would be good practice. If the video has no
   CTA, cta is null. If the hook is weak, say so.
3. BE CONSISTENT. These labels are grouped across a whole library, so prefer a
   vocabulary term over an inventive new phrase unless nothing fits.

The creator's speech is transcribed VERBATIM, including filler words and
profanity. Do not sanitise it back to the user, and do not treat disfluency as
noise — it is part of the voice being studied.

Output raw JSON only. No markdown fences, no commentary."""


def _fmt_overlays(overlays: List[Dict[str, Any]]) -> str:
    if not overlays:
        return "(no on-screen text detected)"
    lines = []
    for o in overlays[:40]:
        span = f"{o.get('t_start', 0):.1f}s"
        if o.get("t_end") is not None and o.get("t_end") != o.get("t_start"):
            span += f"-{o.get('t_end'):.1f}s"
        role = o.get("role") or "overlay"
        lines.append(f'  [{span}] ({role}, {o.get("position") or "?"}) "{o.get("text")}"')
    return "\n".join(lines)


def _taxonomy_constraint() -> str:
    """If a curated taxonomy exists, future classifications must use it —
    open-vocabulary topics fragmented into 186 near-duplicates before this."""
    import json as _json

    tax_path = settings.data_dir / "taxonomy.json"
    if not tax_path.exists():
        return ""
    try:
        topics = _json.load(open(tax_path))["topics"]
    except Exception:
        return ""
    lines = "\n".join(f'- "{t["name"]}": {t.get("definition","")}' for t in topics)
    return (
        "\n\nMANDATORY TOPIC TAXONOMY — `main_topic` MUST be exactly one of these "
        "names (pick the best fit; use \"Other\" only if truly nothing fits):\n" + lines
    )


def build_prompt(evidence: Dict[str, Any]) -> str:
    from .schemas import schema_instructions

    hook = evidence.get("hook") or {}
    editing = evidence.get("editing") or {}
    visual = evidence.get("visual") or {}
    overlays = evidence.get("overlays") or []

    def g(d, k, default="(none)"):
        v = d.get(k)
        return default if v in (None, "") else v

    return f"""=== RAW OBSERVATIONS FOR ONE REEL ===

--- POST METADATA ---
posted:      {evidence.get('taken_at_utc') or 'unknown'}
duration:    {evidence.get('duration_sec') or '?'} s
views:       {evidence.get('view_count') if evidence.get('view_count') is not None else 'unknown'}
likes:       {evidence.get('like_count') if evidence.get('like_count') is not None else 'unknown'}
comments:    {evidence.get('comment_count') if evidence.get('comment_count') is not None else 'unknown'}

--- CAPTION (verbatim) ---
{truncate(evidence.get('caption') or '(empty caption)', 1500)}

--- OPENING SECONDS, SPOKEN (verbatim, exact wall-clock windows) ---
time until first spoken word: {hook.get('time_to_first_word') if hook.get('time_to_first_word') is not None else '?'} s
seconds 0.0-2.0: "{g(hook, 'spoken_0_2', '')}"
seconds 0.0-5.0: "{g(hook, 'spoken_0_5', '')}"
first spoken sentence: "{g(hook, 'spoken_first_sentence', '')}"

--- OPENING SECONDS, ON-SCREEN TEXT (verbatim) ---
seconds 0.0-5.0 on screen: "{g(hook, 'onscreen_0_5', '')}"
first overlay seen: "{g(hook, 'onscreen_hook', '')}"
distinct overlays in the hook window: {hook.get('onscreen_hook_count') or 0}

--- ALL ON-SCREEN TEXT DETECTED ---
{_fmt_overlays(overlays)}

--- FULL TRANSCRIPT (VERBATIM — filler words and profanity intact) ---
{truncate(evidence.get('transcript') or '(no speech detected)', 14000)}

--- EDITING MEASUREMENTS (deterministic) ---
cuts: {g(editing, 'cut_count', '?')} total, {g(editing, 'cuts_first_5s', '?')} in the first 5s
average shot length: {g(editing, 'avg_shot_sec', '?')} s
speaking pace: {g(editing, 'words_per_minute', '?')} wpm ({g(editing, 'speaking_pace', '?')})
burned-in subtitles: {'yes' if editing.get('has_subtitles') else 'no'}
text overlays: {'yes' if editing.get('has_text_overlays') else 'no'}
background music: {'yes' if editing.get('music_detected') else 'no'}

--- VISUAL SCENE (from sampled frames) ---
{g(visual, 'posture', '?')} / {g(visual, 'setting', '?')} / {g(visual, 'shot_type', '?')} shot / \
{g(visual, 'outfit_formality', '?')} {g(visual, 'outfit_category', '')} \
({g(visual, 'clothing_color', '?')}) / background: {g(visual, 'background', '?')} / \
another person present: {'yes' if visual.get('other_person') else 'no'}

=== YOUR TASK ===
{schema_instructions()}

Reminders:
- `hook_evidence[].quote` must appear EXACTLY in the spoken windows, the
  on-screen text, or the caption above.
- `cta` must be the creator's actual words, not a description of them.
- `notable_phrases[].phrase` must be verbatim — these are used later to study
  how this specific person talks.
- `series` is only non-null if this looks like part of a recurring, repeatable
  format (e.g. "texting teardown", "rating your bios"). Otherwise null.""" + _taxonomy_constraint()


def gather_evidence(conn, shortcode: str) -> Dict[str, Any]:
    """Everything raw we know about one reel, ready for prompting."""
    reel = conn.execute("SELECT * FROM reels WHERE shortcode = ?", (shortcode,)).fetchone()
    if reel is None:
        raise KeyError(f"unknown shortcode {shortcode}")
    transcript = conn.execute(
        "SELECT * FROM transcripts WHERE shortcode = ?", (shortcode,)
    ).fetchone()
    hook = conn.execute("SELECT * FROM raw_hook WHERE shortcode = ?", (shortcode,)).fetchone()
    editing = conn.execute("SELECT * FROM raw_editing WHERE shortcode = ?", (shortcode,)).fetchone()
    visual = conn.execute("SELECT * FROM raw_visual WHERE shortcode = ?", (shortcode,)).fetchone()
    overlays = conn.execute(
        "SELECT * FROM raw_overlay_text WHERE shortcode = ? ORDER BY t_start", (shortcode,)
    ).fetchall()

    return {
        "shortcode": shortcode,
        "taken_at_utc": reel["taken_at_utc"],
        "duration_sec": reel["duration_sec"],
        "caption": reel["caption"],
        "view_count": reel["view_count"],
        "like_count": reel["like_count"],
        "comment_count": reel["comment_count"],
        "transcript": transcript["text"] if transcript else None,
        "hook": dict(hook) if hook else {},
        "editing": dict(editing) if editing else {},
        "visual": dict(visual) if visual else {},
        "overlays": [dict(o) for o in overlays],
    }


def classify_one(
    conn,
    shortcode: str,
    *,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
) -> ReelClassification:
    evidence = gather_evidence(conn, shortcode)
    if not evidence.get("transcript"):
        raise ValueError(f"{shortcode} has no transcript yet")

    provider = get_provider(provider_name)
    model = model or settings.llm_model
    prompt = build_prompt(evidence)

    last_error: Optional[str] = None
    for attempt in range(1, 3):
        user_prompt = prompt
        if last_error:
            user_prompt += (
                f"\n\n=== YOUR PREVIOUS ANSWER WAS REJECTED ===\n{last_error}\n"
                "Return corrected JSON. Same keys, valid types."
            )
        resp = provider.complete(
            user_prompt, system=SYSTEM_PROMPT, model=model, max_tokens=4000, temperature=0.0
        )
        try:
            payload = json_from_llm(resp.text)
            classification = ReelClassification.model_validate(payload)
        except (ValidationError, ValueError) as exc:
            last_error = truncate(str(exc), 1500)
            log.warning("[%s] classification attempt %s rejected: %s", shortcode, attempt, last_error[:200])
            continue

        # Ground every quote against the raw material; drop the hallucinated ones.
        haystacks = [
            evidence.get("transcript") or "",
            evidence.get("caption") or "",
            " ".join(o.get("text") or "" for o in evidence.get("overlays") or []),
            (evidence.get("hook") or {}).get("onscreen_all") or "",
        ]
        kept, dropped = [], 0
        for ev in classification.hook_evidence:
            if evidence_is_grounded(ev.quote, haystacks):
                kept.append(ev)
            else:
                dropped += 1
        if dropped:
            log.info("[%s] dropped %s ungrounded hook quote(s)", shortcode, dropped)
        classification.hook_evidence = kept

        for phrase in list(classification.notable_phrases):
            if not evidence_is_grounded(phrase.phrase, haystacks):
                classification.notable_phrases.remove(phrase)

        classification.__pydantic_extra__ = classification.__pydantic_extra__ or {}
        classification.__pydantic_extra__["_provider"] = resp.provider
        classification.__pydantic_extra__["_model"] = resp.model
        classification.__pydantic_extra__["_prompt_version"] = PROMPT_VERSION
        classification.__pydantic_extra__["_ungrounded_quotes_dropped"] = dropped
        return classification

    raise ValueError(f"could not get valid JSON for {shortcode}: {last_error}")


def analyze_reels(
    conn,
    *,
    limit: Optional[int] = None,
    force: bool = False,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    shortcode: Optional[str] = None,
) -> Dict[str, int]:
    from ..db import database as db

    if shortcode:
        row = db.get_reel(conn, shortcode)
        rows = [row] if row else []
    else:
        rows = db.reels_needing(conn, "analyze", limit=limit, force=force)

    if not rows:
        log.info("nothing to analyse")
        return {"done": 0, "skipped": 0, "errors": 0}

    log.info("analysing %s reel(s) with %s/%s", len(rows), provider_name or settings.llm_provider,
             model or settings.llm_model)
    stats = {"done": 0, "skipped": 0, "errors": 0}

    for i, row in enumerate(rows, 1):
        sc = row["shortcode"]
        try:
            db.set_job(conn, sc, "analyze", "running")
            conn.commit()
            classification = classify_one(conn, sc, provider_name=provider_name, model=model)

            hook_row = conn.execute(
                "SELECT spoken_first_sentence FROM raw_hook WHERE shortcode = ?", (sc,)
            ).fetchone()
            db_row = to_db_row(
                sc,
                classification,
                provider=provider_name or settings.llm_provider,
                model=model or settings.llm_model,
                opening_sentence=hook_row["spoken_first_sentence"] if hook_row else None,
                duration_sec=row["duration_sec"],
            )
            written = db.save_analysis(conn, db_row, force=force)
            db.set_job(conn, sc, "analyze", "done")
            conn.commit()

            if written:
                stats["done"] += 1
                log.info(
                    "[%s/%s] %s -> %s / %s / hook=%s",
                    i, len(rows), sc, classification.main_topic,
                    classification.content_mode, classification.resolved_primary_hook(),
                )
            else:
                stats["skipped"] += 1
        except Exception as exc:  # noqa: BLE001
            log.exception("[%s/%s] analysis failed for %s", i, len(rows), sc)
            db.set_job(conn, sc, "analyze", "error", error=str(exc))
            conn.commit()
            stats["errors"] += 1

    log.info("analysis: %s", stats)
    return stats
