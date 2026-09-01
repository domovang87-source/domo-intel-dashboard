"""Frame-level vision: exact on-screen text + consistent scene labels.

One vision call handles both jobs per batch of frames, because they need the
same pixels and the same frame ordering. What comes back is stored twice:

  * raw_overlay_text  — the VERBATIM strings that appeared, with timings
  * raw_visual_frames — the per-frame scene labels, kept individually
  * raw_visual        — a majority-vote rollup for easy grouping

Sampled frames stay on disk (data/frames/<shortcode>/), so this whole stage can
be re-run with a better model later without touching Instagram again.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import settings
from ..logging_setup import get_logger
from ..utils import from_json, json_from_llm, to_json

log = get_logger(__name__)

BATCH_SIZE = 4           # frames per vision call (kept modest: image tokens are the TPM bottleneck)
MERGE_THRESHOLD = 0.86   # difflib ratio above which two OCR strings are "the same"

VISION_SYSTEM = """You are a precise video-frame analyst. You do two things per frame:

(A) TRANSCRIBE ON-SCREEN TEXT EXACTLY.
    Copy the characters as they appear: same words, same capitalisation, same
    punctuation, same emoji. "WAIT FOR THE LAST ONE" is not "wait for the last one".
    Do not translate, correct spelling, or summarise. If a word is partly cut
    off or unreadable, transcribe what is legible and nothing more.
    Ignore the Instagram interface itself (like/comment/share icons, usernames
    in the app chrome, progress bars) unless it is text the creator added.

(B) LABEL THE SCENE with broad, repeatable categories from the given options.
    Consistency across thousands of videos matters far more than nuance. Never
    invent a finer-grained fashion or identity description than the options allow.
    Do not guess about people's identity, ethnicity, or age beyond the options.

Return raw JSON only. No markdown fences, no commentary."""

VISION_PROMPT_TEMPLATE = """You are given {n} frames from one vertical short-form video, in order.
Their timestamps in seconds are: {stamps}

For EACH frame, return one object. Respond with exactly this JSON shape:

{{
  "frames": [
    {{
      "index": 0,
      "t_sec": 0.0,
      "texts": [
        {{
          "text": "<VERBATIM on-screen text, one entry per distinct text block>",
          "position": "top|upper-third|middle|lower-third|bottom",
          "role": "overlay|caption_subtitle|ui|watermark|unknown",
          "is_subtitle": true|false,
          "confidence": 0.0-1.0
        }}
      ],
      "scene": {{
        "posture": "sitting|standing|walking|lying|unknown",
        "setting": "indoors|outdoors|vehicle|unknown",
        "shot_type": "close-up|medium|full-body|unknown",
        "outfit_category": "t-shirt|shirt|hoodie|sweater|jacket|suit|tank|dress|athletic|shirtless|unknown",
        "clothing_color": "<single dominant colour word>",
        "outfit_formality": "casual|professional|going-out|athletic|unknown",
        "hair": "up|down|covered|bald|short|unknown",
        "glasses": true|false,
        "background": "bedroom|living-room|office|kitchen|car|gym|street|outdoors-nature|studio|plain-wall|unknown",
        "camera_angle": "eye-level|low|high|dutch|unknown",
        "camera_motion": "static|handheld|walking|unknown",
        "lighting": "natural|ring-light|dim|harsh|mixed|unknown",
        "other_person_visible": true|false,
        "person_count": 0
      }}
    }}
  ]
}}

Rules:
- "texts" is an empty array if the frame has no added text.
- `is_subtitle` is true only for word-by-word or line-by-line captions that
  track the speech (usually centre or lower-third, changing every frame).
  A static headline that stays put is an overlay, not a subtitle.
- Return exactly {n} frame objects, indexes 0..{last}."""


# ---------------------------------------------------------------------------
# calling the model
# ---------------------------------------------------------------------------
def analyze_frame_batch(
    frames: Sequence[Dict[str, Any]], *, provider_name: Optional[str] = None, model: Optional[str] = None
) -> List[Dict[str, Any]]:
    from ..analysis.providers import get_provider

    provider = get_provider(provider_name)
    stamps = ", ".join(f"{f['t_sec']:.2f}" for f in frames)
    prompt = VISION_PROMPT_TEMPLATE.format(n=len(frames), stamps=stamps, last=len(frames) - 1)

    resp = provider.complete_vision(
        prompt,
        [Path(f["path"]) for f in frames],
        system=VISION_SYSTEM,
        model=model or settings.vision_model,
        max_tokens=4000,
        temperature=0.0,
    )
    payload = json_from_llm(resp.text)
    out = payload.get("frames") if isinstance(payload, dict) else payload
    if not isinstance(out, list):
        raise ValueError(f"vision model returned an unexpected shape: {type(out)}")

    # Trust our own timestamps over the model's echo of them.
    for i, item in enumerate(out[: len(frames)]):
        item["t_sec"] = frames[i]["t_sec"]
        item["frame_path"] = frames[i]["path"]
    return out[: len(frames)]


# ---------------------------------------------------------------------------
# aggregation: frames -> overlay text spans
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def aggregate_overlays(frame_results: Sequence[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    """Collapse per-frame text detections into one row per distinct string.

    OCR jitters ("WAIT FOR THE LAST ONE" vs "WAIT FOR THE LAST ONE!"), so
    near-identical strings are merged and the longest observed variant is kept
    as the canonical verbatim text.
    """
    buckets: List[Dict[str, Any]] = []

    for fr in frame_results:
        t = float(fr.get("t_sec") or 0.0)
        for item in fr.get("texts") or []:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            key = _norm(text)
            match = None
            for b in buckets:
                if b["key"] == key or difflib.SequenceMatcher(None, b["key"], key).ratio() >= MERGE_THRESHOLD:
                    match = b
                    break
            if match is None:
                buckets.append(
                    {
                        "key": key,
                        "variants": [text],
                        "t_start": t,
                        "t_end": t,
                        "frames": [t],
                        "positions": [item.get("position")],
                        "roles": [item.get("role")],
                        "subtitle_votes": [bool(item.get("is_subtitle"))],
                        "confidences": [item.get("confidence")],
                    }
                )
            else:
                match["variants"].append(text)
                match["t_start"] = min(match["t_start"], t)
                match["t_end"] = max(match["t_end"], t)
                match["frames"].append(t)
                match["positions"].append(item.get("position"))
                match["roles"].append(item.get("role"))
                match["subtitle_votes"].append(bool(item.get("is_subtitle")))
                match["confidences"].append(item.get("confidence"))

    rows = []
    for b in buckets:
        variants = sorted(b["variants"], key=len, reverse=True)
        confs = [c for c in b["confidences"] if isinstance(c, (int, float))]
        positions = [p for p in b["positions"] if p]
        roles = [r for r in b["roles"] if r]
        is_subtitle = sum(b["subtitle_votes"]) > len(b["subtitle_votes"]) / 2
        rows.append(
            {
                "text": variants[0],                       # verbatim, longest observed variant
                "t_start": round(b["t_start"], 2),
                "t_end": round(b["t_end"], 2),
                "duration_sec": round(b["t_end"] - b["t_start"], 2),
                "position": Counter(positions).most_common(1)[0][0] if positions else None,
                "role": Counter(roles).most_common(1)[0][0] if roles else "overlay",
                "is_subtitle": 1 if is_subtitle else 0,
                "frame_count": len(b["frames"]),
                "source": f"vision:{model}",
                "confidence": round(sum(confs) / len(confs), 3) if confs else None,
                "frames_json": to_json(sorted(round(t, 2) for t in b["frames"])),
            }
        )
    rows.sort(key=lambda r: (r["t_start"], -r["frame_count"]))
    return rows


# ---------------------------------------------------------------------------
# aggregation: frames -> one visual summary
# ---------------------------------------------------------------------------
_ROLLUP_FIELDS = [
    "posture", "setting", "shot_type", "outfit_category", "clothing_color",
    "outfit_formality", "hair", "background", "camera_angle", "camera_motion", "lighting",
]


def rollup_visual(frame_results: Sequence[Dict[str, Any]], model: str) -> Dict[str, Any]:
    """Majority vote per field, with the agreement level preserved."""
    votes: Dict[str, List[str]] = defaultdict(list)
    glasses_votes: List[bool] = []
    other_person_votes: List[bool] = []
    person_counts: List[int] = []

    for fr in frame_results:
        scene = fr.get("scene") or {}
        for field in _ROLLUP_FIELDS:
            value = scene.get(field)
            if value and str(value).lower() not in ("unknown", "none", "n/a"):
                votes[field].append(str(value).lower())
        if isinstance(scene.get("glasses"), bool):
            glasses_votes.append(scene["glasses"])
        if isinstance(scene.get("other_person_visible"), bool):
            other_person_votes.append(scene["other_person_visible"])
        if isinstance(scene.get("person_count"), int):
            person_counts.append(scene["person_count"])

    summary: Dict[str, Any] = {}
    agreement: Dict[str, Any] = {}
    for field in _ROLLUP_FIELDS:
        values = votes.get(field) or []
        if not values:
            summary[field] = "unknown"
            agreement[field] = 0.0
            continue
        counter = Counter(values)
        top, count = counter.most_common(1)[0]
        # A field that flips around across the video is genuinely "mixed".
        summary[field] = top if count / len(values) >= 0.5 else "mixed"
        agreement[field] = round(count / len(values), 3)

    summary["glasses"] = int(sum(glasses_votes) > len(glasses_votes) / 2) if glasses_votes else None
    summary["other_person"] = (
        int(sum(other_person_votes) > len(other_person_votes) / 3) if other_person_votes else None
    )
    summary["person_count"] = max(person_counts) if person_counts else None
    summary["frames_analyzed"] = len(frame_results)
    summary["model"] = model
    summary["labels_json"] = to_json(
        {"rollup": {k: summary.get(k) for k in _ROLLUP_FIELDS}, "agreement": agreement,
         "frames_analyzed": len(frame_results)}
    )
    return summary


# ---------------------------------------------------------------------------
# subtitle reclassification (deterministic, transcript-grounded)
# ---------------------------------------------------------------------------
_NORM_STRIP = re.compile(r"[^a-z0-9\s]")


def _norm_text(text: str) -> str:
    text = (text or "").lower().replace("'", "").replace("’", "")
    return " ".join(_NORM_STRIP.sub(" ", text).split())


def reclassify_subtitles(
    rows: List[Dict[str, Any]], transcript_text: Optional[str]
) -> List[Dict[str, Any]]:
    """Mark burned-in captions using the transcript as ground truth.

    Vision models are unreliable at telling word-tracking captions apart from
    creative overlays frame-by-frame. But captions BY DEFINITION repeat the
    spoken words, and we hold the verbatim transcript — so a short-lived
    on-screen string whose words appear in the transcript is a caption. A
    static headline (visible across many sampled frames) stays an overlay even
    if it quotes speech.
    """
    if not transcript_text or not rows:
        return rows
    t_norm = _norm_text(transcript_text)
    t_words = set(t_norm.split())

    for r in rows:
        n = _norm_text(r.get("text") or "")
        words = n.split()
        r["_wc"] = len(words)
        r["_short_lived"] = (r.get("frame_count") or 1) <= 3
        r["_match"] = bool(
            words
            and len(words) >= 2
            and (n in t_norm or all(w in t_words for w in words))
        )

    # A reel is caption-styled when several short-lived matches exist; only
    # then do single-word chunks ("HELL") also count as captions.
    caption_reel = sum(1 for r in rows if r["_match"] and r["_short_lived"]) >= 3

    for r in rows:
        is_sub = False
        if r["_short_lived"] and r["_match"]:
            is_sub = True
        elif r["_short_lived"] and caption_reel and r["_wc"] == 1:
            word = _norm_text(r.get("text") or "")
            is_sub = word in t_words
        if is_sub:
            r["is_subtitle"] = 1
            r["role"] = "caption_subtitle"
        for k in ("_wc", "_short_lived", "_match"):
            r.pop(k, None)
    return rows


# ---------------------------------------------------------------------------
# tesseract fallback (only if the binary happens to be installed)
# ---------------------------------------------------------------------------
def tesseract_available() -> bool:
    import shutil

    if not shutil.which("tesseract"):
        return False
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


def ocr_frames_tesseract(frames: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Offline OCR path. Less accurate on stylised captions than a vision model,
    but it costs nothing and needs no network."""
    import pytesseract
    from PIL import Image

    results = []
    for f in frames:
        try:
            with Image.open(f["path"]) as im:
                data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)
                height = im.height
        except Exception as exc:  # noqa: BLE001
            log.debug("tesseract failed on %s: %s", f["path"], exc)
            results.append({"t_sec": f["t_sec"], "frame_path": f["path"], "texts": [], "scene": {}})
            continue

        lines: Dict[tuple, Dict[str, Any]] = {}
        for i, word in enumerate(data["text"]):
            if not word.strip() or int(data["conf"][i] or -1) < 45:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            entry = lines.setdefault(key, {"words": [], "top": data["top"][i], "conf": []})
            entry["words"].append(word)
            entry["conf"].append(int(data["conf"][i]))

        texts = []
        for entry in lines.values():
            rel = entry["top"] / max(1, height)
            position = "top" if rel < 0.2 else "upper-third" if rel < 0.4 else \
                       "middle" if rel < 0.6 else "lower-third" if rel < 0.85 else "bottom"
            texts.append(
                {
                    "text": " ".join(entry["words"]),
                    "position": position,
                    "role": "caption_subtitle" if rel > 0.55 else "overlay",
                    "is_subtitle": rel > 0.55,
                    "confidence": round(sum(entry["conf"]) / len(entry["conf"]) / 100.0, 3),
                }
            )
        results.append({"t_sec": f["t_sec"], "frame_path": f["path"], "texts": texts, "scene": {}})
    return results
