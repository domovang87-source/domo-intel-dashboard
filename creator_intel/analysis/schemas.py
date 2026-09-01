"""Validated shape of the AI INTERPRETATION layer.

Two rules encoded here:
  1. Every label that makes a claim about the video must carry a verbatim
     `quote` lifted from the raw transcript or the raw on-screen text. Labels
     without evidence are the thing that makes a content database useless.
  2. Vocabularies are suggested, not enforced. An unexpected value is normalised
     for grouping but the model's original string is preserved, because a new
     hook type you have not thought of yet is a finding, not an error.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROMPT_VERSION = "hookforensics-v1"

# --- suggested vocabularies -------------------------------------------------
HOOK_TYPES = [
    "curiosity", "secret", "warning", "controversial_claim", "negative_claim",
    "demographic_callout", "direct_address", "command", "promise", "list",
    "story_opening", "question", "confession", "authority_claim",
    "wait_until_the_end", "the_last_one_is_best", "manipulative_framing",
    "shocking_framing", "specific_person_or_profession", "archetype_callout",
    "statistic", "myth_bust", "contrarian", "relatable_scenario",
    "pattern_interrupt", "social_proof", "urgency", "fear", "flex", "other",
]

HOOK_STRUCTURES = [
    "statement", "question", "if_then", "you_should_never", "most_people_do_x",
    "here_is_why", "number_list", "callout_then_claim", "story_in_media_res",
    "problem_then_promise", "negation_then_correction", "quote_then_reaction",
    "command_then_reason", "other",
]

CONTENT_MODES = ["story", "advice", "rant", "sales", "reaction", "skit", "qna", "demo", "other"]

CONTENT_FORMATS = [
    "talking_head", "voiceover", "green_screen", "screen_recording", "skit",
    "interview", "street_interview", "duet_or_stitch", "montage", "text_on_screen", "other",
]

CTA_TYPES = [
    "none", "comment", "follow", "like", "share", "save", "dm", "link_in_bio",
    "buy", "book_call", "join_program", "watch_more", "subscribe", "other",
]

TONES = [
    "blunt", "playful", "serious", "sarcastic", "empathetic", "aggressive",
    "confident", "conspiratorial", "instructional", "hype", "deadpan",
    "vulnerable", "authoritative", "other",
]

LENGTH_BUCKETS = ["<15s", "15-30s", "30-60s", "60-90s", "90s+"]


def bucket_length(duration: Optional[float]) -> Optional[str]:
    if duration is None:
        return None
    d = float(duration)
    if d < 15:
        return "<15s"
    if d < 30:
        return "15-30s"
    if d < 60:
        return "30-60s"
    if d < 90:
        return "60-90s"
    return "90s+"


def _normalize(value: Optional[str], vocab: List[str]) -> Optional[str]:
    """Snap to the vocabulary when recognisable; otherwise return as-is."""
    if not value:
        return value
    slug = value.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    if slug in vocab:
        return slug
    for term in vocab:
        if term != "other" and (term in slug or slug in term):
            return term
    return slug


# --- sub-models -------------------------------------------------------------
class HookEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    hook_type: str = Field(description="one of HOOK_TYPES")
    quote: str = Field(description="verbatim span from the transcript or on-screen text")
    source: str = Field(default="spoken", description="spoken | onscreen | caption")
    reason: Optional[str] = None

    @field_validator("hook_type")
    @classmethod
    def _norm_type(cls, v: str) -> str:
        return _normalize(v, HOOK_TYPES) or "other"

    @field_validator("source")
    @classmethod
    def _norm_source(cls, v: str) -> str:
        v = (v or "spoken").lower()
        return v if v in {"spoken", "onscreen", "caption"} else "spoken"


class Claim(BaseModel):
    model_config = ConfigDict(extra="allow")

    claim: str
    quote: Optional[str] = Field(default=None, description="verbatim supporting span")


class Archetype(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str = Field(description="e.g. 'lawyer', 'older woman', 'younger man', 'ex-boyfriend'")
    category: Optional[str] = Field(default=None, description="profession | age_group | relationship_role | other")
    mention: Optional[str] = Field(default=None, description="verbatim phrase used in the video")


class NotablePhrase(BaseModel):
    model_config = ConfigDict(extra="allow")

    phrase: str = Field(description="VERBATIM phrase as spoken")
    why: Optional[str] = Field(default=None, description="what makes it characteristic")
    t_sec: Optional[float] = None


# --- main model -------------------------------------------------------------
class ReelClassification(BaseModel):
    """One LLM verdict about one Reel. Regenerable; never overwrites raw data."""

    model_config = ConfigDict(extra="allow")

    # topic
    main_topic: str
    subtopic: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    dating_category: Optional[str] = Field(
        default=None,
        description="dating-specific bucket: attraction, texting, situationships, "
        "breakups, dating apps, confidence, masculinity, female psychology, etc.",
    )
    target_audience: Optional[str] = None

    # format / mode
    content_format: Optional[str] = None
    content_mode: Optional[str] = Field(default=None, description="story|advice|rant|sales|reaction|...")
    series: Optional[str] = Field(default=None, description="recurring series this belongs to, or null")
    series_confidence: Optional[float] = None

    # hook interpretation (raw hook language lives in raw_hook, not here)
    hook_types: List[str] = Field(default_factory=list)
    hook_primary_type: Optional[str] = None
    hook_structure: Optional[str] = None
    hook_evidence: List[HookEvidence] = Field(default_factory=list)
    hooks_stacked: Optional[bool] = Field(
        default=None, description="true if more than one distinct hook device is used in the opening"
    )
    spoken_vs_onscreen_hook: Optional[str] = Field(
        default=None, description="same | complementary | contradictory | onscreen_only | spoken_only | none"
    )
    hook_notes: Optional[str] = None

    # substance
    summary: str
    key_claims: List[Claim] = Field(default_factory=list)
    archetypes: List[Archetype] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    notable_phrases: List[NotablePhrase] = Field(default_factory=list)

    # delivery / ask
    tone: Optional[str] = None
    secondary_tones: List[str] = Field(default_factory=list)
    cta: Optional[str] = Field(default=None, description="verbatim CTA line, or null")
    cta_type: Optional[str] = None

    # meta
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    next_video_ideas: List[str] = Field(default_factory=list)

    # --- normalisers ------------------------------------------------------
    @field_validator("content_mode")
    @classmethod
    def _norm_mode(cls, v):
        return _normalize(v, CONTENT_MODES)

    @field_validator("content_format")
    @classmethod
    def _norm_format(cls, v):
        return _normalize(v, CONTENT_FORMATS)

    @field_validator("cta_type")
    @classmethod
    def _norm_cta(cls, v):
        return _normalize(v, CTA_TYPES)

    @field_validator("tone")
    @classmethod
    def _norm_tone(cls, v):
        return _normalize(v, TONES)

    @field_validator("hook_structure")
    @classmethod
    def _norm_structure(cls, v):
        return _normalize(v, HOOK_STRUCTURES)

    @field_validator("hook_types", mode="before")
    @classmethod
    def _norm_hook_types(cls, v):
        if isinstance(v, str):
            v = [v]
        return [_normalize(x, HOOK_TYPES) or "other" for x in (v or []) if x]

    @field_validator("hook_primary_type")
    @classmethod
    def _norm_primary(cls, v):
        return _normalize(v, HOOK_TYPES)

    @field_validator("main_topic")
    @classmethod
    def _clean_topic(cls, v: str) -> str:
        return (v or "unknown").strip()

    def resolved_primary_hook(self) -> Optional[str]:
        return self.hook_primary_type or (self.hook_types[0] if self.hook_types else None)


# --- schema text handed to the model ---------------------------------------
def schema_instructions() -> str:
    return f"""Return ONE JSON object with exactly these keys:

{{
  "main_topic": string,                  // 2-4 words, the dominant subject
  "subtopic": string|null,               // narrower angle within main_topic
  "topics": [string],                    // 1-5 topical tags
  "dating_category": string|null,        // dating-specific bucket if applicable
  "target_audience": string|null,        // who this is aimed at

  "content_format": string,              // {" | ".join(CONTENT_FORMATS)}
  "content_mode": string,                // {" | ".join(CONTENT_MODES)}
  "series": string|null,                 // recurring series/format name, else null
  "series_confidence": number|null,      // 0-1

  "hook_types": [string],                // from: {", ".join(HOOK_TYPES)}
  "hook_primary_type": string,           // the single dominant one
  "hook_structure": string,              // from: {", ".join(HOOK_STRUCTURES)}
  "hook_evidence": [                     // one entry per hook_type, MUST quote verbatim
    {{"hook_type": string, "quote": string, "source": "spoken"|"onscreen"|"caption", "reason": string}}
  ],
  "hooks_stacked": boolean,              // more than one hook device in the opening
  "spoken_vs_onscreen_hook": string,     // same|complementary|contradictory|onscreen_only|spoken_only|none
  "hook_notes": string|null,

  "summary": string,                     // 1-3 sentences, factual
  "key_claims": [{{"claim": string, "quote": string}}],
  "archetypes": [{{"label": string, "category": string, "mention": string}}],
  "entities": [string],
  "notable_phrases": [{{"phrase": string, "why": string}}],

  "tone": string,                        // from: {", ".join(TONES)}
  "secondary_tones": [string],
  "cta": string|null,                    // the VERBATIM call-to-action line, else null
  "cta_type": string,                    // {" | ".join(CTA_TYPES)}

  "confidence": number,                  // 0-1, your confidence in this labelling
  "next_video_ideas": [string]           // 2-4 concrete follow-up video ideas
}}"""


def evidence_is_grounded(quote: str, haystacks: List[str], min_len: int = 6) -> bool:
    """Loose verbatim check: is this quote actually present in the raw material?"""
    if not quote or len(quote.strip()) < min_len:
        return True  # too short to judge
    needle = " ".join(quote.lower().split())
    for hay in haystacks:
        if not hay:
            continue
        if needle in " ".join(hay.lower().split()):
            return True
    return False


def to_db_row(
    shortcode: str,
    classification: ReelClassification,
    *,
    provider: str,
    model: str,
    opening_sentence: Optional[str],
    duration_sec: Optional[float],
) -> Dict[str, Any]:
    from ..utils import to_json

    return {
        "shortcode": shortcode,
        "provider": provider,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "main_topic": classification.main_topic,
        "subtopic": classification.subtopic,
        "dating_category": classification.dating_category,
        "content_format": classification.content_format,
        "content_mode": classification.content_mode,
        "series": classification.series,
        "tone": classification.tone,
        "cta": classification.cta,
        "cta_type": classification.cta_type,
        "summary": classification.summary,
        "opening_sentence": opening_sentence,
        "hook_primary_type": classification.resolved_primary_hook(),
        "hook_structure": classification.hook_structure,
        "length_bucket": bucket_length(duration_sec),
        "target_audience": classification.target_audience,
        "confidence": classification.confidence,
        "hook_types_json": to_json(classification.hook_types),
        "hook_evidence_json": to_json([e.model_dump() for e in classification.hook_evidence]),
        "topics_json": to_json(classification.topics),
        "archetypes_json": to_json([a.model_dump() for a in classification.archetypes]),
        "key_claims_json": to_json([c.model_dump() for c in classification.key_claims]),
        "notable_phrases_json": to_json([p.model_dump() for p in classification.notable_phrases]),
        "entities_json": to_json(classification.entities),
        "classification_json": to_json(classification.model_dump()),
    }
