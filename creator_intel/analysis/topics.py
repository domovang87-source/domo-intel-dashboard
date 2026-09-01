"""Topic merging: collapse LLM label fragmentation.

Classification runs one reel at a time, so the model can't keep its own labels
consistent ("dating in Japan" / "Japanese dating" / "dating in Tokyo"). This
pass looks at ALL distinct topic labels together and produces an alias map to
canonical topics, stored in `topic_aliases` and applied at query time — the
per-reel `analysis.main_topic` (the model's original words) is never rewritten,
in keeping with the never-destroy-source-data rule.

Idempotent and incremental: already-mapped aliases are kept stable so charts
don't reshuffle; only new labels are sent to the model.
"""
from __future__ import annotations

from typing import Dict, Optional

from ..logging_setup import get_logger
from ..utils import json_from_llm, utcnow_iso

log = get_logger(__name__)

MERGE_SYSTEM = """You normalise topic labels for ONE creator's video library.

CRITICAL CONTEXT: all videos come from a single creator working in a single
niche. That means the niche word itself ("dating", "fitness", "money") is NOT
a topic — it describes the whole library. The topic is the ANGLE within the
niche. A canonical label that could describe most of the library is useless
and forbidden.

Rules:
1. Merge ONLY true synonyms and fragments of the same specific subject:
   "dating in Japan" / "Japanese dating" / "Tokyo dating" -> "Dating in Japan".
   "dating advice" / "dating tips" -> "General Dating Advice".
2. Different angles stay separate even though they share the niche word:
   "dating psychology", "approaching women", "Albanian dating",
   "San Francisco dating" are FOUR different topics. When in doubt, do NOT merge.
3. Never output a canonical label that is just the niche word ("Dating") or
   would absorb more than ~30% of the labels.
4. Canonical names: short Title-Case noun phrases, 1-4 words.
5. Reuse the provided EXISTING canonical names whenever a new label fits one —
   stability matters more than elegance.
6. Every input label must appear in exactly one cluster.

Return raw JSON only:
{"clusters": [{"canonical": "Dating in Japan", "aliases": ["dating in Japan", "Japanese dating"]}]}"""


def _ensure_table(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS topic_aliases ("
        "alias TEXT PRIMARY KEY, canonical TEXT NOT NULL, created_at TEXT)"
    )


def load_alias_map(conn) -> Dict[str, str]:
    _ensure_table(conn)
    return {
        r["alias"]: r["canonical"]
        for r in conn.execute("SELECT alias, canonical FROM topic_aliases")
    }


def merge_topics(
    conn,
    *,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    force: bool = False,
) -> Dict[str, int]:
    from .providers import get_provider

    _ensure_table(conn)
    counts = {
        r["main_topic"]: r["n"]
        for r in conn.execute(
            "SELECT main_topic, COUNT(*) n FROM analysis "
            "WHERE main_topic IS NOT NULL AND main_topic <> '' GROUP BY main_topic"
        )
    }
    if not counts:
        return {"new_aliases": 0, "canonicals": 0}

    existing = load_alias_map(conn)
    if force:
        conn.execute("DELETE FROM topic_aliases")
        existing = {}
    unmapped = [t for t in counts if t.lower() not in existing]
    if not unmapped:
        return {"new_aliases": 0, "canonicals": len(set(existing.values()))}

    canonicals = sorted(set(existing.values()))
    topic_lines = "\n".join(f"- {t}  ({counts[t]} videos)" for t in sorted(counts))
    prompt = (
        f"EXISTING canonical topics (reuse when fitting):\n"
        + ("\n".join(f"- {c}" for c in canonicals) if canonicals else "(none yet)")
        + f"\n\nALL topic labels in the library right now:\n{topic_lines}\n\n"
        "Cluster every label above."
    )

    provider = get_provider(provider_name)
    resp = provider.complete(prompt, system=MERGE_SYSTEM, model=model, max_tokens=8000)
    payload = json_from_llm(resp.text)
    clusters = payload.get("clusters") if isinstance(payload, dict) else payload
    if not isinstance(clusters, list):
        raise ValueError(f"unexpected merge response shape: {type(clusters)}")

    now = utcnow_iso()
    new_count = 0
    covered = set()
    for cluster in clusters:
        canonical = (cluster.get("canonical") or "").strip()
        if not canonical:
            continue
        aliases = set(a.strip() for a in cluster.get("aliases") or [] if a and a.strip())
        aliases.add(canonical)
        for alias in aliases:
            key = alias.lower()
            covered.add(key)
            if key in existing and not force:
                continue  # stability: never remap an existing alias
            conn.execute(
                "INSERT INTO topic_aliases (alias, canonical, created_at) VALUES (?,?,?) "
                "ON CONFLICT(alias) DO NOTHING",
                (key, canonical, now),
            )
            new_count += 1

    # Anything the model forgot maps to itself so it is never silently dropped.
    for topic in counts:
        if topic.lower() not in covered and topic.lower() not in existing:
            conn.execute(
                "INSERT INTO topic_aliases (alias, canonical, created_at) VALUES (?,?,?) "
                "ON CONFLICT(alias) DO NOTHING",
                (topic.lower(), topic, now),
            )
            new_count += 1

    conn.commit()
    total_canonicals = conn.execute(
        "SELECT COUNT(DISTINCT canonical) c FROM topic_aliases"
    ).fetchone()["c"]
    log.info("topic merge: %s new alias(es), %s canonical topics", new_count, total_canonicals)
    return {"new_aliases": new_count, "canonicals": total_canonicals}
