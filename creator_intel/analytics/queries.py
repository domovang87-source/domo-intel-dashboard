"""Analytics helpers shared by the dashboard and ad-hoc scripts.

Everything returns a pandas DataFrame built from `v_reel_full`, so raw
observations and AI labels sit side by side and can be cross-tabulated:
"do my lawyer videos outperform" is a groupby on an exploded archetype column,
not a bespoke pipeline.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ..db.database import connect, query_df

# Columns that hold JSON arrays and are commonly exploded for grouping.
JSON_LIST_COLUMNS = {
    "hook_types_json": "hook_type",
    "topics_json": "topic",
    "archetypes_json": "archetype",
    "key_claims_json": "claim",
    "notable_phrases_json": "phrase",
}


def load_reels(db_path=None) -> pd.DataFrame:
    df = query_df("SELECT * FROM v_reel_full ORDER BY taken_at_ts DESC", db_path=db_path)
    if df.empty:
        return df
    # Apply the topic-alias map (see analysis/topics.py). The model's original
    # label is preserved in main_topic_raw.
    try:
        aliases = query_df("SELECT alias, canonical FROM topic_aliases", db_path=db_path)
        if not aliases.empty and "main_topic" in df:
            amap = dict(zip(aliases["alias"], aliases["canonical"]))
            df["main_topic_raw"] = df["main_topic"]
            df["main_topic"] = df["main_topic"].map(
                lambda t: amap.get(t.lower(), t) if isinstance(t, str) else t
            )
    except Exception:
        pass  # table not created yet — raw topics are shown as-is
    # Curated taxonomy (topic_curated) beats everything: it is the deliberate,
    # whole-library re-classification. Falls through gracefully if absent.
    try:
        curated = query_df("SELECT shortcode, topic FROM topic_curated", db_path=db_path)
        if not curated.empty and "main_topic" in df:
            cmap = dict(zip(curated["shortcode"], curated["topic"]))
            if "main_topic_raw" not in df:
                df["main_topic_raw"] = df["main_topic"]
            df["main_topic"] = df.apply(
                lambda r: cmap.get(r["shortcode"], r["main_topic"]), axis=1)
    except Exception:
        pass
    df["taken_at"] = pd.to_datetime(df["taken_at_utc"], errors="coerce", utc=True)
    df["posted_date"] = df["taken_at"].dt.date
    for col in ("view_count", "like_count", "comment_count", "duration_sec"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["engagement_rate"] = pd.to_numeric(df.get("engagement_rate"), errors="coerce")
    return df


def overview(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"videos": 0}
    has_views = df["view_count"].notna()
    return {
        "videos": len(df),
        "analyzed": int(df["has_analysis"].sum()) if "has_analysis" in df else 0,
        "transcribed": int(df["has_transcript"].sum()) if "has_transcript" in df else 0,
        "with_visual": int(df["has_visual"].sum()) if "has_visual" in df else 0,
        "with_views": int(has_views.sum()),
        "total_views": int(df.loc[has_views, "view_count"].sum()) if has_views.any() else 0,
        "median_views": float(df.loc[has_views, "view_count"].median()) if has_views.any() else None,
        "topics": int(df["main_topic"].nunique()) if "main_topic" in df else 0,
        "series": int(df["series"].dropna().nunique()) if "series" in df else 0,
        "date_min": df["taken_at"].min(),
        "date_max": df["taken_at"].max(),
        "total_minutes": round(float(df["duration_sec"].sum() or 0) / 60.0, 1),
    }


def by_dimension(
    df: pd.DataFrame,
    dimension: str,
    *,
    metric: str = "view_count",
    min_count: int = 1,
) -> pd.DataFrame:
    """videos / median / mean / total of `metric`, grouped by `dimension`."""
    if df.empty or dimension not in df:
        return pd.DataFrame()
    work = df[df[dimension].notna() & (df[dimension].astype(str).str.len() > 0)].copy()
    if work.empty:
        return pd.DataFrame()
    grouped = work.groupby(dimension).agg(
        videos=(dimension, "size"),
        median_metric=(metric, "median"),
        mean_metric=(metric, "mean"),
        total_metric=(metric, "sum"),
        max_metric=(metric, "max"),
    )
    grouped = grouped[grouped["videos"] >= min_count]
    if grouped.empty:
        return grouped.reset_index()
    overall = work[metric].median()
    grouped["vs_median_pct"] = (
        ((grouped["median_metric"] - overall) / overall * 100).round(1) if overall else None
    )
    return grouped.sort_values("median_metric", ascending=False).reset_index()


def exploded(df: pd.DataFrame, json_column: str, *, value_key: Optional[str] = None) -> pd.DataFrame:
    """One row per (reel, list item) for a JSON-array column.

    Handles both plain string arrays (topics) and object arrays (archetypes),
    pulling `value_key` out of objects when given.
    """
    if df.empty or json_column not in df:
        return pd.DataFrame()
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        raw = row.get(json_column)
        if not raw:
            continue
        try:
            items = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                value = item.get(value_key or "label") or item.get("phrase") or item.get("claim")
                extra = item
            else:
                value, extra = item, {}
            if not value:
                continue
            records.append(
                {
                    "shortcode": row["shortcode"],
                    "value": str(value).strip(),
                    "view_count": row.get("view_count"),
                    "like_count": row.get("like_count"),
                    "comment_count": row.get("comment_count"),
                    "taken_at": row.get("taken_at"),
                    "main_topic": row.get("main_topic"),
                    "series": row.get("series"),
                    "detail": extra,
                }
            )
    return pd.DataFrame(records)


def archetype_performance(df: pd.DataFrame, *, min_count: int = 2) -> pd.DataFrame:
    """Answers "do my lawyer videos outperform?" — grouped by mentioned archetype."""
    ex = exploded(df, "archetypes_json", value_key="label")
    if ex.empty:
        return ex
    ex["value"] = ex["value"].str.lower().str.strip()
    grouped = ex.groupby("value").agg(
        videos=("shortcode", "nunique"),
        median_views=("view_count", "median"),
        mean_views=("view_count", "mean"),
        max_views=("view_count", "max"),
    )
    grouped = grouped[grouped["videos"] >= min_count]
    baseline = df["view_count"].median()
    if baseline:
        grouped["vs_median_pct"] = ((grouped["median_views"] - baseline) / baseline * 100).round(1)
    return grouped.sort_values("median_views", ascending=False).reset_index().rename(
        columns={"value": "archetype"}
    )


def hook_language_table(df: pd.DataFrame, *, limit: int = 200) -> pd.DataFrame:
    """The exact-wording table: raw opening language next to performance.

    This is the point of the whole project — comparing
    "Guys, this is super manipulative" with "Guys, this is a big secret".
    """
    cols = [
        "shortcode", "url", "taken_at", "view_count", "like_count", "comment_count",
        "spoken_0_2", "spoken_0_5", "spoken_first_sentence", "onscreen_hook",
        "hooks_differ", "hook_primary_type", "hook_structure", "hook_types_json",
        "time_to_first_word", "words_in_first_2s", "wps_first_5s",
        "cuts_first_5s", "main_topic", "series",
    ]
    available = [c for c in cols if c in df.columns]
    out = df[available].copy()
    if "view_count" in out:
        out = out.sort_values("view_count", ascending=False, na_position="last")
    return out.head(limit)


def phrase_frequency(df: pd.DataFrame, *, top_n: int = 60) -> pd.DataFrame:
    """Recurring language patterns across the library."""
    ex = exploded(df, "notable_phrases_json", value_key="phrase")
    if ex.empty:
        return ex
    ex["norm"] = ex["value"].str.lower().str.strip()
    grouped = ex.groupby("norm").agg(
        uses=("shortcode", "nunique"),
        median_views=("view_count", "median"),
        example=("value", "first"),
    )
    return grouped.sort_values(["uses", "median_views"], ascending=False).head(top_n).reset_index(
        drop=True
    ).rename(columns={"example": "phrase"})[["phrase", "uses", "median_views"]]


def correlations(df: pd.DataFrame, target: str = "view_count") -> pd.DataFrame:
    """Spearman correlation of numeric raw measurements against performance.

    Rank-based on purpose: view counts are heavily skewed, so Pearson would be
    dominated by one or two outliers.
    """
    if df.empty or target not in df:
        return pd.DataFrame()
    candidates = [
        "duration_sec", "time_to_first_word", "words_in_first_2s", "words_in_first_5s",
        "wps_first_5s", "cut_count", "cuts_first_5s", "avg_shot_sec", "words_per_minute",
        "speech_wpm", "transcript_word_count", "text_overlay_count", "hook_text_similarity",
        "zoom_events", "like_count", "comment_count",
    ]
    work = df.copy()
    rows = []
    for col in candidates:
        if col not in work or col == target:
            continue
        series = pd.to_numeric(work[col], errors="coerce")
        pair = pd.concat([series, pd.to_numeric(work[target], errors="coerce")], axis=1).dropna()
        if len(pair) < 5 or pair.iloc[:, 0].nunique() < 3:
            continue
        # Spearman = Pearson on ranks; computed manually so scipy isn't needed.
        rho = pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank())
        if pd.isna(rho):
            continue
        rows.append({"measurement": col, "spearman_rho": round(float(rho), 3), "n": len(pair)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.reindex(out["spearman_rho"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def search_transcripts(query: str, *, limit: int = 50, db_path=None) -> pd.DataFrame:
    """Full-text search across verbatim transcripts, captions and overlay text."""
    if not query.strip():
        return pd.DataFrame()
    with connect(db_path, read_only=True) as conn:
        try:
            rows = conn.execute(
                """
                SELECT f.shortcode,
                       snippet(transcript_fts, 1, '**', '**', ' … ', 20) AS transcript_hit,
                       snippet(transcript_fts, 3, '**', '**', ' … ', 12) AS overlay_hit,
                       bm25(transcript_fts) AS score
                FROM transcript_fts f
                WHERE transcript_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        except Exception:
            # FTS syntax errors (unbalanced quotes etc.) fall back to LIKE.
            rows = conn.execute(
                """
                SELECT r.shortcode, substr(t.text, 1, 300) AS transcript_hit,
                       '' AS overlay_hit, 0 AS score
                FROM reels r JOIN transcripts t ON t.shortcode = r.shortcode
                WHERE t.text LIKE ? LIMIT ?
                """,
                (f"%{query}%", limit),
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# Blueprint findings — every measured lever as "typical views with vs without".
# Computed live from the DB so numbers stay current as the archive grows.
# ---------------------------------------------------------------------------
def _vs(df: pd.DataFrame, mask, label_yes: str, label_no: str, extra: str = "") -> Dict[str, Any]:
    yes, no = df[mask], df[~mask]
    if len(yes) < 5 or len(no) < 5:
        return {}
    return {
        "habit": label_yes,
        "with": int(yes["v"].median()), "n_with": len(yes),
        "without": int(no["v"].median()), "n_without": len(no),
        "vs": label_no, "note": extra,
    }


def blueprint_delivery(db_path=None) -> pd.DataFrame:
    d = query_df("""
        SELECT r.shortcode, r.view_count v, p.rms_cv, p.median_f0,
               o.open_ratio, o.pauses_pm, o.dramatic,
               e.opens_smiling, e.nose_scrunch, e.laugh_frames, e.frames_json,
               ed.music_detected, a.content_mode
        FROM reels r
        LEFT JOIN raw_prosody p USING(shortcode)
        LEFT JOIN raw_opening o USING(shortcode)
        LEFT JOIN raw_expression e USING(shortcode)
        LEFT JOIN raw_editing ed USING(shortcode)
        LEFT JOIN analysis a USING(shortcode)
        WHERE r.view_count IS NOT NULL""", db_path=db_path)

    def early_laugh(fj):
        try:
            return any(f.get("laugh_face") and float(f.get("t_sec", 99)) <= 10
                       for f in json.loads(fj))
        except Exception:
            return False

    d["laugh10"] = d["frames_json"].map(early_laugh)
    rows = []
    dd = d[d["dramatic"].notna()]
    rows.append(_vs(dd, dd["dramatic"] > 0, "Dramatic pauses (1.2s+ stop)", "none"))
    rows.append(_vs(d, d["laugh10"], "Laugh in first 10s", "no early laugh"))
    dp = d[d["rms_cv"].notna()]
    punchy_cut = dp["rms_cv"].quantile(2 / 3)
    flat_cut = dp["rms_cv"].quantile(1 / 3)
    dpp = dp[(dp["rms_cv"] >= punchy_cut) | (dp["rms_cv"] <= flat_cut)]
    rows.append(_vs(dpp, dpp["rms_cv"] >= punchy_cut, "Punchy voice (volume contrast)", "flat delivery"))
    do = d[d["open_ratio"].notna()]
    rows.append(_vs(do, do["open_ratio"] < 0.75, "Whisper-soft open (first 3s)", "normal open"))
    dm = d[d["music_detected"].notna()]
    rows.append(_vs(dm, dm["music_detected"] == 1, "Music bed on", "no music"))
    de = d[d["nose_scrunch"].notna()]
    rows.append(_vs(de, de["nose_scrunch"] > 0, "Nose scrunch 👃", "no scrunch"))
    df0 = d[d["median_f0"].notna()]
    hi_cut = df0["median_f0"].quantile(2 / 3)
    rows.append(_vs(df0, df0["median_f0"] < hi_cut, "Natural/low register", "high girly register"))
    dmode = d[d["content_mode"].notna()]
    rows.append(_vs(dmode, dmode["content_mode"] != "story", "Advice/tactics mode", "pure story mode"))
    return pd.DataFrame([r for r in rows if r])


def blueprint_script(db_path=None) -> pd.DataFrame:
    import re as _re
    d = query_df("""SELECT r.view_count v, t.text, e.avg_shot_sec, e.cuts_first_5s,
                           t.segments_json
                    FROM reels r JOIN transcripts t USING(shortcode)
                    LEFT JOIN raw_editing e USING(shortcode)
                    WHERE r.view_count IS NOT NULL""", db_path=db_path)
    PROF = _re.compile(r"\b(fuck\w*|shit\w*|ass|damn|hell|bitch\w*)\b", _re.I)
    TACTIC = _re.compile(r"here's (how|what|the)|this is how|what you (do|say)|you're gonna|step one|number one", _re.I)
    d["prof"] = d["text"].fillna("").map(lambda t: len(PROF.findall(t)))

    def payoff_time(sj):
        try:
            for seg in json.loads(sj):
                if TACTIC.search(seg.get("text", "")):
                    return float(seg["start"])
        except Exception:
            pass
        return None

    d["payoff"] = d["segments_json"].map(payoff_time)
    rows = []
    heavy_cut = d.loc[d["prof"] > 0, "prof"].median() if (d["prof"] > 0).any() else 99
    dh = d[(d["prof"] == 0) | (d["prof"] >= heavy_cut)]
    rows.append(_vs(dh, dh["prof"] >= heavy_cut, "Heavy profanity", "clean script"))
    dp = d[d["payoff"].notna()]
    rows.append(_vs(dp, dp["payoff"] <= 5, "Tactic lands inside 5s", "tactic held 5s+"))
    ds = d[d["avg_shot_sec"].notna()]
    long_cut = ds["avg_shot_sec"].quantile(2 / 3)
    fast_cut = ds["avg_shot_sec"].quantile(1 / 3)
    dss = ds[(ds["avg_shot_sec"] >= long_cut) | (ds["avg_shot_sec"] <= fast_cut)]
    rows.append(_vs(dss, dss["avg_shot_sec"] >= long_cut, "Long takes", "fast cuts"))
    dc = d[d["cuts_first_5s"].notna()]
    rows.append(_vs(dc, dc["cuts_first_5s"] == 0, "Zero cuts in first 5s", "1+ cuts in first 5s"))
    return pd.DataFrame([r for r in rows if r])


def blueprint_posting(db_path=None) -> pd.DataFrame:
    d = query_df("""SELECT r.view_count v, r.taken_at_utc, vis.other_person
                    FROM reels r LEFT JOIN raw_visual vis USING(shortcode)
                    WHERE r.view_count IS NOT NULL""", db_path=db_path)
    d["dt"] = pd.to_datetime(d["taken_at_utc"], utc=True).dt.tz_convert("America/Los_Angeles")
    rows = []
    rows.append(_vs(d, d["dt"].dt.dayofweek >= 4, "Posted Fri–Sun", "Mon–Thu"))
    rows.append(_vs(d, d["dt"].dt.hour.between(13, 17), "Posted afternoon (PT)", "other times"))
    viral_times = d.loc[d["v"] >= 500000, "dt"].tolist()
    d["after_viral"] = d["dt"].map(
        lambda t: any(0 < (t - vt).total_seconds() <= 48 * 3600 for vt in viral_times))
    rows.append(_vs(d, d["after_viral"], "Within 48h after a 500k+ video", "normal window",
                    "ride hot streaks — post daily while hot"))
    dv = d[d["other_person"].notna()]
    rows.append(_vs(dv, dv["other_person"] != 1, "Solo on camera", "another person visible"))
    return pd.DataFrame([r for r in rows if r])


def hook_word_lexicon(db_path=None, min_n: int = 12) -> pd.DataFrame:
    import re as _re
    from collections import defaultdict as _dd
    d = query_df("""SELECT r.view_count v, h.spoken_0_5 FROM reels r
                    JOIN raw_hook h USING(shortcode) WHERE r.view_count IS NOT NULL""",
                 db_path=db_path)
    lex = _dd(list)
    for r in d.itertuples():
        for w in set(_re.findall(r"[a-z']+", (r.spoken_0_5 or "").lower())):
            lex[w].append(r.v)
    rows = [{"word": w, "videos": len(vs), "typical_views": int(pd.Series(vs).median())}
            for w, vs in lex.items() if len(vs) >= min_n]
    return pd.DataFrame(rows).sort_values("typical_views", ascending=False).reset_index(drop=True)
