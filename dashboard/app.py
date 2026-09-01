"""Streamlit dashboard for creator-intel.

    python -m creator_intel dashboard
    # or: streamlit run dashboard/app.py

Reads the SQLite database read-only, so it is safe to leave open while the
pipeline is still ingesting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from creator_intel.analytics.queries import (  # noqa: E402
    archetype_performance,
    by_dimension,
    correlations,
    exploded,
    hook_language_table,
    load_reels,
    overview,
    phrase_frequency,
    search_transcripts,
)
from creator_intel.config import settings  # noqa: E402

st.set_page_config(page_title="Creator Intel", page_icon="🎬", layout="wide")

METRIC_LABELS = {
    "view_count": "views",
    "like_count": "likes",
    "comment_count": "comments",
    "share_count": "shares",
    "save_count": "saves",
    "reach_count": "reach",
    "follows_count": "follows gained",
    "engagement_rate": "engagement rate",
}


@st.cache_data(ttl=60)
def _load():
    return load_reels()


def _fmt(value, digits: int = 0) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}"
    return str(value)


def bar(df: pd.DataFrame, x: str, y: str, *, title: str = "", horizontal: bool = True, color: str = None):
    if df.empty:
        st.info("Not enough data yet.")
        return
    enc_x, enc_y = (alt.X(f"{y}:Q", title=y), alt.Y(f"{x}:N", sort="-x", title=None)) if horizontal else (
        alt.X(f"{x}:N", sort="-y", title=None), alt.Y(f"{y}:Q", title=y)
    )
    chart = alt.Chart(df).mark_bar(cornerRadius=3).encode(
        x=enc_x, y=enc_y,
        color=alt.Color(f"{color}:Q", legend=None, scale=alt.Scale(scheme="reds")) if color else alt.value("#dc2626"),
        tooltip=list(df.columns),
    ).properties(height=max(200, 26 * len(df)) if horizontal else 320, title=title)
    st.altair_chart(chart, use_container_width=True)


# ---------------------------------------------------------------------------
# load + guard
# ---------------------------------------------------------------------------
if not settings.db_path.exists():
    st.error(f"No database at `{settings.db_path}`. Run `python -m creator_intel init` first.")
    st.stop()

df_all = _load()
if df_all.empty:
    st.title("🎬 Creator Intel")
    st.warning("The database is empty. Run `python -m creator_intel run --limit 5` to archive some Reels.")
    st.stop()

# ---------------------------------------------------------------------------
# sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

# Only offer metrics that actually have data (views/shares/saves appear once
# an Insights CSV has been imported).
metric_options = [
    k for k in METRIC_LABELS if k in df_all.columns and df_all[k].notna().any()
] or list(METRIC_LABELS)
metric = st.sidebar.selectbox(
    "Performance metric", metric_options, format_func=lambda k: METRIC_LABELS[k]
)

dates = df_all["taken_at"].dropna()
if not dates.empty:
    dmin, dmax = dates.min().date(), dates.max().date()
    date_range = st.sidebar.date_input("Posted between", value=(dmin, dmax), min_value=dmin, max_value=dmax)
else:
    date_range = None


def _multi(label: str, column: str):
    if column not in df_all:
        return []
    options = sorted(df_all[column].dropna().astype(str).unique())
    return st.sidebar.multiselect(label, options) if options else []


sel_topic = _multi("Topic", "main_topic")
sel_series = _multi("Series", "series")
sel_hook = _multi("Hook type", "hook_primary_type")
sel_mode = _multi("Mode (story/advice/…)", "content_mode")
sel_category = _multi("Dating category", "dating_category")

min_views = 0
if df_all["view_count"].notna().any():
    max_views = int(df_all["view_count"].max())
    min_views = st.sidebar.slider("Minimum views", 0, max_views, 0, step=max(1, max_views // 100))

df = df_all.copy()
if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0], tz="UTC"), pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1)
    df = df[(df["taken_at"] >= start) & (df["taken_at"] < end)]
for column, selection in (
    ("main_topic", sel_topic), ("series", sel_series), ("hook_primary_type", sel_hook),
    ("content_mode", sel_mode), ("dating_category", sel_category),
):
    if selection:
        df = df[df[column].astype(str).isin(selection)]
if min_views:
    df = df[df["view_count"].fillna(0) >= min_views]

st.sidebar.caption(f"{len(df)} of {len(df_all)} videos match")
st.sidebar.divider()
st.sidebar.caption(f"DB: `{settings.db_path}`")
if st.sidebar.button("Reload data"):
    st.cache_data.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------
st.title("🎬 Creator Intel")
stats = overview(df)
c = st.columns(6)
c[0].metric("Videos", _fmt(stats["videos"]))
c[1].metric("Analysed", _fmt(stats.get("analyzed")))
c[2].metric("Total views", _fmt(stats.get("total_views")))
c[3].metric("Median views", _fmt(stats.get("median_views")))
c[4].metric("Topics", _fmt(stats.get("topics")))
c[5].metric("Minutes archived", _fmt(stats.get("total_minutes"), 1))

if df["view_count"].isna().all():
    st.info(
        "No view counts are stored yet. Instagram only exposes view counts for some posts via "
        "the web interface — topic/hook analysis still works, ranked-by-views does not."
    )

HOOK_MEANINGS = {
    "curiosity": "opens a loop — you have to keep watching to find out",
    "secret": "promises insider info most people don't know",
    "warning": "tells you to stop doing something / danger ahead",
    "controversial_claim": "says something people will argue about",
    "negative_claim": "leads with what's wrong or doesn't work",
    "demographic_callout": "names exactly who it's for ('YC founders…')",
    "direct_address": "talks straight at YOU ('Guys, …')",
    "command": "orders the viewer to do something",
    "promise": "guarantees a result if you keep watching",
    "list": "counts things ('3 ways to…')",
    "story_opening": "drops you into a story mid-action",
    "question": "opens by asking you something",
    "confession": "admits something personal",
    "authority_claim": "leads with credentials or experience",
    "wait_until_the_end": "tells you the payoff is at the end",
    "the_last_one_is_best": "'the last one is the best'",
    "manipulative_framing": "frames the content as manipulation/taboo",
    "shocking_framing": "deliberately shocking angle",
    "specific_person_or_profession": "about a specific type of person (lawyer, ex…)",
    "archetype_callout": "calls out a dating archetype",
    "statistic": "opens with a number or stat",
    "myth_bust": "'everything you know about X is wrong'",
    "contrarian": "takes the opposite of common advice",
    "relatable_scenario": "'you know when she…' — a familiar situation",
    "pattern_interrupt": "something unexpected that stops the scroll",
    "social_proof": "leans on results or other people's approval",
    "urgency": "act now / limited time framing",
    "fear": "leads with fear of loss or failure",
    "flex": "leads with success/status display",
}

PAGES = ["⭐ Start Here", "🧬 Blueprint", "🔍 Drill Down", "🎲 Calendar", "Hooks", "Videos", "Search", "Reel Detail", "All Tables"]

# Programmatic navigation (chart clicks land here BEFORE the nav widget renders,
# because Streamlit forbids changing a widget's state after it's instantiated).
if "_goto" in st.session_state:
    _goto = st.session_state.pop("_goto")
    st.session_state["nav"] = _goto.get("page", PAGES[0])
    if "dim" in _goto:
        st.session_state["drill_dim"] = _goto["dim"]
    if "pick" in _goto:
        st.session_state["pending_pick"] = _goto["pick"]
if "nav" not in st.session_state:
    st.session_state["nav"] = PAGES[0]

# st.radio keeps its state across reruns — unlike st.tabs, which resets to the
# first tab on every widget interaction (the "why am I back on Start Here" bug).
page = st.radio("Navigate", PAGES, horizontal=True, key="nav", label_visibility="collapsed")
st.divider()


def _jump(dim_label: str, value):
    st.session_state["_goto"] = {"page": "🔍 Drill Down", "dim": dim_label, "pick": value}
    st.rerun()


# ---------------------------------------------------------------------------
# ⭐ Start Here
# ---------------------------------------------------------------------------
if page == PAGES[0]:
    ml = METRIC_LABELS[metric]
    base_med = df[metric].median()

    st.subheader("What's working for you")
    st.caption(f"Auto-generated from your library, measured by **{ml}**. Change the metric in the sidebar.")

    insights = []
    ht = by_dimension(df, "hook_primary_type", metric=metric, min_count=15)
    if not ht.empty and base_med:
        top_h = ht.iloc[0]
        lift = (top_h["median_metric"] - base_med) / base_med * 100
        insights.append(("✅", f"Your best opener is **{str(top_h['hook_primary_type']).replace('_',' ')}** hooks — "
                         f"median {_fmt(top_h['median_metric'])} {ml} across {int(top_h['videos'])} videos "
                         f"({lift:+.0f}% vs your typical video)."))
        used_counts = df["hook_primary_type"].value_counts()
        if not used_counts.empty and used_counts.index[0] != top_h["hook_primary_type"]:
            mu_row = ht[ht["hook_primary_type"] == used_counts.index[0]]
            if not mu_row.empty:
                insights.append(("⚠️", f"But you USE **{str(used_counts.index[0]).replace('_',' ')}** most often "
                                 f"({int(used_counts.iloc[0])} videos, median {_fmt(mu_row.iloc[0]['median_metric'])}). "
                                 f"You're defaulting to your second-best weapon."))
    tp = by_dimension(df, "main_topic", metric=metric, min_count=8)
    if len(tp) >= 2 and base_med:
        top_t = tp.iloc[0]
        lift = (top_t["median_metric"] - base_med) / base_med * 100
        if lift > 15:
            insights.append(("✅", f"**{top_t['main_topic']}** is your topic outlier — "
                             f"median {_fmt(top_t['median_metric'])} {ml} ({lift:+.0f}% vs typical, {int(top_t['videos'])} videos)."))
    cm = by_dimension(df, "content_mode", metric=metric, min_count=8)
    if len(cm) >= 2 and base_med:
        worst = cm.iloc[-1]
        drop = (worst["median_metric"] - base_med) / base_med * 100
        if drop < -25:
            insights.append(("🛑", f"**{str(worst['content_mode']).replace('_',' ').title()}** videos underperform — "
                             f"median {_fmt(worst['median_metric'])} {ml} ({drop:.0f}% vs typical, {int(worst['videos'])} videos)."))
    if "follows_count" in df and df["follows_count"].notna().any():
        top_f = df.sort_values("follows_count", ascending=False).iloc[0]
        insights.append(("👥", f"Reels have gained you **{_fmt(df['follows_count'].sum())} followers** total. "
                         f"The single biggest: “{(top_f.get('spoken_0_2') or '')[:60]}…” (+{_fmt(top_f['follows_count'])})."))
    for icon, text in insights[:5]:
        st.markdown(f"{icon} &nbsp; {text}", unsafe_allow_html=True)
        st.markdown("")

    st.divider()
    tstats = df[df["main_topic"].notna()].groupby("main_topic").agg(
        followers=("follows_count", "sum"), views=("view_count", "sum"),
        videos=("shortcode", "size")).reset_index()
    tstats = tstats[tstats["videos"] >= 3]
    tstats["topic"] = tstats["main_topic"].astype(str).str.title()
    top_follow = tstats.sort_values("followers", ascending=False).head(6)
    top_views = tstats.sort_values("views", ascending=False).head(6)
    domain = list(dict.fromkeys(top_follow["topic"].tolist() + top_views["topic"].tolist()))
    topic_scale = alt.Scale(domain=domain, scheme="tableau10")

    left, right = st.columns(2)
    with left:
        st.subheader("Top 6 topics → followers")
        st.caption("Which subjects actually grow your account. **Click a bar to drill in.**")
        if not top_follow.empty:
            sel_f = alt.selection_point(fields=["main_topic"], name="pickf")
            chart_f = (
                alt.Chart(top_follow)
                .mark_bar(cornerRadius=4, cursor="pointer")
                .encode(
                    x=alt.X("followers:Q", title="followers gained"),
                    y=alt.Y("topic:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                    color=alt.Color("topic:N", scale=topic_scale, legend=None),
                    tooltip=["topic", "followers", "views", "videos"],
                )
                .add_params(sel_f)
                .properties(height=260)
            )
            ev_f = st.altair_chart(chart_f, use_container_width=True, on_select="rerun", key="sh_follow")
            try:
                pts = ev_f.selection.pickf
                if pts:
                    _jump("Topic", pts[0].get("main_topic"))
            except Exception:
                pass
    with right:
        st.subheader("Top 6 topics → views")
        st.caption("Same colors as the left chart. **Click a slice to drill in.**")
        if not top_views.empty:
            sel_v = alt.selection_point(fields=["main_topic"], name="pickv")
            chart_v = (
                alt.Chart(top_views)
                .mark_arc(innerRadius=58, cursor="pointer")
                .encode(
                    theta=alt.Theta("views:Q"),
                    color=alt.Color("topic:N", scale=topic_scale,
                                    legend=alt.Legend(title=None, orient="right")),
                    tooltip=["topic", "views", "followers", "videos"],
                )
                .add_params(sel_v)
                .properties(height=260)
            )
            ev_v = st.altair_chart(chart_v, use_container_width=True, on_select="rerun", key="sh_views")
            try:
                pts = ev_v.selection.pickv
                if pts:
                    _jump("Topic", pts[0].get("main_topic"))
            except Exception:
                pass

# ---------------------------------------------------------------------------
# 🔍 Drill Down
# ---------------------------------------------------------------------------
if page == PAGES[2]:
    ml = METRIC_LABELS[metric]
    DIMS = {
        "Topic": "main_topic",
        "Hook type": "hook_primary_type",
        "Hook structure": "hook_structure",
        "Format (story/advice/…)": "content_mode",
        "Series": "series",
        "Video length": "length_bucket",
        "Setting": "setting",
        "Speaking pace": "speaking_pace",
    }
    DIMS = {k: v for k, v in DIMS.items() if v in df.columns and df[v].notna().any()}

    dim_label = st.radio("Slice your library by:", list(DIMS), horizontal=True, key="drill_dim")
    dim = DIMS[dim_label]

    # Topics need >=3 videos before a median means anything (one-video
    # topics were topping the charts and misleading).
    grouped = by_dimension(df, dim, metric=metric, min_count=3 if dim == 'main_topic' else 2)
    if grouped.empty:
        st.info("Not enough data for this slice yet.")
    else:
        grouped = grouped.head(20)
        sel = alt.selection_point(fields=[dim], name="picked")
        chart = (
            alt.Chart(grouped)
            .mark_bar(cornerRadius=3, cursor="pointer")
            .encode(
                x=alt.X("median_metric:Q", title=f"median {ml}"),
                y=alt.Y(f"{dim}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
                color=alt.condition(sel, alt.value("#b8860b"), alt.value("#dc2626")),
                tooltip=[dim, "videos", "median_metric", "vs_median_pct"],
            )
            .add_params(sel)
            .properties(height=max(220, 30 * len(grouped)))
        )
        st.caption("**Click a bar** to open that slice below.")
        event = st.altair_chart(chart, use_container_width=True, on_select="rerun", key=f"drill_{dim}_{metric}")

        picked = st.session_state.pop("pending_pick", None)
        if picked is None:
            try:
                pts = event.selection.picked  # type: ignore[union-attr]
                if pts:
                    picked = pts[0].get(dim)
            except Exception:
                picked = None
        options = ["(click a bar above, or pick here)"] + grouped[dim].astype(str).tolist()
        fallback = st.selectbox("Drill into:", options, key=f"drillsel_{dim}")
        if picked is None and fallback != options[0]:
            picked = fallback

        if picked:
            seg = df[df[dim].astype(str) == str(picked)]
            if seg.empty:
                st.info("No videos in this slice with the current filters.")
            else:
                st.divider()
                display = str(picked).replace("_", " ").title() if dim != "main_topic" else str(picked)
                st.subheader(f"🔬 {display}")

                # plain-English meaning for hook types
                if dim == "hook_primary_type" and str(picked) in HOOK_MEANINGS:
                    st.markdown(f"*What this means: this opener {HOOK_MEANINGS[str(picked)]}.*")

                # deterministic summary — written from the numbers, no API needed
                base_med = df[metric].median()
                seg_med = seg[metric].median()
                parts = [f"**{len(seg)} videos.**"]
                if base_med and pd.notna(seg_med):
                    diff = (seg_med - base_med) / base_med * 100
                    verdict = "outperforms" if diff > 10 else ("underperforms" if diff < -10 else "performs about the same as")
                    parts.append(f"Median {_fmt(seg_med)} {ml} — {verdict} your typical video ({diff:+.0f}%).")
                if "share_count" in seg and seg["share_count"].notna().any() and df["share_count"].notna().any():
                    smult = seg["share_count"].median() / max(1, df["share_count"].median())
                    if smult >= 1.5:
                        parts.append(f"These get **{smult:.1f}× your usual shares** — the algorithm's favorite signal.")
                if "follows_count" in seg and seg["follows_count"].notna().any():
                    parts.append(f"Total followers gained: **{_fmt(seg['follows_count'].sum())}**.")
                best = seg.sort_values(metric, ascending=False).iloc[0]
                if best.get("spoken_0_2"):
                    parts.append(f"Your biggest one opens: “{best['spoken_0_2']}”")
                st.info(" ".join(parts))

                # TOP 4 CONCRETE EXAMPLES — front and center
                st.markdown("##### 🏆 Top 4 — steal these openings")
                for _, row in seg.sort_values(metric, ascending=False, na_position="last").head(4).iterrows():
                    line = f"**{_fmt(row.get(metric))} {ml}** — “{(row.get('spoken_0_2') or row.get('spoken_first_sentence') or '')[:90]}”"
                    if row.get("onscreen_hook"):
                        line += f" &nbsp;·&nbsp; on screen: *{str(row.get('onscreen_hook'))[:50]}*"
                    if row.get("url"):
                        line += f" &nbsp;[open]({row['url']})"
                    st.markdown(line)

                st.markdown(f"##### All videos in “{display}” (best first)")
                seg_show = seg.sort_values(metric, ascending=False, na_position="last")
                cols = [c for c in ["shortcode", "url", metric, "share_count", "follows_count",
                                    "spoken_0_2", "onscreen_hook", "main_topic", "taken_at_utc"] if c in seg_show.columns]
                st.dataframe(
                    seg_show[cols], use_container_width=True, hide_index=True, height=380,
                    column_config={"url": st.column_config.LinkColumn("url", display_text="open")},
                )

                with st.expander("Advanced: break this slice down further"):
                    cc_label = st.selectbox("Second dimension:", [l for l in DIMS if l != dim_label], key=f"cross_{dim}")
                    cross = by_dimension(seg, DIMS[cc_label], metric=metric, min_count=2).head(6)
                    if not cross.empty and len(cross) > 1:
                        ccdim = DIMS[cc_label]
                        cross[ccdim] = cross[ccdim].astype(str).str.replace("_", " ").str.title()
                        cchart = (
                            alt.Chart(cross)
                            .mark_bar(cornerRadius=3, color="#dc2626")
                            .encode(
                                x=alt.X("median_metric:Q", title=f"median {ml}"),
                                y=alt.Y(f"{ccdim}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
                                tooltip=[ccdim, "videos", "median_metric"],
                            )
                            .properties(height=34 * len(cross) + 20,
                                        title=f"{display} × {cc_label.lower()} — top {len(cross)}, median {ml}")
                        )
                        st.altair_chart(cchart, use_container_width=True)
                    else:
                        st.caption("This slice doesn't split further on that dimension.")
                st.caption("Copy a shortcode into the **Reel Detail** tab for the full forensics of any one video.")


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
if page == PAGES[8]:
    left, right = st.columns(2)
    with left:
        st.subheader("Posting cadence")
        cadence = df.dropna(subset=["taken_at"]).copy()
        if not cadence.empty:
            cadence["month"] = cadence["taken_at"].dt.to_period("M").dt.to_timestamp()
            monthly = cadence.groupby("month").agg(videos=("shortcode", "count"),
                                                   median_metric=(metric, "median")).reset_index()
            st.altair_chart(
                alt.Chart(monthly).mark_bar(color="#dc2626").encode(
                    x=alt.X("month:T", title=None), y=alt.Y("videos:Q", title="videos"),
                    tooltip=["month:T", "videos:Q", "median_metric:Q"],
                ).properties(height=280),
                use_container_width=True,
            )
    with right:
        st.subheader(f"{METRIC_LABELS[metric].title()} over time")
        pts = df.dropna(subset=["taken_at", metric])
        if not pts.empty:
            st.altair_chart(
                alt.Chart(pts).mark_circle(size=90, opacity=0.7, color="#dc2626").encode(
                    x=alt.X("taken_at:T", title=None), y=alt.Y(f"{metric}:Q", title=METRIC_LABELS[metric]),
                    tooltip=["shortcode", "main_topic", "spoken_0_2", metric],
                ).properties(height=280),
                use_container_width=True,
            )
        else:
            st.info("No performance data available for this metric.")

    st.subheader("What correlates with performance")
    st.caption(
        "Spearman rank correlation between raw measurements and "
        f"{METRIC_LABELS[metric]}. Rank-based because view counts are heavily skewed. "
        "Correlation is not causation, and small libraries produce noisy numbers."
    )
    corr = correlations(df, target=metric)
    if corr.empty:
        st.info("Needs at least ~5 videos with the metric present.")
    else:
        st.dataframe(corr, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Topics & Series
# ---------------------------------------------------------------------------
if page == PAGES[8]:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Videos by topic")
        topics = by_dimension(df, "main_topic", metric=metric)
        bar(topics.head(25), "main_topic", "videos")
    with col_b:
        st.subheader(f"Median {METRIC_LABELS[metric]} by topic")
        st.caption("Highest-performing topics. `vs_median_pct` compares against your overall median.")
        if not topics.empty:
            st.dataframe(
                topics[["main_topic", "videos", "median_metric", "mean_metric", "max_metric", "vs_median_pct"]],
                use_container_width=True, hide_index=True,
            )

    st.divider()
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Recurring series")
        series = by_dimension(df, "series", metric=metric)
        if series.empty:
            st.info("No recurring series identified yet.")
        else:
            st.dataframe(series, use_container_width=True, hide_index=True)
    with col_d:
        st.subheader("Story vs advice vs rant vs sales")
        modes = by_dimension(df, "content_mode", metric=metric)
        if modes.empty:
            st.info("Run `analyze` to populate this.")
        else:
            st.dataframe(modes, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("People & archetypes mentioned")
    st.caption(
        'This is the "do my lawyer videos outperform?" table — grouped by the archetypes '
        "the AI found mentioned, with the median performance of each group."
    )
    arche = archetype_performance(df, min_count=1)
    if arche.empty:
        st.info("No archetypes extracted yet.")
    else:
        st.dataframe(arche, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Dating categories")
    cats = by_dimension(df, "dating_category", metric=metric)
    if not cats.empty:
        st.dataframe(cats, use_container_width=True, hide_index=True)

    st.subheader("What to make next")
    st.caption("Follow-up ideas the model proposed, from your highest-performing videos.")
    ideas = []
    top = df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(15)
    for _, row in top.iterrows():
        payload = row.get("classification_json")
        if not payload:
            continue
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            continue
        for idea in data.get("next_video_ideas") or []:
            ideas.append({"idea": idea, "from": row["shortcode"], METRIC_LABELS[metric]: row.get(metric),
                          "topic": row.get("main_topic")})
    if ideas:
        st.dataframe(pd.DataFrame(ideas), use_container_width=True, hide_index=True)
    else:
        st.info("Run `analyze` to generate follow-up ideas.")

# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

if page == PAGES[4]:
    st.subheader("Hook types")
    st.caption(
        "How each opening style performs for you. 'Your best example' is the exact "
        "opening of your highest-viewed video using that hook."
    )
    present = df[df["hook_primary_type"].notna()]
    if present.empty:
        st.info("Run `analyze` to populate this.")
    else:
        rows = []
        for htype, grp in present.groupby("hook_primary_type"):
            top = grp.sort_values(metric, ascending=False).iloc[0] if grp[metric].notna().any() else grp.iloc[0]
            example = top.get("spoken_0_2") or top.get("spoken_first_sentence") or ""
            rows.append(
                {
                    "hook type": str(htype).replace("_", " "),
                    "what it means": HOOK_MEANINGS.get(str(htype), ""),
                    "videos": len(grp),
                    f"median {METRIC_LABELS[metric]}": grp[metric].median(),
                    "your best example": f"“{example}”" if example else "",
                    f"its {METRIC_LABELS[metric]}": top.get(metric),
                }
            )
        hook_table_df = pd.DataFrame(rows).sort_values(
            f"median {METRIC_LABELS[metric]}", ascending=False
        )
        st.dataframe(hook_table_df, use_container_width=True, hide_index=True)

    st.subheader("Hook structures")
    structures = by_dimension(df, "hook_structure", metric=metric)
    if not structures.empty:
        st.dataframe(structures, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Exact opening language")
    st.markdown(
        "The core table of this project: **verbatim** first words next to performance, so you can compare "
        "*“Guys, this is super manipulative”* against *“Guys, this is a big secret”* directly — "
        "not just their shared label."
    )
    hook_table = hook_language_table(df, limit=500)
    show_cols = [c for c in [
        "shortcode", metric, "spoken_0_2", "spoken_0_5", "onscreen_hook", "hooks_differ",
        "hook_primary_type", "hook_structure", "time_to_first_word", "words_in_first_2s",
        "cuts_first_5s", "main_topic",
    ] if c in hook_table.columns]
    st.dataframe(hook_table[show_cols], use_container_width=True, hide_index=True, height=420)

    st.divider()
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Spoken vs on-screen hook")
        st.caption("Does saying one thing while showing another work better for you?")
        if "hooks_differ" in df:
            work = df.dropna(subset=["hooks_differ"]).copy()
            if work.empty:
                st.info("Run `vision` to detect on-screen text.")
            else:
                work["stacked"] = work["hooks_differ"].map({1: "different", 0: "same message"})
                st.dataframe(by_dimension(work, "stacked", metric=metric), use_container_width=True, hide_index=True)
    with col_d:
        st.subheader("Opening pace")
        st.caption("Words in the first 2 seconds, bucketed.")
        work = df.dropna(subset=["words_in_first_2s"]).copy()
        if not work.empty:
            work["opening_density"] = pd.cut(
                work["words_in_first_2s"], bins=[-1, 2, 4, 6, 8, 100],
                labels=["0-2 words", "3-4", "5-6", "7-8", "9+"],
            ).astype(str)
            st.dataframe(by_dimension(work, "opening_density", metric=metric), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Recurring phrases & language patterns")
    phrases = phrase_frequency(df)
    if phrases.empty:
        st.info("Run `analyze` to extract notable phrases.")
    else:
        st.dataframe(phrases, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Visual & Editing
# ---------------------------------------------------------------------------
if page == PAGES[8]:
    st.caption(
        "All raw measurements. Group counts are small at first — treat these as hypotheses "
        "to test with more videos, not conclusions."
    )
    visual_dims = [
        ("posture", "Posture"), ("setting", "Indoors / outdoors"), ("shot_type", "Shot type"),
        ("outfit_formality", "Outfit formality"), ("outfit_category", "Outfit"),
        ("clothing_color", "Clothing colour"), ("background", "Background"),
        ("lighting", "Lighting"), ("camera_angle", "Camera angle"), ("hair", "Hair"),
    ]
    cols = st.columns(2)
    for i, (column, label) in enumerate(visual_dims):
        if column not in df or df[column].dropna().empty:
            continue
        with cols[i % 2]:
            st.subheader(label)
            st.dataframe(by_dimension(df, column, metric=metric), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Editing")
    edit_cols = st.columns(2)
    for i, (column, label) in enumerate(
        [("speaking_pace", "Speaking pace"), ("length_bucket", "Video length")]
    ):
        if column not in df or df[column].dropna().empty:
            continue
        with edit_cols[i % 2]:
            st.markdown(f"**{label}**")
            st.dataframe(by_dimension(df, column, metric=metric), use_container_width=True, hide_index=True)

    st.subheader("Production elements")
    st.caption(
        "How many of your analysed videos use each element. A views comparison only "
        "appears when you have enough videos on BOTH sides — if nearly everything "
        "has subtitles, there's nothing to compare against (that's normal)."
    )
    element_rows = []
    for column, label in [
        ("has_subtitles", "Burned-in subtitles"),
        ("has_text_overlays", "Text overlays (headlines)"),
        ("music_detected", "Background music"),
        ("has_jump_cuts", "Jump cuts"),
        ("has_zooms", "Zooms"),
    ]:
        if column not in df:
            continue
        known = df[df[column].notna()]
        if known.empty:
            continue
        with_it = known[known[column] == 1]
        without = known[known[column] == 0]
        row = {
            "element": label,
            "videos using it": f"{len(with_it)} of {len(known)} ({len(with_it) / len(known) * 100:.0f}%)",
        }
        if len(with_it) >= 3 and len(without) >= 3:
            row[f"median {METRIC_LABELS[metric]} with"] = with_it[metric].median()
            row[f"median {METRIC_LABELS[metric]} without"] = without[metric].median()
        else:
            row[f"median {METRIC_LABELS[metric]} with"] = None
            row[f"median {METRIC_LABELS[metric]} without"] = None
        element_rows.append(row)
    if element_rows:
        st.dataframe(pd.DataFrame(element_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Run `vision` to populate this.")

# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------
if page == PAGES[5]:
    st.subheader(f"Ranked by {METRIC_LABELS[metric]}")
    ranked = df.sort_values(metric, ascending=False, na_position="last")
    show = [c for c in [
        "shortcode", "url", "taken_at_utc", metric, "like_count", "comment_count",
        "duration_sec", "main_topic", "subtopic", "series", "content_mode",
        "hook_primary_type", "spoken_0_2", "summary",
    ] if c in ranked.columns]
    st.dataframe(
        ranked[show], use_container_width=True, hide_index=True, height=600,
        column_config={"url": st.column_config.LinkColumn("url", display_text="open")},
    )
    st.download_button(
        "Download this table as CSV",
        ranked[show].to_csv(index=False).encode("utf-8"),
        file_name="creator_intel_videos.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Transcript search
# ---------------------------------------------------------------------------
if page == PAGES[6]:
    st.subheader("Searchable transcript archive")
    st.caption("Full-text search across verbatim transcripts, captions and on-screen text.")
    query = st.text_input("Search", placeholder='e.g. situationship, "text her back", lawyer')
    if query:
        hits = search_transcripts(query, limit=100)
        if hits.empty:
            st.info("No matches.")
        else:
            merged = hits.merge(
                df_all[[c for c in ["shortcode", "url", "taken_at_utc", "view_count", "main_topic", "spoken_0_2"]
                        if c in df_all.columns]],
                on="shortcode", how="left",
            )
            st.write(f"{len(merged)} match(es)")
            for _, row in merged.iterrows():
                with st.expander(
                    f"{row['shortcode']} — {row.get('main_topic') or 'unlabelled'} — "
                    f"{_fmt(row.get('view_count'))} views"
                ):
                    st.markdown(f"**Transcript:** …{row.get('transcript_hit') or ''}…")
                    if row.get("overlay_hit"):
                        st.markdown(f"**On screen:** {row['overlay_hit']}")
                    if row.get("url"):
                        st.markdown(f"[Open on Instagram]({row['url']})")

    st.divider()
    st.subheader("Semantic search (creator voice)")
    st.caption(
        "Vector search over your own back catalogue. This is the retrieval layer a future "
        "script generator uses: find your closest past videos, then write in that voice."
    )
    idea = st.text_input("Describe an idea", placeholder="e.g. why she went cold after three good dates")
    top_k = st.slider("How many examples", 5, 50, 20)
    if idea:
        try:
            from creator_intel.db.database import connect
            from creator_intel.voice.search import semantic_search

            with connect(read_only=True) as conn:
                matches = semantic_search(conn, idea, k=top_k)
            if not matches:
                st.warning("No embeddings stored yet — run `python -m creator_intel embed`.")
            else:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "similarity": m["similarity"], "shortcode": m["shortcode"],
                                "views": m.get("view_count"), "topic": m.get("main_topic"),
                                "hook": m.get("spoken_hook"), "url": m.get("url"),
                            }
                            for m in matches
                        ]
                    ),
                    use_container_width=True, hide_index=True,
                    column_config={"url": st.column_config.LinkColumn("url", display_text="open")},
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Semantic search failed: {exc}")

# ---------------------------------------------------------------------------
# Reel detail
# ---------------------------------------------------------------------------
if page == PAGES[7]:
    st.subheader("Reel forensics")
    options = df["shortcode"].tolist()
    if not options:
        st.info("No videos match the current filters.")
    else:
        pick = st.selectbox("Reel", options, format_func=lambda sc: f"{sc} — {df[df.shortcode == sc]['main_topic'].iloc[0] or '?'}")
        row = df[df["shortcode"] == pick].iloc[0]

        head = st.columns(4)
        head[0].metric("Views", _fmt(row.get("view_count")))
        head[1].metric("Likes", _fmt(row.get("like_count")))
        head[2].metric("Comments", _fmt(row.get("comment_count")))
        head[3].metric("Duration", f"{_fmt(row.get('duration_sec'), 1)}s")

        video_col, data_col = st.columns([1, 2])
        with video_col:
            path = row.get("local_video_path")
            if path and Path(path).exists():
                st.video(path)
            else:
                st.info("Video file not found locally.")
            if row.get("url"):
                st.markdown(f"[Open on Instagram]({row['url']})")

        with data_col:
            st.markdown("### RAW — what actually happened")
            st.markdown(
                f"""
| | |
|---|---|
| time to first word | `{row.get('time_to_first_word')}` s |
| spoken 0.0–2.0s | **“{row.get('spoken_0_2') or ''}”** |
| spoken 0.0–5.0s | “{row.get('spoken_0_5') or ''}” |
| first sentence | “{row.get('spoken_first_sentence') or ''}” |
| on-screen 0–5s | **“{row.get('onscreen_0_5') or ''}”** |
| hooks differ | `{row.get('hooks_differ')}` (similarity {row.get('hook_text_similarity')}) |
| cuts | {row.get('cut_count')} total, {row.get('cuts_first_5s')} in first 5s |
| avg shot | {row.get('avg_shot_sec')} s |
| pace | {row.get('words_per_minute')} wpm ({row.get('speaking_pace')}) |
| subtitles / overlays / music | {bool(row.get('has_subtitles'))} / {bool(row.get('has_text_overlays'))} / {bool(row.get('music_detected'))} |
| visual | {row.get('posture')}, {row.get('setting')}, {row.get('shot_type')}, {row.get('outfit_formality')} {row.get('outfit_category')} ({row.get('clothing_color')}), bg {row.get('background')} |
"""
            )

            st.markdown("### INTERPRETATION — AI labels")
            st.markdown(
                f"**{row.get('main_topic')}** / {row.get('subtopic')} · "
                f"{row.get('content_mode')} · tone *{row.get('tone')}* · "
                f"hook **{row.get('hook_primary_type')}** ({row.get('hook_structure')})"
            )
            if row.get("summary"):
                st.write(row["summary"])
            if row.get("cta"):
                st.markdown(f"**CTA (verbatim):** “{row['cta']}”")

            payload = row.get("classification_json")
            if payload:
                with st.expander("Full classification JSON"):
                    try:
                        st.json(json.loads(payload))
                    except (json.JSONDecodeError, TypeError):
                        st.code(payload)

        st.divider()
        col_x, col_y = st.columns(2)
        with col_x:
            st.markdown("#### On-screen text detected")
            from creator_intel.db.database import connect

            with connect(read_only=True) as conn:
                overlays = pd.DataFrame(
                    [dict(r) for r in conn.execute(
                        "SELECT t_start, t_end, text, position, role, is_subtitle, frame_count "
                        "FROM raw_overlay_text WHERE shortcode = ? ORDER BY t_start", (pick,)
                    )]
                )
            if overlays.empty:
                st.info("None detected — run `vision`.")
            else:
                st.dataframe(overlays, use_container_width=True, hide_index=True)

        with col_y:
            st.markdown("#### Sampled frames (hook window)")
            with connect(read_only=True) as conn:
                frames = [dict(r) for r in conn.execute(
                    "SELECT t_sec, path FROM raw_frames WHERE shortcode = ? AND phase = 'hook' ORDER BY t_sec",
                    (pick,)
                )]
            existing = [f for f in frames if Path(f["path"]).exists()]
            if existing:
                st.image(
                    [f["path"] for f in existing[:8]],
                    caption=[f"{f['t_sec']}s" for f in existing[:8]],
                    width=110,
                )
            else:
                st.info("No frames extracted — run `forensics`.")

        st.markdown("#### Verbatim transcript")
        st.text_area("transcript", row.get("transcript") or "", height=220, label_visibility="collapsed")
        with st.expander("Timestamped transcript"):
            st.code(row.get("transcript_timestamped") or "(none)")


# ---------------------------------------------------------------------------
# 🎲 Calendar — data-driven content randomizer + funnel calendar
# ---------------------------------------------------------------------------
if page == PAGES[3]:
    import random as _random
    from datetime import date, timedelta

    st.subheader("🎲 The Content Machine")
    st.caption(
        "Every suggestion is pulled from YOUR data: topics weighted by median views, "
        "hooks are your literal top performers to re-skin, CTAs match the funnel stage."
    )

    dv = df[df["view_count"].notna()]
    tstats = dv[dv["main_topic"].notna()].groupby("main_topic").agg(
        n=("shortcode", "size"), med=("view_count", "median")).query("n >= 5")

    # Funnel mapping over the curated taxonomy. TOF = reach engines (views/shares),
    # MOF = trust/playbook (comment-keyword -> AskDomo funnel), BOF = conversion.
    STAGES = {
        "🔥 TOF — reach": {
            "topics": ["Age Gap Game", "Manipulation Tactics", "Ethnicity Rankings",
                       "Simps & Nice Guys", "Roster & Casual"],
            "cta": "Soft only: “Follow for part two.”",
        },
        "🤝 MOF — trust": {
            "topics": ["Make Her Obsessed", "Avoidants & Red Flags", "Kiss & Touch",
                       "Cold Approach", "Texting & Apps", "First Dates", "Japan Dating",
                       "Girl Archetypes"],
            "cta": "Comment keyword → “Comment ‘X’ and I’ll send you the breakdown.”",
        },
        "💰 BOF — convert": {
            "topics": ["SF & Tech", "Domo Lore", "First Dates"],
            "cta": "Offer: client story + “link in bio” / “DM me ‘CALL’.”",
        },
    }
    for cfg in STAGES.values():
        cfg["topics"] = [t for t in cfg["topics"] if t in tstats.index]

    COLORS = ["purple 💜", "red ❤️", "brown 🤎"]
    PROD = ("stand · indoor · music bed · captions · 60–90s · "
            "😄 smile/laugh in the first 10s (+35%) · 🎚 2–3 vocal PUNCHES, natural register · 👃 scrunch approved")

    def _weighted_topic(rng, topics):
        weights = [max(1.0, float(tstats.loc[t, "med"])) ** 0.5 for t in topics]
        return rng.choices(topics, weights=weights, k=1)[0]

    def _top_hooks(topic, k=3):
        seg = dv[dv["main_topic"] == topic].sort_values("view_count", ascending=False)
        out = []
        for _, r in seg.head(k).iterrows():
            hook = (r.get("spoken_0_2") or r.get("spoken_first_sentence") or "").strip()
            if hook:
                out.append((hook[:110], int(r["view_count"])))
        return out

    if st.button("🎲 Spin me a video", type="primary"):
        st.session_state["spin_seed"] = _random.randint(0, 10**9)
    if "spin_seed" in st.session_state:
        rng = _random.Random(st.session_state["spin_seed"])
        stage_name = rng.choice(list(STAGES))
        cfg = STAGES[stage_name]
        topic = _weighted_topic(rng, cfg["topics"])
        color = rng.choice(COLORS)
        st.markdown(f"### {stage_name.split('—')[0].strip()} · **{topic}**")
        st.markdown(f"**Median for this topic:** {_fmt(tstats.loc[topic, 'med'])} views "
                    f"({int(tstats.loc[topic, 'n'])} videos)")
        st.markdown("**Steal one of these hook shapes (your own top 3):**")
        for hook, views in _top_hooks(topic):
            st.markdown(f"- “{hook}” — *{_fmt(views)} views* → keep the sentence shape, swap the subject")
        st.markdown(f"**CTA:** {cfg['cta']}")
        st.markdown(f"**Fit:** {color} &nbsp;·&nbsp; **Production:** {PROD}")
        st.divider()

    st.subheader("📅 Content calendar")
    ccol1, ccol2, ccol3 = st.columns(3)
    weeks = ccol1.slider("Weeks", 1, 4, 2)
    per_week = ccol2.slider("Posts per week", 3, 7, 5)
    if ccol3.button("♻️ Reshuffle calendar"):
        st.session_state["cal_seed"] = _random.randint(0, 10**9)
    rng = _random.Random(st.session_state.get("cal_seed", 4242))

    # 5-post funnel rhythm: 2 reach, 2 trust, 1 convert (classic 40/40/20)
    pattern = ["🔥 TOF — reach", "🤝 MOF — trust", "🔥 TOF — reach", "🤝 MOF — trust", "💰 BOF — convert",
               "🤝 MOF — trust", "🔥 TOF — reach"]
    start = date.today() + timedelta(days=1)
    rows = []
    recent = []
    for w in range(weeks):
        for p in range(per_week):
            stage_name = pattern[p % len(pattern)]
            cfg = STAGES[stage_name]
            pool = [t for t in cfg["topics"] if t not in recent[-3:]] or cfg["topics"]
            topic = _weighted_topic(rng, pool)
            recent.append(topic)
            hooks = _top_hooks(topic, k=8)
            hook = rng.choice(hooks) if hooks else ("", 0)
            day = start + timedelta(days=w * 7 + round(p * 7 / per_week))
            rows.append({
                "date": day.strftime("%a %b %d"),
                "funnel": stage_name,
                "topic": topic,
                "steal this hook shape": f"“{hook[0]}” ({_fmt(hook[1])} views)" if hook[0] else "",
                "cta": cfg["cta"].split(":")[0],
                "fit": rng.choice(COLORS),
            })
    cal_df = pd.DataFrame(rows)
    st.dataframe(cal_df, use_container_width=True, hide_index=True,
                 height=44 * len(cal_df) + 40)
    st.download_button("Download calendar as CSV",
                       cal_df.to_csv(index=False).encode("utf-8"),
                       file_name="domo_content_calendar.csv", mime="text/csv")
    st.caption(
        "Rhythm: 2 reach / 2 trust / 1 convert per 5 posts. Topics are weighted toward "
        "your medians but rotated so you don't post Age Gap five times a week. "
        "Every hook shown is one of your own top performers for that topic — re-skin, don't repeat."
    )


# ---------------------------------------------------------------------------
# 🧬 Blueprint — every measured lever, live from the database
# ---------------------------------------------------------------------------
if page == PAGES[1]:
    from creator_intel.analytics.queries import (
        blueprint_delivery, blueprint_posting, blueprint_script, hook_word_lexicon)

    @st.cache_data(ttl=300)
    def _bp():
        return (blueprint_delivery(), blueprint_script(), blueprint_posting(), hook_word_lexicon())

    st.subheader("🧬 Your Blueprint — every measured lever")
    st.caption(
        "“Typical” = the middle video of that group (the median). All numbers recompute "
        "live from your archive, so this page stays true as new reels sync in."
    )
    deliv, script, posting, lexicon = _bp()

    def factlist(bdf: pd.DataFrame, title: str):
        st.markdown(f"##### {title}")
        for _, r in bdf.sort_values("with", ascending=False).iterrows():
            up = r["with"] >= r["without"]
            arrow = "🟢" if up else "🔴"
            note = f" — *{r['note']}*" if r.get("note") else ""
            st.markdown(
                f"{arrow} **{r['habit']}** — typical **{r['with']:,}** views "
                f"(n={r['n_with']}) vs {r['without']:,} {r['vs']} (n={r['n_without']}){note}"
            )
        st.markdown("")

    left, right = st.columns(2)
    with left:
        factlist(script, "📝 Script architecture")
        factlist(posting, "📅 When & how to post")
    with right:
        factlist(deliv, "🎤 Delivery")

    st.divider()
    st.markdown("##### 🔤 Words in your first 5 seconds (12+ videos each)")
    lc1, lc2 = st.columns(2)
    with lc1:
        st.markdown("**Print money:**")
        st.dataframe(lexicon.head(12), use_container_width=True, hide_index=True)
    with lc2:
        st.markdown("**Bury these:**")
        st.dataframe(lexicon.tail(8).sort_values("typical_views"),
                     use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("##### 🧪 The formula, one line at a time")
    st.markdown("""
| Layer | The move |
|---|---|
| **Topic** | Age Gap · Manipulation Tactics · Ethnicity for reach; playbook advice for trust; SF & Tech + client stories to sell |
| **Script** | Numbered list (2 items > 3) or taboo confession · tactic lands inside 5 seconds · heavy profanity on reach videos |
| **Open** | ≤7 words spoken · different headline on screen · zero cuts in the first 5s · smile or laugh inside 10s |
| **Voice** | Natural register · 2–3 volume punches · 1–2 dramatic full-second pauses before reveals |
| **Camera** | Standing · indoors · solo · long takes · warm colors · captions + music on · 60–90s |
| **Post it** | Fri–Sun afternoon (PT) · ride every 500k+ video with dailies for 48h · sequel every winner |
| **Close** | Reach → “follow for part two” · Trust → comment keyword · Convert → offer/link in bio |
""")
