-- ===========================================================================
-- creator-intel schema
--
-- ARCHITECTURAL RULE
--   Tables prefixed `raw_` hold OBSERVATIONS: things that measurably happened
--   in the video (exact words spoken between 0.0s and 2.0s, number of scene
--   cuts, pixels of overlay text). They are produced by deterministic tools or
--   by transcription/OCR, and they are never rewritten by a label.
--
--   The table `analysis` holds INTERPRETATIONS: an LLM's opinion about what
--   those observations mean (hook_type, tone, topic). Interpretations can be
--   regenerated, re-prompted and thrown away. Observations cannot.
--
--   Overwrites are never silent: transcripts / analysis / raw_* rows are copied
--   into `*_archive` tables by triggers before an UPDATE or DELETE lands.
-- ===========================================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- 1. REELS  (raw ingestion facts, one row per Instagram Reel)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reels (
    shortcode              TEXT PRIMARY KEY,          -- stable Instagram slug, e.g. "C8xYz..."
    instagram_id           TEXT,                      -- numeric media id
    owner_username         TEXT,
    owner_id               TEXT,
    url                    TEXT,                      -- https://www.instagram.com/reel/<shortcode>/
    typename               TEXT,                      -- GraphVideo / GraphSidecar / ...
    is_video               INTEGER DEFAULT 1,

    taken_at_utc           TEXT,                      -- ISO-8601
    taken_at_ts            INTEGER,                   -- unix seconds (for cheap range queries)

    caption                TEXT,                      -- verbatim caption
    caption_hashtags       TEXT,                      -- JSON array
    caption_mentions       TEXT,                      -- JSON array
    accessibility_caption  TEXT,
    location_name          TEXT,

    -- local archive
    local_video_path       TEXT,
    local_thumbnail_path   TEXT,
    local_audio_path       TEXT,
    raw_metadata_path      TEXT,                      -- untouched instaloader JSON on disk
    video_bytes            INTEGER,
    video_sha1             TEXT,

    -- container facts (filled by the media layer, not by Instagram)
    duration_sec           REAL,
    width                  INTEGER,
    height                 INTEGER,
    fps                    REAL,
    has_audio_stream       INTEGER,

    -- latest known performance snapshot (history lives in metrics_history)
    view_count             INTEGER,
    play_count             INTEGER,
    like_count             INTEGER,
    comment_count          INTEGER,
    -- only available via the Insights CSV import (Instagram's web API does not
    -- expose these): see `import-insights` command
    share_count            INTEGER,
    save_count             INTEGER,
    reach_count            INTEGER,
    follows_count          INTEGER,
    insights_imported_at   TEXT,
    engagement_total       INTEGER,                   -- likes + comments
    engagement_rate        REAL,                      -- (likes+comments)/views

    metrics_json           TEXT,                      -- JSON: any other counters IG exposes
    raw_metadata_json      TEXT,                      -- JSON: full node dict as ingested
    extra_json             TEXT,                      -- JSON: free-form escape hatch

    first_seen_at          TEXT,
    media_downloaded_at    TEXT,
    metadata_updated_at    TEXT,
    updated_at             TEXT
);

CREATE INDEX IF NOT EXISTS idx_reels_taken_at ON reels(taken_at_ts DESC);
CREATE INDEX IF NOT EXISTS idx_reels_owner    ON reels(owner_username);
CREATE INDEX IF NOT EXISTS idx_reels_views    ON reels(view_count DESC);

-- Append-only performance snapshots. Instagram counters move; we keep the
-- whole curve so "how fast did this reel accumulate views" stays answerable.
CREATE TABLE IF NOT EXISTS metrics_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    shortcode      TEXT NOT NULL REFERENCES reels(shortcode) ON DELETE CASCADE,
    captured_at    TEXT NOT NULL,
    view_count     INTEGER,
    play_count     INTEGER,
    like_count     INTEGER,
    comment_count  INTEGER,
    metrics_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_metrics_history_sc ON metrics_history(shortcode, captured_at);


-- ---------------------------------------------------------------------------
-- 2. TRANSCRIPTS  (RAW — verbatim, never cleaned up)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS transcripts (
    shortcode          TEXT PRIMARY KEY REFERENCES reels(shortcode) ON DELETE CASCADE,
    engine             TEXT,        -- faster_whisper | openai
    model              TEXT,
    compute_type       TEXT,
    language           TEXT,
    language_prob      REAL,

    text               TEXT,        -- VERBATIM. Filler words, repeats, profanity intact.
    timestamped_text   TEXT,        -- "[00:00.32 -> 00:02.10] ..." human-readable
    segments_json      TEXT,        -- JSON array of {id,start,end,text,...}
    words_json         TEXT,        -- JSON array of {word,start,end,probability}

    word_count         INTEGER,
    segment_count      INTEGER,
    speech_duration    REAL,        -- sum of segment durations
    audio_duration     REAL,
    wpm                REAL,        -- words / minute of SPEECH (not of video)
    avg_word_conf      REAL,

    transcribed_at     TEXT,
    duration_of_run    REAL
);

CREATE TABLE IF NOT EXISTS transcripts_archive (
    archived_at  TEXT DEFAULT (datetime('now')),
    shortcode    TEXT, engine TEXT, model TEXT, compute_type TEXT,
    language TEXT, language_prob REAL, text TEXT, timestamped_text TEXT,
    segments_json TEXT, words_json TEXT, word_count INTEGER, segment_count INTEGER,
    speech_duration REAL, audio_duration REAL, wpm REAL, avg_word_conf REAL,
    transcribed_at TEXT, duration_of_run REAL
);

CREATE TRIGGER IF NOT EXISTS trg_transcripts_archive_update
BEFORE UPDATE ON transcripts
BEGIN
    INSERT INTO transcripts_archive
        (shortcode, engine, model, compute_type, language, language_prob, text,
         timestamped_text, segments_json, words_json, word_count, segment_count,
         speech_duration, audio_duration, wpm, avg_word_conf, transcribed_at, duration_of_run)
    VALUES
        (OLD.shortcode, OLD.engine, OLD.model, OLD.compute_type, OLD.language,
         OLD.language_prob, OLD.text, OLD.timestamped_text, OLD.segments_json,
         OLD.words_json, OLD.word_count, OLD.segment_count, OLD.speech_duration,
         OLD.audio_duration, OLD.wpm, OLD.avg_word_conf, OLD.transcribed_at, OLD.duration_of_run);
END;


-- ---------------------------------------------------------------------------
-- 3. RAW HOOK  (exact opening language — the single most important table)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_hook (
    shortcode                  TEXT PRIMARY KEY REFERENCES reels(shortcode) ON DELETE CASCADE,

    -- timing
    time_to_first_word         REAL,     -- dead air before the first spoken word
    first_word                 TEXT,
    first_word_start           REAL,

    -- exact spoken language, sliced by wall-clock window
    spoken_first_sentence      TEXT,     -- exact first spoken sentence, verbatim
    spoken_first_sentence_end  REAL,
    spoken_0_2                 TEXT,     -- exact words spoken in seconds 0.0-2.0
    spoken_0_3                 TEXT,
    spoken_0_5                 TEXT,     -- exact words spoken in seconds 0.0-5.0
    spoken_0_10                TEXT,
    spoken_first_15_words      TEXT,
    spoken_hook                TEXT,     -- best-effort hook span (first sentence, or 0-5s)
    spoken_hook_end            REAL,
    spoken_hook_word_count     INTEGER,

    -- exact on-screen language over the same windows (from OCR / vision)
    onscreen_0_2               TEXT,
    onscreen_0_3               TEXT,
    onscreen_0_5               TEXT,
    onscreen_hook              TEXT,     -- first distinct overlay text seen
    onscreen_hook_first_seen   REAL,
    onscreen_all               TEXT,     -- every distinct overlay string, newline-joined

    -- cheap deterministic comparisons (still raw: string maths, not opinion)
    hooks_differ               INTEGER,  -- 1 if spoken and on-screen hooks are not near-identical
    hook_text_similarity       REAL,     -- 0..1 token overlap between spoken and on-screen hook
    onscreen_hook_count        INTEGER,  -- distinct overlay strings within the hook window
    words_in_first_2s          INTEGER,
    words_in_first_5s          INTEGER,
    wps_first_5s               REAL,     -- opening speaking speed, words/second

    computed_at                TEXT
);


-- ---------------------------------------------------------------------------
-- 4. RAW FRAMES + OVERLAY TEXT  (visual source material, preserved on disk)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_frames (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    shortcode   TEXT NOT NULL REFERENCES reels(shortcode) ON DELETE CASCADE,
    t_sec       REAL NOT NULL,
    phase       TEXT,          -- 'hook' (0..HOOK_WINDOW_SEC) | 'body'
    path        TEXT,          -- kept on disk so vision analysis can be rerun later
    width       INTEGER,
    height      INTEGER,
    extracted_at TEXT,
    UNIQUE(shortcode, t_sec)
);
CREATE INDEX IF NOT EXISTS idx_frames_sc ON raw_frames(shortcode, t_sec);

CREATE TABLE IF NOT EXISTS raw_overlay_text (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    shortcode     TEXT NOT NULL REFERENCES reels(shortcode) ON DELETE CASCADE,
    text          TEXT NOT NULL,       -- VERBATIM as it appears on screen
    t_start       REAL,                -- first frame it was seen in
    t_end         REAL,                -- last frame it was seen in
    duration_sec  REAL,                -- t_end - t_start (approximate, sampling-limited)
    position      TEXT,                -- top | upper-third | middle | lower-third | bottom
    role          TEXT,                -- overlay | caption_subtitle | ui | watermark | unknown
    is_subtitle   INTEGER DEFAULT 0,
    frame_count   INTEGER,             -- how many sampled frames contained it
    source        TEXT,                -- vision:<model> | tesseract
    confidence    REAL,
    frames_json   TEXT,                -- JSON array of timestamps it appeared at
    created_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_overlay_sc ON raw_overlay_text(shortcode, t_start);


-- ---------------------------------------------------------------------------
-- 5. RAW VISUAL  (structured scene labels — model-derived, per frame + rollup)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_visual_frames (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    shortcode    TEXT NOT NULL REFERENCES reels(shortcode) ON DELETE CASCADE,
    t_sec        REAL,
    frame_path   TEXT,
    labels_json  TEXT,         -- JSON: the vision model's per-frame answer
    model        TEXT,
    created_at   TEXT,
    UNIQUE(shortcode, t_sec, model)
);

CREATE TABLE IF NOT EXISTS raw_visual (
    shortcode          TEXT PRIMARY KEY REFERENCES reels(shortcode) ON DELETE CASCADE,
    posture            TEXT,   -- sitting | standing | walking | lying | mixed | unknown
    setting            TEXT,   -- indoors | outdoors | vehicle | unknown
    shot_type          TEXT,   -- close-up | medium | full-body | mixed | unknown
    outfit_category    TEXT,   -- t-shirt | hoodie | button-up | suit | tank | dress | unknown
    clothing_color     TEXT,
    outfit_formality   TEXT,   -- casual | professional | going-out | athletic | unknown
    hair               TEXT,   -- up | down | covered | bald | unknown
    glasses            INTEGER,
    background         TEXT,   -- bedroom | office | car | gym | street | studio | ...
    camera_angle       TEXT,   -- eye-level | low | high | dutch | unknown
    camera_motion      TEXT,   -- static | handheld | walking | unknown
    lighting           TEXT,   -- natural | ring-light | dim | harsh | mixed | unknown
    other_person       INTEGER,
    person_count       INTEGER,
    frames_analyzed    INTEGER,
    labels_json        TEXT,   -- JSON: full rollup incl. per-field agreement
    model              TEXT,
    created_at         TEXT
);


-- ---------------------------------------------------------------------------
-- 6. RAW EDITING  (deterministic ffmpeg/audio measurements)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_editing (
    shortcode              TEXT PRIMARY KEY REFERENCES reels(shortcode) ON DELETE CASCADE,
    duration_sec           REAL,
    dead_air_before_speech REAL,
    cut_count              INTEGER,
    cuts_first_5s          INTEGER,
    cuts_per_minute        REAL,
    avg_shot_sec           REAL,
    median_shot_sec        REAL,
    min_shot_sec           REAL,
    scene_changes_json     TEXT,   -- JSON array of cut timestamps
    jump_cut_count         INTEGER,-- cuts shorter than 1.2s apart (rapid-fire editing)
    has_jump_cuts          INTEGER,
    zoom_events            INTEGER,-- detected framing-scale changes
    has_zooms              INTEGER,

    silence_ratio          REAL,   -- fraction of the video with no speech
    speech_ratio           REAL,
    words_per_minute       REAL,   -- over TOTAL video duration
    speaking_pace          TEXT,   -- slow | measured | fast | rapid  (bucketed from wpm)
    longest_pause_sec      REAL,

    has_subtitles          INTEGER,-- burned-in subtitle track detected via OCR cadence
    subtitle_frame_ratio   REAL,
    has_text_overlays      INTEGER,
    text_overlay_count     INTEGER,
    music_detected         INTEGER,-- non-speech audio energy present under/around speech
    music_confidence       REAL,

    audio_json             TEXT,   -- JSON: silence spans, loudness stats
    created_at             TEXT
);


-- ---------------------------------------------------------------------------
-- 7. ANALYSIS  (INTERPRETATION — regenerable LLM labels)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS analysis (
    shortcode            TEXT PRIMARY KEY REFERENCES reels(shortcode) ON DELETE CASCADE,
    provider             TEXT,
    model                TEXT,
    prompt_version       TEXT,

    -- flattened for fast grouping in the dashboard; full payload in classification_json
    main_topic           TEXT,
    subtopic             TEXT,
    dating_category      TEXT,
    content_format       TEXT,
    content_mode         TEXT,     -- story | advice | rant | sales | reaction | ...
    series               TEXT,     -- recurring series/category
    tone                 TEXT,
    cta                  TEXT,
    cta_type             TEXT,
    summary              TEXT,
    opening_sentence     TEXT,     -- copied from raw_hook so the label sits next to the evidence
    hook_primary_type    TEXT,
    hook_structure       TEXT,
    length_bucket        TEXT,     -- <15s | 15-30s | 30-60s | 60-90s | 90s+
    target_audience      TEXT,
    confidence           REAL,

    hook_types_json      TEXT,     -- JSON array
    hook_evidence_json   TEXT,     -- JSON: exact quoted spans backing each hook_type
    topics_json          TEXT,
    archetypes_json      TEXT,     -- people / archetypes mentioned
    key_claims_json      TEXT,
    notable_phrases_json TEXT,     -- language patterns worth reusing
    entities_json        TEXT,
    classification_json  TEXT,     -- JSON: the entire validated model output
    created_at           TEXT
);

CREATE TABLE IF NOT EXISTS analysis_archive (
    archived_at TEXT DEFAULT (datetime('now')),
    shortcode TEXT, provider TEXT, model TEXT, prompt_version TEXT,
    classification_json TEXT, created_at TEXT
);

CREATE TRIGGER IF NOT EXISTS trg_analysis_archive_update
BEFORE UPDATE ON analysis
BEGIN
    INSERT INTO analysis_archive (shortcode, provider, model, prompt_version, classification_json, created_at)
    VALUES (OLD.shortcode, OLD.provider, OLD.model, OLD.prompt_version, OLD.classification_json, OLD.created_at);
END;

CREATE INDEX IF NOT EXISTS idx_analysis_topic  ON analysis(main_topic);
CREATE INDEX IF NOT EXISTS idx_analysis_series ON analysis(series);
CREATE INDEX IF NOT EXISTS idx_analysis_hook   ON analysis(hook_primary_type);


-- ---------------------------------------------------------------------------
-- 8. CREATOR VOICE  (embeddings for "find the 20 videos most like this idea")
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS embeddings (
    shortcode        TEXT NOT NULL REFERENCES reels(shortcode) ON DELETE CASCADE,
    kind             TEXT NOT NULL DEFAULT 'transcript',  -- transcript | hook | caption | summary
    model            TEXT NOT NULL,
    dim              INTEGER,
    vector           BLOB,          -- float32 little-endian
    source_text      TEXT,
    source_hash      TEXT,          -- lets us skip re-embedding unchanged text
    created_at       TEXT,
    PRIMARY KEY (shortcode, kind, model)
);


-- ---------------------------------------------------------------------------
-- 9. PIPELINE BOOKKEEPING  (resumability + honest failure records)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS job_state (
    shortcode    TEXT NOT NULL,
    stage        TEXT NOT NULL,   -- ingest | media | transcribe | forensics | vision | analyze | embed
    status       TEXT NOT NULL,   -- pending | running | done | error | skipped
    attempts     INTEGER DEFAULT 0,
    error        TEXT,
    detail_json  TEXT,
    started_at   TEXT,
    updated_at   TEXT,
    PRIMARY KEY (shortcode, stage)
);
CREATE INDEX IF NOT EXISTS idx_job_state_stage ON job_state(stage, status);

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    command     TEXT,
    started_at  TEXT,
    ended_at    TEXT,
    ok          INTEGER,
    stats_json  TEXT,
    error       TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key    TEXT PRIMARY KEY,
    value  TEXT
);


-- ---------------------------------------------------------------------------
-- 10. FULL-TEXT SEARCH over verbatim transcripts + captions
-- ---------------------------------------------------------------------------
CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts USING fts5(
    shortcode UNINDEXED,
    text,
    caption,
    overlay_text,
    tokenize = 'porter unicode61'
);


-- ---------------------------------------------------------------------------
-- 11. CONVENIENCE VIEW  (everything joined, for the dashboard / ad-hoc SQL)
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS v_reel_full;
CREATE VIEW v_reel_full AS
SELECT
    r.shortcode, r.url, r.owner_username, r.taken_at_utc, r.taken_at_ts,
    r.caption, r.local_video_path, r.duration_sec, r.width, r.height,
    r.view_count, r.play_count, r.like_count, r.comment_count,
    r.share_count, r.save_count, r.reach_count, r.follows_count,
    r.engagement_total, r.engagement_rate,

    t.text                AS transcript,
    t.timestamped_text    AS transcript_timestamped,
    t.word_count          AS transcript_word_count,
    t.wpm                 AS speech_wpm,
    t.language            AS transcript_language,

    h.time_to_first_word, h.spoken_first_sentence, h.spoken_0_2, h.spoken_0_3,
    h.spoken_0_5, h.spoken_hook, h.onscreen_0_5, h.onscreen_hook, h.onscreen_all,
    h.hooks_differ, h.hook_text_similarity, h.onscreen_hook_count,
    h.words_in_first_2s, h.words_in_first_5s, h.wps_first_5s,

    e.cut_count, e.cuts_first_5s, e.avg_shot_sec, e.dead_air_before_speech,
    e.has_subtitles, e.has_text_overlays, e.text_overlay_count, e.music_detected,
    e.words_per_minute, e.speaking_pace, e.has_jump_cuts, e.has_zooms, e.zoom_events,

    v.posture, v.setting, v.shot_type, v.outfit_category, v.clothing_color,
    v.outfit_formality, v.hair, v.glasses, v.background, v.camera_angle,
    v.lighting, v.other_person, v.person_count,

    a.main_topic, a.subtopic, a.dating_category, a.content_format, a.content_mode,
    a.series, a.tone, a.cta, a.cta_type, a.summary, a.hook_primary_type,
    a.hook_structure, a.length_bucket, a.target_audience,
    a.hook_types_json, a.archetypes_json, a.key_claims_json, a.notable_phrases_json,
    a.topics_json, a.classification_json,

    (t.shortcode IS NOT NULL) AS has_transcript,
    (a.shortcode IS NOT NULL) AS has_analysis,
    (v.shortcode IS NOT NULL) AS has_visual
FROM reels r
LEFT JOIN transcripts  t ON t.shortcode = r.shortcode
LEFT JOIN raw_hook     h ON h.shortcode = r.shortcode
LEFT JOIN raw_editing  e ON e.shortcode = r.shortcode
LEFT JOIN raw_visual   v ON v.shortcode = r.shortcode
LEFT JOIN analysis     a ON a.shortcode = r.shortcode;
