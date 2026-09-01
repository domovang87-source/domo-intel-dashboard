"""Central configuration, loaded once from `.env` + environment.

Every module reads settings from here rather than touching os.environ, so
swapping a backend later is a one-line change in .env.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Project root == the directory containing this package's parent.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root, without clobbering real env vars.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _resolve(p: str) -> Path:
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@dataclass(frozen=True)
class Settings:
    # --- paths -------------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: _resolve(_env("DATA_DIR", "data")))
    db_path: Path = field(default_factory=lambda: _resolve(_env("DB_PATH", "data/creator_intel.db")))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO").upper())

    # --- instagram ---------------------------------------------------------
    ig_username: str = field(default_factory=lambda: _env("IG_USERNAME"))
    ig_password: str = field(default_factory=lambda: _env("IG_PASSWORD"))
    ig_session_dir: Path = field(
        default_factory=lambda: _resolve(_env("IG_SESSION_DIR", "data/session"))
    )
    ig_request_delay: float = field(default_factory=lambda: _env_float("IG_REQUEST_DELAY", 2.5))
    ig_max_posts: int = field(default_factory=lambda: _env_int("IG_MAX_POSTS", 0))

    # --- llm ---------------------------------------------------------------
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai").lower())
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-4o-mini"))
    vision_model: str = field(default_factory=lambda: _env("VISION_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))

    # --- transcription -----------------------------------------------------
    transcribe_backend: str = field(
        default_factory=lambda: _env("TRANSCRIBE_BACKEND", "faster_whisper").lower()
    )
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "large-v3"))
    whisper_compute_type: str = field(default_factory=lambda: _env("WHISPER_COMPUTE_TYPE", "int8"))
    whisper_device: str = field(default_factory=lambda: _env("WHISPER_DEVICE", "cpu"))
    whisper_language: Optional[str] = field(default_factory=lambda: _env("WHISPER_LANGUAGE") or None)
    whisper_vad_filter: bool = field(default_factory=lambda: _env_bool("WHISPER_VAD_FILTER", False))

    # --- forensics ---------------------------------------------------------
    hook_window_sec: float = field(default_factory=lambda: _env_float("HOOK_WINDOW_SEC", 5.0))
    hook_frame_interval: float = field(default_factory=lambda: _env_float("HOOK_FRAME_INTERVAL", 0.5))
    body_frame_interval: float = field(default_factory=lambda: _env_float("BODY_FRAME_INTERVAL", 3.0))
    max_vision_frames: int = field(default_factory=lambda: _env_int("MAX_VISION_FRAMES", 16))
    scene_threshold: float = field(default_factory=lambda: _env_float("SCENE_THRESHOLD", 0.30))

    # --- derived paths -----------------------------------------------------
    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / "frames"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def raw_dir(self) -> Path:
        """Untouched instaloader JSON payloads, kept forever as source of truth."""
        return self.data_dir / "raw_metadata"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    def ensure_dirs(self) -> None:
        for p in (
            self.data_dir,
            self.media_dir,
            self.frames_dir,
            self.audio_dir,
            self.raw_dir,
            self.logs_dir,
            self.ig_session_dir,
            self.db_path.parent,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def api_key_for(self, provider: str) -> str:
        return {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.gemini_api_key,
        }.get(provider.lower(), "")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
