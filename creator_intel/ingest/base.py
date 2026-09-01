"""Ingestion contract.

Anything that can hand us (metadata dict + a local video file) can be dropped
in here later — a TikTok source, a Meta Graph API source, or a folder of files
you exported by hand — without the rest of the pipeline noticing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional, Protocol


@dataclass
class IngestResult:
    total_seen: int = 0
    new_reels: int = 0
    updated: int = 0
    downloaded: int = 0
    skipped_existing: int = 0
    errors: int = 0
    error_detail: list = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_seen": self.total_seen,
            "new_reels": self.new_reels,
            "updated": self.updated,
            "downloaded": self.downloaded,
            "skipped_existing": self.skipped_existing,
            "errors": self.errors,
        }


class IngestionSource(Protocol):
    """A source of creator videos."""

    name: str

    def authenticate(self) -> None:
        """Establish whatever session is required. Idempotent."""

    def iter_items(self, limit: Optional[int] = None) -> Iterator[Any]:
        """Yield platform-native post objects, newest first."""

    def to_record(self, item: Any) -> Dict[str, Any]:
        """Map a platform-native post onto the `reels` table shape."""

    def download(self, item: Any, record: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch media to the local archive; return path fields to merge."""
