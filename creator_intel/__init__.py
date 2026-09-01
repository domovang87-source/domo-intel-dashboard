"""creator-intel — local Instagram creator intelligence system.

Pipeline:
    Instagram ingestion -> local media archive -> transcription
    -> content forensics -> AI analysis -> SQLite -> analytics

Design rule that governs the whole codebase: RAW observations and AI
INTERPRETATIONS are stored in separate tables and never overwrite each other.
"""

__version__ = "0.1.0"
