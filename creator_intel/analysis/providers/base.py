"""Provider-agnostic LLM interface.

Three calls is all the pipeline needs. Implementing them for a new vendor is
enough to swap the whole analysis layer over:

    complete()        text in  -> text out (JSON-mode aware)
    complete_vision() text + images -> text out
    embed()           texts -> vectors
"""
from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Sequence


class ProviderError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    usage: Optional[dict] = None


class LLMProvider(Protocol):
    name: str

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        json_mode: bool = True,
    ) -> LLMResponse:
        ...

    def complete_vision(
        self,
        prompt: str,
        images: Sequence[Path],
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        json_mode: bool = True,
    ) -> LLMResponse:
        ...

    def embed(self, texts: Sequence[str], *, model: Optional[str] = None) -> List[List[float]]:
        ...


# Vision-model image tokens scale with pixel tiles, and some models (notably
# gpt-4o-mini) bill them at a steep multiplier. 512px width keeps big overlay
# text perfectly legible while cutting per-image tokens ~3x. Frames on disk
# stay full resolution — only the wire copy is downscaled.
VISION_MAX_WIDTH = 512


def encode_image(path: Path, max_width: Optional[int] = VISION_MAX_WIDTH) -> tuple:
    """(base64_string, media_type) for an image on disk, downscaled in memory."""
    path = Path(path)
    if max_width:
        try:
            import io

            from PIL import Image

            with Image.open(path) as im:
                if im.width > max_width:
                    height = round(im.height * max_width / im.width)
                    im = im.resize((max_width, height), Image.LANCZOS)
                    buf = io.BytesIO()
                    im.convert("RGB").save(buf, format="JPEG", quality=85)
                    return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"
        except ImportError:
            pass
    data = path.read_bytes()
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return base64.b64encode(data).decode("ascii"), media_type
