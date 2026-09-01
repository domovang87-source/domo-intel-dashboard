"""Google Gemini implementation of LLMProvider.

Requires `pip install google-genai` and GEMINI_API_KEY.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from ...config import settings
from ...logging_setup import get_logger
from .base import LLMResponse, ProviderError

log = get_logger(__name__)

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_EMBED_MODEL = "text-embedding-004"


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: Optional[str] = None):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("pip install google-genai to use LLM_PROVIDER=gemini") from exc

        key = api_key or settings.gemini_api_key
        if not key:
            raise ProviderError("GEMINI_API_KEY is not set in .env")
        self._genai = genai
        self._client = genai.Client(api_key=key)

    def _config(self, max_tokens: int, temperature: float, json_mode: bool, system: Optional[str]):
        from google.genai import types

        return types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
            system_instruction=system,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30), reraise=True)
    def _call(self, contents, model, config) -> LLMResponse:
        resp = self._client.models.generate_content(model=model, contents=contents, config=config)
        return LLMResponse(text=(resp.text or "").strip(), model=model, provider=self.name)

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
        model = model or (settings.llm_model if "gemini" in settings.llm_model else DEFAULT_MODEL)
        return self._call([prompt], model, self._config(max_tokens, temperature, json_mode, system))

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
        from google.genai import types

        parts: List = [prompt]
        for path in images:
            path = Path(path)
            parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg"))
        model = model or (settings.vision_model if "gemini" in settings.vision_model else DEFAULT_MODEL)
        return self._call(parts, model, self._config(max_tokens, temperature, json_mode, system))

    def embed(self, texts: Sequence[str], *, model: Optional[str] = None) -> List[List[float]]:
        model = model or (
            settings.embedding_model if "embedding" in settings.embedding_model else DEFAULT_EMBED_MODEL
        )
        resp = self._client.models.embed_content(model=model, contents=list(texts))
        return [list(e.values) for e in resp.embeddings]
