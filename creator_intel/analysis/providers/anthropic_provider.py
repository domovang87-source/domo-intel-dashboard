"""Anthropic (Claude) implementation of LLMProvider.

Requires `pip install anthropic` and ANTHROPIC_API_KEY. Claude has no
"JSON mode" flag, so JSON is enforced by assistant prefill: we start the reply
with "{" and re-attach it, which reliably prevents prose preambles.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from ...config import settings
from ...logging_setup import get_logger
from .base import LLMResponse, ProviderError, encode_image

log = get_logger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-5"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("pip install anthropic to use LLM_PROVIDER=anthropic") from exc

        key = api_key or settings.anthropic_api_key
        if not key:
            raise ProviderError("ANTHROPIC_API_KEY is not set in .env")
        self._client = anthropic.Anthropic(api_key=key, timeout=180.0, max_retries=0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30), reraise=True)
    def _call(self, content, model, system, max_tokens, temperature, json_mode) -> LLMResponse:
        messages = [{"role": "user", "content": content}]
        if json_mode:
            messages.append({"role": "assistant", "content": "{"})
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        text = "".join(block.text for block in resp.content if block.type == "text").strip()
        if json_mode and not text.startswith("{"):
            text = "{" + text
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        return LLMResponse(text=text, model=model, provider=self.name, usage=usage)

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
        model = model or (settings.llm_model if "claude" in settings.llm_model else DEFAULT_MODEL)
        return self._call(
            [{"type": "text", "text": prompt}], model, system, max_tokens, temperature, json_mode
        )

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
        content: List[dict] = []
        for path in images:
            b64, media_type = encode_image(Path(path))
            content.append(
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}}
            )
        content.append({"type": "text", "text": prompt})
        model = model or (settings.vision_model if "claude" in settings.vision_model else DEFAULT_MODEL)
        return self._call(content, model, system, max_tokens, temperature, json_mode)

    def embed(self, texts: Sequence[str], *, model: Optional[str] = None) -> List[List[float]]:
        raise ProviderError(
            "Anthropic does not serve an embeddings endpoint. "
            "Keep EMBEDDING_MODEL on OpenAI, or plug in a local sentence-transformers model."
        )
