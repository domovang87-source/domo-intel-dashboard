"""OpenAI implementation of LLMProvider."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ...config import settings
from ...logging_setup import get_logger
from .base import LLMResponse, ProviderError, encode_image

log = get_logger(__name__)


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: Optional[str] = None):
        from openai import OpenAI

        key = api_key or settings.openai_api_key
        if not key:
            raise ProviderError("OPENAI_API_KEY is not set in .env")
        self._client = OpenAI(api_key=key, timeout=180.0, max_retries=0)

    # -- internals ---------------------------------------------------------
    def _chat(self, messages, model: str, max_tokens: int, temperature: float, json_mode: bool):
        kwargs = {"model": model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Newer reasoning-style models renamed max_tokens and reject temperature;
        # try the modern shape first, then fall back for older chat models.
        attempts = [
            {**kwargs, "max_completion_tokens": max_tokens, "temperature": temperature},
            {**kwargs, "max_completion_tokens": max_tokens},
            {**kwargs, "max_tokens": max_tokens, "temperature": temperature},
        ]
        last_exc = None
        for attempt in attempts:
            try:
                return self._client.chat.completions.create(**attempt)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                last_exc = exc
                if "unsupported" in msg or "unrecognized" in msg or "not supported" in msg:
                    continue
                raise
        raise ProviderError(f"OpenAI rejected every parameter shape: {last_exc}")

    @retry(
        stop=stop_after_attempt(5),
        # TPM rate-limit windows reset per minute, so back off far enough to
        # actually clear one: 5s, 10s, 20s, 40s.
        wait=wait_exponential(multiplier=5, min=5, max=90),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call(self, messages, model, max_tokens, temperature, json_mode) -> LLMResponse:
        resp = self._chat(messages, model, max_tokens, temperature, json_mode)
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else None
        return LLMResponse(text=text, model=model, provider=self.name, usage=usage)

    # -- public ------------------------------------------------------------
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
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        return self._call(messages, model or settings.llm_model, max_tokens, temperature, json_mode)

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
        content: List[dict] = [{"type": "text", "text": prompt}]
        for path in images:
            b64, media_type = encode_image(Path(path))
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}", "detail": "high"},
                }
            )
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": content}
        ]
        return self._call(
            messages, model or settings.vision_model, max_tokens, temperature, json_mode
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30), reraise=True)
    def embed(self, texts: Sequence[str], *, model: Optional[str] = None) -> List[List[float]]:
        model = model or settings.embedding_model
        resp = self._client.embeddings.create(model=model, input=list(texts))
        return [item.embedding for item in resp.data]
