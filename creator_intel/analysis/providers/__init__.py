"""Provider registry. Swap vendors by changing LLM_PROVIDER in .env."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from ...config import settings
from .base import LLMProvider, LLMResponse, ProviderError  # noqa: F401

_BUILDERS = {}


def _register():
    if _BUILDERS:
        return
    from .openai_provider import OpenAIProvider

    _BUILDERS["openai"] = OpenAIProvider

    try:
        from .anthropic_provider import AnthropicProvider

        _BUILDERS["anthropic"] = AnthropicProvider
    except Exception:  # optional dependency
        pass
    try:
        from .gemini_provider import GeminiProvider

        _BUILDERS["gemini"] = GeminiProvider
    except Exception:  # optional dependency
        pass


@lru_cache(maxsize=4)
def get_provider(name: Optional[str] = None) -> LLMProvider:
    _register()
    key = (name or settings.llm_provider).lower()
    if key not in _BUILDERS:
        raise ProviderError(
            f"unknown/unavailable LLM provider {key!r}. Available: {sorted(_BUILDERS)}"
        )
    return _BUILDERS[key]()


def available_providers():
    _register()
    return sorted(_BUILDERS)
