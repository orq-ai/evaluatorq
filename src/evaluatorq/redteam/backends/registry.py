"""Backend resolution for dynamic red teaming runtime."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.redteam.exceptions import BackendError, CredentialError

if TYPE_CHECKING:
    from collections.abc import Callable

    from openai import AsyncOpenAI

    from evaluatorq.redteam.backends.base import Backend
    from evaluatorq.redteam.contracts import LLMCallConfig, LLMConfig, TargetConfig


ORQ_DEFAULT_BASE_URL = "https://my.orq.ai"
_ROUTER_SUFFIX = "/v2/router"


def create_async_llm_client(role_config: LLMCallConfig | None = None) -> AsyncOpenAI:
    """Create an OpenAI-compatible async client.

    If role_config.client is set, returns it directly.
    Otherwise auto-detects from environment variables.

    Preference order:
    1. ``role_config.client`` if provided
    2. Standard OpenAI env (``OPENAI_API_KEY`` + optional ``OPENAI_BASE_URL``)
    3. ORQ env (``ORQ_API_KEY`` + optional ``ORQ_BASE_URL``, defaults to https://my.orq.ai)

    When using ORQ, the router suffix ``/v2/router`` is appended automatically
    to produce the OpenAI-compatible completions endpoint.
    """
    if role_config is not None and role_config.client is not None:
        return role_config.client

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        msg = (
            "openai package is required for LLM-based attack generation. "
            "Install it with: pip install openai"
        )
        raise BackendError(msg) from exc

    if os.getenv("ROUTER_BASE_URL") and not os.getenv("ORQ_BASE_URL"):
        logger.warning("ROUTER_BASE_URL is no longer supported; rename it to ORQ_BASE_URL")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        if openai_base_url:
            return AsyncOpenAI(api_key=openai_api_key, base_url=openai_base_url)
        return AsyncOpenAI(api_key=openai_api_key)

    orq_api_key = os.getenv("ORQ_API_KEY")
    if orq_api_key:
        base_url = os.getenv("ORQ_BASE_URL", ORQ_DEFAULT_BASE_URL).rstrip("/")
        router_url = f"{base_url}{_ROUTER_SUFFIX}"
        return AsyncOpenAI(api_key=orq_api_key, base_url=router_url)

    msg = (
        "Missing LLM credentials. Set either OPENAI_API_KEY (optionally OPENAI_BASE_URL) "
        "or ORQ_API_KEY (optionally ORQ_BASE_URL)."
    )
    raise CredentialError(msg)


_BACKEND_REGISTRY: dict[str, Callable[..., Backend]] = {}


def register_backend(name: str, factory: Callable[..., Backend]) -> None:
    """Register a backend factory for use with resolve_backend()."""
    _BACKEND_REGISTRY[name.strip().lower()] = factory


def resolve_backend(
    backend: str = "orq",
    *,
    llm_client: AsyncOpenAI | None = None,
    target_config: TargetConfig | None = None,
    pipeline_config: LLMConfig | None = None,
) -> Backend:
    """Resolve a backend by name."""
    normalized = backend.strip().lower()
    factory = _BACKEND_REGISTRY.get(normalized)
    if factory is None:
        raise BackendError(
            f"Unsupported backend: {backend!r}. Available: {sorted(_BACKEND_REGISTRY)}"
        )
    return factory(llm_client=llm_client, target_config=target_config, pipeline_config=pipeline_config)


def _create_openai_backend(
    llm_client: AsyncOpenAI | None = None,
    target_config: TargetConfig | None = None,
    **_: object,
) -> Backend:
    from evaluatorq.redteam.backends.openai import OpenAIBackend

    system_prompt = target_config.system_prompt if target_config else None
    return OpenAIBackend(client=llm_client, system_prompt=system_prompt)


def _create_orq_backend(
    llm_client: AsyncOpenAI | None = None,
    target_config: TargetConfig | None = None,
    pipeline_config: LLMConfig | None = None,
    **_: object,
) -> Backend:
    try:
        from evaluatorq.redteam.backends.orq import ORQBackend
    except ImportError as exc:
        raise BackendError("ORQ backend requested but ORQ dependencies are unavailable.") from exc

    timeout_ms = pipeline_config.target_agent_timeout_ms if pipeline_config else None
    return ORQBackend(timeout_ms=timeout_ms)


register_backend("openai", _create_openai_backend)
register_backend("orq", _create_orq_backend)
