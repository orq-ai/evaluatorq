"""Backend resolution for dynamic red teaming runtime."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.redteam.backends.base import BackendBundle
from evaluatorq.redteam.exceptions import BackendError, CredentialError
from evaluatorq.redteam.backends.base import NoopMemoryCleanup
from evaluatorq.redteam.backends.openai import (
    OpenAIContextProvider,
    OpenAIErrorMapper,
    OpenAITargetFactory,
)
from evaluatorq.redteam.contracts import LLMConfig, PIPELINE_CONFIG, TargetConfig

if TYPE_CHECKING:
    from openai import AsyncOpenAI


ORQ_DEFAULT_BASE_URL = "https://my.orq.ai"
_ROUTER_SUFFIX = "/v2/router"


def create_async_llm_client() -> "AsyncOpenAI":
    """Create an OpenAI-compatible async client.

    Preference order:
    1. Standard OpenAI env (``OPENAI_API_KEY`` + optional ``OPENAI_BASE_URL``)
    2. ORQ env (``ORQ_API_KEY`` + optional ``ORQ_BASE_URL``, defaults to https://my.orq.ai)

    When using ORQ, the router suffix ``/v2/router`` is appended automatically
    to produce the OpenAI-compatible completions endpoint.
    """
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


_BACKEND_REGISTRY: dict[str, Callable[..., BackendBundle]] = {}


def register_backend(name: str, factory: Callable[..., BackendBundle]) -> None:
    """Register a backend factory for use with resolve_backend()."""
    _BACKEND_REGISTRY[name.strip().lower()] = factory


def resolve_backend(
    backend: str = "orq",
    llm_client: "AsyncOpenAI | None" = None,
    target_config: TargetConfig | None = None,
    pipeline_config: LLMConfig | None = None,
) -> BackendBundle:
    """Resolve runtime backend bundle with lazy optional imports.

    Args:
        backend: Backend name (e.g. ``"orq"`` or ``"openai"``).
        llm_client: Pre-configured client for the OpenAI backend.
            When provided, skips ``create_async_llm_client()`` for
            the ``"openai"`` backend.
        target_config: Optional target configuration (e.g. system prompt).
        pipeline_config: Optional LLMConfig instance. Defaults to module-level PIPELINE_CONFIG.
    """
    normalized = backend.strip().lower()
    factory = _BACKEND_REGISTRY.get(normalized)
    if factory is not None:
        return factory(llm_client=llm_client, target_config=target_config, pipeline_config=pipeline_config)
    raise BackendError(f"Unsupported backend: {backend!r}. Available: {sorted(_BACKEND_REGISTRY)}")


def _create_openai_backend(
    llm_client: "AsyncOpenAI | None" = None,
    target_config: TargetConfig | None = None,
    **kwargs: object,
) -> BackendBundle:
    system_prompt = target_config.system_prompt if target_config else None
    client = llm_client or create_async_llm_client()
    return BackendBundle(
        name="openai",
        target_factory=OpenAITargetFactory(client, system_prompt=system_prompt),
        context_provider=OpenAIContextProvider(system_prompt=system_prompt),
        memory_cleanup=NoopMemoryCleanup(),
        error_mapper=OpenAIErrorMapper(),
    )


def _create_orq_backend(
    llm_client: "AsyncOpenAI | None" = None,
    target_config: TargetConfig | None = None,
    pipeline_config: LLMConfig | None = None,
) -> BackendBundle:
    cfg = pipeline_config or PIPELINE_CONFIG
    try:
        from evaluatorq.redteam.backends.orq import ORQErrorMapper, create_orq_backend
    except ImportError as exc:
        msg = "ORQ backend requested but ORQ dependencies are unavailable."
        raise BackendError(msg) from exc
    target_factory, context_provider, memory_cleanup = create_orq_backend(
        timeout_ms=cfg.target_agent_timeout_ms,
    )
    return BackendBundle(
        name="orq",
        target_factory=target_factory,
        context_provider=context_provider,
        memory_cleanup=memory_cleanup,
        error_mapper=ORQErrorMapper(),
    )


register_backend("openai", _create_openai_backend)
register_backend("orq", _create_orq_backend)
