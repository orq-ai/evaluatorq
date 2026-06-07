"""Backend resolution for dynamic red teaming runtime."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.redteam.exceptions import BackendError, CredentialError

if TYPE_CHECKING:
    from collections.abc import Callable

    from openai import AsyncOpenAI

    from evaluatorq.contracts import AgentContext, AgentTarget
    from evaluatorq.redteam.backends.base import Backend
    from evaluatorq.redteam.contracts import LLMCallConfig, LLMConfig, TargetConfig


# Retained for backward-compatible imports; canonical values live in common.llm_client.
ORQ_DEFAULT_BASE_URL = "https://my.orq.ai"


def create_async_llm_client(role_config: LLMCallConfig | None = None) -> AsyncOpenAI:
    """Create an OpenAI-compatible async client.

    Thin red-team wrapper over :func:`evaluatorq.common.llm_client.resolve_llm_client`
    (the single source of truth for env-var precedence). If ``role_config.client``
    is set it is returned directly; otherwise the client is auto-detected.

    Preference order:
    1. ``role_config.client`` if provided
    2. ORQ env (``ORQ_API_KEY`` + optional ``ORQ_BASE_URL``, defaults to https://my.orq.ai)
    3. Standard OpenAI env (``OPENAI_API_KEY`` + optional ``OPENAI_BASE_URL``)

    When using ORQ, the router suffix ``/v3/router`` is appended automatically
    to produce the OpenAI-compatible completions endpoint.
    """
    from evaluatorq.common.llm_client import MissingLLMCredentialsError, resolve_llm_client

    if os.getenv("ROUTER_BASE_URL") and not os.getenv("ORQ_BASE_URL"):
        logger.warning("ROUTER_BASE_URL is no longer supported; rename it to ORQ_BASE_URL")

    config_client = role_config.client if role_config is not None else None
    try:
        return resolve_llm_client(
            config_client,
            default_orq_host=ORQ_DEFAULT_BASE_URL,
            honor_openai_base_url=True,
        ).client
    except ImportError as exc:
        msg = (
            "openai package is required for LLM-based attack generation. "
            "Install it with: pip install openai"
        )
        raise BackendError(msg) from exc
    except MissingLLMCredentialsError as exc:
        raise CredentialError(str(exc)) from exc


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


def _create_openresponses_backend(
    llm_client: AsyncOpenAI | None = None,
    target_config: TargetConfig | None = None,
    pipeline_config: LLMConfig | None = None,
    **_: object,  # absorbs unknown kwargs from resolve_backend's uniform signature
) -> Backend:
    from evaluatorq.redteam.backends.openresponses import OpenResponsesBackend

    instructions = target_config.system_prompt if target_config else None
    timeout_ms = pipeline_config.target_agent_timeout_ms if pipeline_config else None
    # retry_count is the number of *retries*; OrqResponsesTarget wants total attempts (initial + retries), hence the +1.
    retry_attempts = pipeline_config.retry_count + 1 if pipeline_config else None
    retry_statuses = pipeline_config.retry_on_codes if pipeline_config else None
    return OpenResponsesBackend(
        client=llm_client,
        instructions=instructions,
        timeout_ms=timeout_ms,
        retry_attempts=retry_attempts,
        retry_statuses=retry_statuses,
    )


register_backend("openai", _create_openai_backend)
register_backend("orq", _create_orq_backend)
register_backend("openresponses", _create_openresponses_backend)
