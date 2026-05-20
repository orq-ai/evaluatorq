"""Backend resolution for dynamic red teaming runtime."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.redteam.backends.base import BackendBundle, DefaultErrorMapper, NoopMemoryCleanup
from evaluatorq.redteam.backends.openai import (
    OpenAIContextProvider,
    OpenAIErrorMapper,
    OpenAITargetFactory,
)
from evaluatorq.redteam.contracts import AgentContext
from evaluatorq.redteam.exceptions import BackendError, CredentialError

if TYPE_CHECKING:
    from collections.abc import Callable

    from openai import AsyncOpenAI

    from evaluatorq.redteam.backends.base import AgentTarget
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


_BACKEND_REGISTRY: dict[str, Callable[..., BackendBundle]] = {}


def register_backend(name: str, factory: Callable[..., BackendBundle]) -> None:
    """Register a backend factory for use with resolve_backend()."""
    _BACKEND_REGISTRY[name.strip().lower()] = factory


def resolve_backend(
    backend: str = "orq",
    llm_client: AsyncOpenAI | None = None,
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
    llm_client: AsyncOpenAI | None = None,
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
    llm_client: AsyncOpenAI | None = None,
    target_config: TargetConfig | None = None,
    pipeline_config: LLMConfig | None = None,
    **kwargs: object,
) -> BackendBundle:
    try:
        from evaluatorq.redteam.backends.orq import ORQErrorMapper, create_orq_backend
    except ImportError as exc:
        msg = "ORQ backend requested but ORQ dependencies are unavailable."
        raise BackendError(msg) from exc
    target_factory, context_provider, memory_cleanup = create_orq_backend(
        timeout_ms=pipeline_config.target_agent_timeout_ms if pipeline_config else None,
    )
    return BackendBundle(
        name="orq",
        target_factory=target_factory,
        context_provider=context_provider,
        memory_cleanup=memory_cleanup,
        error_mapper=ORQErrorMapper(),
    )


def _create_openresponses_backend(
    llm_client: AsyncOpenAI | None = None,
    target_config: TargetConfig | None = None,
    pipeline_config: LLMConfig | None = None,
    **kwargs: object,
) -> BackendBundle:
    instructions = target_config.system_prompt if target_config else None
    timeout_ms = pipeline_config.target_agent_timeout_ms if pipeline_config else None
    retry_attempts = pipeline_config.retry_count + 1 if pipeline_config else None
    retry_statuses = pipeline_config.retry_on_codes if pipeline_config else None
    return BackendBundle(
        name="openresponses",
        target_factory=_OrqResponsesTargetFactory(
            client=llm_client,
            instructions=instructions,
            timeout_ms=timeout_ms,
            retry_attempts=retry_attempts,
            retry_statuses=retry_statuses,
        ),
        context_provider=_OpenResponsesContextProvider(instructions=instructions),
        memory_cleanup=NoopMemoryCleanup(),
        error_mapper=_OpenResponsesErrorMapper(),
    )


class _OrqResponsesTargetFactory:
    """Redteam backend factory backed by the shared simulation Responses target."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        instructions: str | None = None,
        timeout_ms: int | None = None,
        retry_attempts: int | None = None,
        retry_statuses: list[int] | None = None,
    ) -> None:
        self._client = client
        self._instructions = instructions
        self._timeout_ms = timeout_ms
        self._retry_attempts = retry_attempts
        self._retry_statuses = retry_statuses

    def create_target(self, agent_key: str) -> AgentTarget:
        from evaluatorq.contracts import LLMCallConfig
        from evaluatorq.simulation.target import OrqResponsesTarget

        config = LLMCallConfig(
            model=agent_key,
            api="responses",
            timeout_ms=self._timeout_ms or 240_000,
            client=self._client,
        )
        return OrqResponsesTarget(
            config,
            instructions=self._instructions,
            client=self._client,
            retry_attempts=self._retry_attempts,
            retry_statuses=self._retry_statuses,
        )


class _OpenResponsesContextProvider:
    def __init__(self, instructions: str | None = None) -> None:
        self._instructions = instructions

    async def get_agent_context(self, agent_key: str) -> AgentContext:
        return AgentContext(
            key=agent_key,
            display_name=agent_key,
            description="OpenResponses agent target",
            system_prompt=self._instructions,
            model=agent_key,
        )


class _OpenResponsesErrorMapper(DefaultErrorMapper):
    def map_error(self, exc: Exception) -> tuple[str, str]:
        from evaluatorq.redteam.backends.base import extract_provider_error_code, extract_status_code

        status_code = extract_status_code(exc)
        if status_code is not None:
            return f"openresponses.http.{status_code}", f"{type(exc).__name__}: {exc}"
        provider_code = extract_provider_error_code(exc)
        if provider_code:
            return f"openresponses.code.{provider_code}", f"{type(exc).__name__}: {exc}"
        name = type(exc).__name__.lower()
        if "ratelimit" in name:
            return "openresponses.rate_limit", f"{type(exc).__name__}: {exc}"
        if "timeout" in name:
            return "openresponses.timeout", f"{type(exc).__name__}: {exc}"
        if "authentication" in name:
            return "openresponses.auth", f"{type(exc).__name__}: {exc}"
        return "openresponses.unknown", f"{type(exc).__name__}: {exc}"


register_backend("openai", _create_openai_backend)
register_backend("orq", _create_orq_backend)
register_backend("openresponses", _create_openresponses_backend)
