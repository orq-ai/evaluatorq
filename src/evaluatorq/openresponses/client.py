"""Shared AsyncOpenAI client construction for OpenResponses targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from evaluatorq.common.llm_client import LLMClient


def build_simulation_client(
    config_client: LLMClient | None = None,
    *,
    extra_api_key: str | None = None,
    require_orq: bool = False,
    max_retries: int | None = 0,
) -> tuple[AsyncOpenAI, bool]:
    """Build AsyncOpenAI client.

    Thin simulation wrapper over
    `evaluatorq.common.llm_client.resolve_llm_client` (the single source of
    truth for env-var precedence). Returns (client, owned) where owned=False means
    the caller must not close it.

    See `evaluatorq.common.llm_client.resolve_llm_client` for provider precedence.
    An ``extra_api_key`` is treated as an ORQ key and routed through the Orq router.

    When ``require_orq`` is True, the ``OPENAI_API_KEY`` and ``ANTHROPIC_API_KEY``
    fallbacks are disabled: the client must route through Orq (used by ORQ-agent
    targets whose ``agent/<key>`` model id only resolves on the Orq router).

    ``max_retries`` feeds the SDK's client-side retry budget; the default ``0``
    keeps ``with_retry`` as the single retry owner for simulation calls. Pass
    ``None`` only for a caller that intentionally owns retries at the SDK layer.
    When ``max_retries=0``, an injected ``config_client`` is cloned with its SDK
    retry budget disabled; the caller's client is not mutated and remains
    unowned.
    """
    from evaluatorq.common.llm_client import resolve_llm_client

    resolved = resolve_llm_client(
        config_client,
        extra_api_key=extra_api_key,
        honor_openai_base_url=False,
        require_orq=require_orq,
        max_retries=max_retries,
    )
    from evaluatorq.common.retry import without_client_retries

    client = without_client_retries(resolved.client) if max_retries == 0 else resolved.client
    return client, resolved.owned


__all__ = ['build_simulation_client']
