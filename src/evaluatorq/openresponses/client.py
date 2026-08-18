"""Shared AsyncOpenAI client construction for OpenResponses targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def build_simulation_client(
    config_client: AsyncOpenAI | None = None,
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

    Resolution order:
    1. ``config_client`` — injected client, used as-is (not owned).
    2. ``extra_api_key`` argument, treated as an ORQ key and routed through
       the Orq router.
    3. ``ORQ_API_KEY`` env var — routes through
       ``ORQ_BASE_URL/v3/router`` (default: ``https://my.orq.ai/v3/router``).
    4. ``OPENAI_API_KEY`` env var — uses the OpenAI SDK default base URL so
       traffic goes to OpenAI directly, not to the Orq router.

    When ``require_orq`` is True, step 4 is disabled: the client must route
    through Orq (used by ORQ-agent targets whose ``agent/<key>`` model id only
    resolves on the Orq router).

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
