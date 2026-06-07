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
) -> tuple[AsyncOpenAI, bool]:
    """Build AsyncOpenAI client.

    Thin simulation wrapper over
    :func:`evaluatorq.common.llm_client.resolve_llm_client` (the single source of
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
    """
    from evaluatorq.common.llm_client import resolve_llm_client

    resolved = resolve_llm_client(
        config_client,
        extra_api_key=extra_api_key,
        honor_openai_base_url=False,
        require_orq=require_orq,
    )
    return resolved.client, resolved.owned


__all__ = ["build_simulation_client"]
