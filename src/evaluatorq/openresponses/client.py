"""Shared AsyncOpenAI client construction for OpenResponses targets."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def build_simulation_client(
    config_client: AsyncOpenAI | None = None,
    *,
    extra_api_key: str | None = None,
) -> tuple[AsyncOpenAI, bool]:
    """Build AsyncOpenAI client.

    Returns (client, owned) where owned=False means caller must not close it.

    Resolution order:
    1. ``config_client`` — injected client, used as-is (not owned).
    2. ``extra_api_key`` argument, treated as an ORQ key and routed through
       the Orq router.
    3. ``ORQ_API_KEY`` env var — routes through
       ``ORQ_BASE_URL/v2/router`` (default: ``https://api.orq.ai/v2/router``).
    4. ``OPENAI_API_KEY`` env var — uses the OpenAI SDK default base URL so
       traffic goes to OpenAI directly, not to the Orq router.
    """
    from openai import AsyncOpenAI

    if config_client is not None:
        return config_client, False

    orq_key = extra_api_key or os.environ.get("ORQ_API_KEY")
    resolved = orq_key or os.environ.get("OPENAI_API_KEY")

    if not resolved:
        raise ValueError(
            "No API key found. Set ORQ_API_KEY or OPENAI_API_KEY, "
            "or pass a pre-built client."
        )

    base_url: str | None = (
        f"{os.environ.get('ORQ_BASE_URL', 'https://api.orq.ai')}/v2/router"
        if orq_key
        else None
    )

    return AsyncOpenAI(base_url=base_url, api_key=resolved), True


__all__ = ["build_simulation_client"]
