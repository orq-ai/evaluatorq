"""Single source of truth for OpenAI-compatible LLM client resolution.

Both the red-team pipeline (``redteam.backends.registry``) and the simulation
stack (``openresponses.client``) resolve an ``AsyncOpenAI`` client from the same
two env vars. Historically each had its own copy of the precedence logic, which
drifted (different default host, different docstrings). This module owns the
precedence decision once; the call sites are thin policy wrappers.

Precedence (highest first):
1. An explicitly injected ``config_client`` — used as-is, never owned.
2. ``ORQ_API_KEY`` (or an explicit ``extra_api_key``) — routed through the Orq
   AI Router at ``ORQ_BASE_URL/v3/router`` (default host ``https://my.orq.ai``).
3. ``OPENAI_API_KEY`` — talks to OpenAI (or, when ``honor_openai_base_url`` is
   set, to ``OPENAI_BASE_URL`` for vLLM/OpenRouter/Azure-compatible endpoints).

``ORQ_API_KEY`` wins when both are set: the Orq router is the default path, and
``ORQ_API_KEY`` is also required for tracing/result-upload, so it is the key most
likely to be present intentionally.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from openai import AsyncOpenAI

ORQ_DEFAULT_HOST = "https://my.orq.ai"
ORQ_ROUTER_SUFFIX = "/v3/router"


class MissingLLMCredentialsError(ValueError):
    """No ``ORQ_API_KEY``/``OPENAI_API_KEY`` and no injected client.

    Subclasses ``ValueError`` so existing simulation call sites that document a
    ``ValueError`` contract keep working; the red-team wrapper re-raises it as a
    ``CredentialError`` to preserve its own domain exception.
    """


class ResolvedClient(NamedTuple):
    """Result of :func:`resolve_llm_client`."""

    client: AsyncOpenAI
    owned: bool
    """``False`` when the client was injected — the caller must not close it."""
    routes_through_orq: bool
    """``True`` when requests hit the Orq router (``…/v3/router``)."""


def client_routes_through_orq(client: AsyncOpenAI | None) -> bool:
    """True when ``client``'s ``base_url`` points at the Orq router.

    Gates router-only request fields (e.g. the ``retry`` ``extra_body``) on the
    *actual* endpoint rather than on env vars, so an injected OpenAI client does
    not receive ORQ-specific fields just because ``ORQ_API_KEY`` happens to be in
    the environment (it is needed for tracing). Host-agnostic: matches any host
    whose path ends in ``/v3/router``.
    """
    base_url = getattr(client, "base_url", None)
    if base_url is None:
        return False
    return str(base_url).rstrip("/").endswith(ORQ_ROUTER_SUFFIX)


def resolve_llm_client(
    config_client: AsyncOpenAI | None = None,
    *,
    extra_api_key: str | None = None,
    default_orq_host: str = ORQ_DEFAULT_HOST,
    honor_openai_base_url: bool = True,
    require_orq: bool = False,
) -> ResolvedClient:
    """Resolve an ``AsyncOpenAI`` client from an injected client or env vars.

    Args:
        config_client: Pre-built client to use as-is (returned not owned).
        extra_api_key: Explicit ORQ key, treated like ``ORQ_API_KEY`` but taking
            precedence over the env var (used by the simulation legacy path).
        default_orq_host: Host used when ``ORQ_BASE_URL`` is unset.
        honor_openai_base_url: When True, the OpenAI branch respects
            ``OPENAI_BASE_URL``; when False it forces the OpenAI SDK default.
        require_orq: When True, the ``OPENAI_API_KEY`` fallback is disabled — the
            client must route through Orq (injected ``config_client``,
            ``extra_api_key``, or ``ORQ_API_KEY``). Used by ORQ-agent targets
            whose ``agent/<key>`` model id only resolves on the Orq router, so an
            ``OPENAI_API_KEY`` left in the env must never silently capture them.

    Raises:
        MissingLLMCredentialsError: No injected client and no usable ORQ key
            (and, unless ``require_orq``, no ``OPENAI_API_KEY`` either).
    """
    if config_client is not None:
        return ResolvedClient(
            client=config_client,
            owned=False,
            routes_through_orq=client_routes_through_orq(config_client),
        )

    from openai import AsyncOpenAI

    orq_api_key = extra_api_key or os.environ.get("ORQ_API_KEY")
    if orq_api_key:
        host = os.environ.get("ORQ_BASE_URL", default_orq_host).rstrip("/")
        router_url = f"{host}{ORQ_ROUTER_SUFFIX}"
        return ResolvedClient(
            client=AsyncOpenAI(api_key=orq_api_key, base_url=router_url),
            owned=True,
            routes_through_orq=True,
        )

    if require_orq:
        raise MissingLLMCredentialsError(
            "ORQ routing is required for this target but no Orq credentials were "
            "found. Set ORQ_API_KEY (optionally ORQ_BASE_URL), or pass a pre-built "
            "Orq-routed client. The OPENAI_API_KEY fallback is disabled here: "
            "ORQ agent targets must run on the Orq router."
        )

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        base_url = os.environ.get("OPENAI_BASE_URL") if honor_openai_base_url else None
        client = (
            AsyncOpenAI(api_key=openai_api_key, base_url=base_url)
            if base_url
            else AsyncOpenAI(api_key=openai_api_key)
        )
        return ResolvedClient(
            client=client,
            owned=True,
            routes_through_orq=client_routes_through_orq(client),
        )

    raise MissingLLMCredentialsError(
        "Missing LLM credentials. Set either ORQ_API_KEY (optionally ORQ_BASE_URL) "
        "or OPENAI_API_KEY (optionally OPENAI_BASE_URL), or pass a pre-built client."
    )


__all__ = [
    "ORQ_DEFAULT_HOST",
    "ORQ_ROUTER_SUFFIX",
    "MissingLLMCredentialsError",
    "ResolvedClient",
    "client_routes_through_orq",
    "resolve_llm_client",
]
