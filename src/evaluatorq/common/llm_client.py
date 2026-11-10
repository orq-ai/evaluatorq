"""Single source of truth for OpenAI-compatible LLM client resolution.

Both the red-team pipeline (``redteam.backends.registry``) and the simulation
stack (``openresponses.client``) resolve an LLM client from the same three env
vars. Historically each had its own copy of the precedence logic, which
drifted (different default host, different docstrings). This module owns the
precedence decision once; the call sites are thin policy wrappers.

Precedence (highest first):
1. An explicitly injected ``config_client`` — used as-is, never owned.
2. ``ORQ_API_KEY`` (or an explicit ``extra_api_key``) — routed through the Orq
   AI Router at ``ORQ_BASE_URL/v3/router`` (default host ``https://my.orq.ai``).
3. ``OPENAI_API_KEY`` — talks to OpenAI (or, when ``honor_openai_base_url`` is
   set, to ``OPENAI_BASE_URL`` for vLLM/OpenRouter/Azure-compatible endpoints).
4. ``ANTHROPIC_API_KEY`` — an ``AsyncAnthropic`` wrapped by the
   ``evaluatorq.anthropic`` adapter, talking to Anthropic directly (the SDK reads
   ``ANTHROPIC_BASE_URL`` itself). Needs the ``evaluatorq[anthropic]`` extra.

``ORQ_API_KEY`` wins when several are set: the Orq router is the default path, and
``ORQ_API_KEY`` is also required for tracing/result-upload, so it is the key most
likely to be present intentionally. ``OPENAI_API_KEY`` beats ``ANTHROPIC_API_KEY``
so existing setups resolve as before; unset the other keys to use Claude directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, cast, overload

from evaluatorq.common.env_config import DEFAULT_ORQ_BASE_URL, env_raw

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicVertex
    from openai import AsyncOpenAI

    from evaluatorq.anthropic.client import AnthropicLLMClient

    LLMClient = AsyncOpenAI | AsyncAnthropic | AsyncAnthropicBedrock | AsyncAnthropicVertex | AnthropicLLMClient
else:
    LLMClient = Any

ORQ_DEFAULT_HOST = DEFAULT_ORQ_BASE_URL
ORQ_ROUTER_SUFFIX = '/v3/router'


def orq_base_url() -> str:
    """The Orq host a run is talking to: ``ORQ_BASE_URL`` or the prod default.

    Recorded on run/report records (sim, red team, evaluatorq) so a saved run
    remembers which deployment — prod / staging / on-prem — actually served it.
    Only meaningful when the run used an Orq agent/response; callers omit it
    otherwise. See ``dashboard.orq_workspace.classify_host`` for prod/staging/
    on-prem detection from this value.
    """
    return env_raw('ORQ_BASE_URL', DEFAULT_ORQ_BASE_URL).rstrip('/')


class ModelClientMismatchError(ValueError):
    """A model id that the client it was sent to cannot serve, such as an OpenAI default model reaching an Anthropic client.

    A run-wide configuration mistake: every call would fail the same way. Judge, classify and signals code re-raises it instead of turning it into a per-item outcome, so the run stops where the mistake is named.
    """


class MissingLLMCredentialsError(ValueError):
    """No ``ORQ_API_KEY``/``OPENAI_API_KEY``/``ANTHROPIC_API_KEY`` and no injected client.

    Subclasses ``ValueError`` so existing simulation call sites that document a
    ``ValueError`` contract keep working; the red-team wrapper re-raises it as a
    ``CredentialError`` to preserve its own domain exception.
    """


class ResolvedClient(NamedTuple):
    """Result of `resolve_llm_client`."""

    client: AsyncOpenAI
    owned: bool
    """``False`` when the client was injected — the caller must not close it."""
    routes_through_orq: bool
    """``True`` when requests hit the Orq router (``…/v3/router``)."""


def is_anthropic_client(obj: object) -> bool:
    """True for an Anthropic SDK adapter (``evaluatorq.anthropic``), detected by its marker.

    Reads a class attribute instead of importing the adapter, so it works when the ``anthropic`` extra is not installed.
    """
    return getattr(obj, 'evaluatorq_provider', None) == 'anthropic'


def _is_anthropic_sdk_client(client: object) -> bool:
    """True for an instance of a class from the ``anthropic`` package, or of a subclass of one, that has ``messages``."""
    if not hasattr(client, 'messages'):
        return False
    return any(cls.__module__ == 'anthropic' or cls.__module__.startswith('anthropic.') for cls in type(client).__mro__)


@overload
def as_llm_client(client: None) -> None: ...


@overload
def as_llm_client(client: Any) -> AsyncOpenAI: ...


def as_llm_client(client: Any) -> AsyncOpenAI | None:
    """Return ``client`` ready for evaluatorq: a raw Anthropic SDK client is wrapped, anything else passes through.

    Every public entry point that accepts an injected client calls this, so the adapter is built in one place. ``None``, an OpenAI client and an already wrapped adapter come back unchanged. A raw ``AsyncAnthropic``, ``AsyncAnthropicBedrock`` or ``AsyncAnthropicVertex`` (or a subclass of one) is recognised by the module name of a class in its MRO and its ``messages`` attribute, so nothing here imports ``anthropic``; the adapter module is imported only when a raw client is seen.
    """
    if client is None or is_anthropic_client(client):
        return client
    if not _is_anthropic_sdk_client(client):
        return client
    from evaluatorq.anthropic.client import wrap_anthropic_client

    # The adapter is duck-typed as an `AsyncOpenAI` everywhere downstream.
    return cast('AsyncOpenAI', wrap_anthropic_client(client))


def client_routes_through_orq(client: AsyncOpenAI | None) -> bool:
    """True when ``client``'s ``base_url`` points at the Orq router.

    Gates router-only request fields (e.g. the ``retry`` ``extra_body``) on the
    *actual* endpoint rather than on env vars, so an injected OpenAI client does
    not receive ORQ-specific fields just because ``ORQ_API_KEY`` happens to be in
    the environment (it is needed for tracing). Host-agnostic: matches any host
    whose path ends in ``/v3/router``.
    """
    base_url = getattr(client, 'base_url', None)
    if base_url is None:
        return False
    return str(base_url).rstrip('/').endswith(ORQ_ROUTER_SUFFIX)


def resolve_results_base_url(
    client: AsyncOpenAI | None = None,
    *,
    default_orq_host: str = DEFAULT_ORQ_BASE_URL,
) -> str:
    """Resolve the Orq host that results should be uploaded to.

    Prefers the host of the client actually used for inference when it routes
    through the Orq router — so a staging/custom Orq endpoint receives both the
    inference traffic and the results — stripping the ``/v3/router`` suffix to
    get the bare host. A non-Orq client (e.g. direct OpenAI) is ignored: uploads
    always go to Orq, never to the inference provider. Falls back to
    ``ORQ_BASE_URL`` / ``default_orq_host`` when no Orq-routed client is given,
    preserving the original env-based behaviour.
    """
    if client is not None and client_routes_through_orq(client):
        host = str(client.base_url).rstrip('/')
        return host[: -len(ORQ_ROUTER_SUFFIX)].rstrip('/')
    return env_raw('ORQ_BASE_URL', default_orq_host).rstrip('/')


def resolve_llm_client(
    config_client: LLMClient | None = None,
    *,
    extra_api_key: str | None = None,
    orq_host: str | None = None,
    default_orq_host: str = DEFAULT_ORQ_BASE_URL,
    honor_openai_base_url: bool = True,
    require_orq: bool = False,
    max_retries: int | None = None,
) -> ResolvedClient:
    """Resolve an LLM client from an injected client or env vars.

    The client is an ``AsyncOpenAI`` for the Orq and OpenAI keys, and an adapter-wrapped ``AsyncAnthropic`` for ``ANTHROPIC_API_KEY``.

    Args:
        config_client: Pre-built client to use as-is (returned not owned).
        extra_api_key: Explicit ORQ key, treated like ``ORQ_API_KEY`` but taking
            precedence over the env var (used by the simulation legacy path).
        orq_host: Explicit Orq host for this client, taking precedence over
            ``ORQ_BASE_URL`` when an Orq key is used.
        default_orq_host: Host used when ``ORQ_BASE_URL`` is unset.
        honor_openai_base_url: When True, the OpenAI branch respects
            ``OPENAI_BASE_URL``; when False it forces the OpenAI SDK default.
        require_orq: When True, the ``OPENAI_API_KEY`` and ``ANTHROPIC_API_KEY`` fallbacks are disabled — the
            client must route through Orq (injected ``config_client``,
            ``extra_api_key``, or ``ORQ_API_KEY``). Used by ORQ-agent targets
            whose ``agent/<key>`` model id only resolves on the Orq router, so an
            ``OPENAI_API_KEY`` left in the env must never silently capture them.
        max_retries: Client-side retry budget (number of *retries*, not attempts)
            for clients built here. The OpenAI and Anthropic SDKs retry 408/409/429/5xx and
            connection errors with ``Retry-After`` support. A caller that passes
            an explicit ``max_retries`` makes this the single client-side retry
            layer and must not add its own wrapper on top; a caller that owns
            retry itself must pass ``0`` so the layers cannot stack (see
            ``OrqResponsesTarget``, which wraps its calls in ``with_retry``).
            Legacy simulation paths that still wrap ``with_retry`` around an
            SDK-default client predate this rule and are tracked to migrate.
            ``None`` keeps the SDK default (2). Ignored for an injected
            ``config_client``, which is used exactly as the caller built it.

    Raises:
        MissingLLMCredentialsError: No injected client and no usable ORQ key
            (and, unless ``require_orq``, no ``OPENAI_API_KEY`` or
            ``ANTHROPIC_API_KEY`` either).
        ImportError: Only ``ANTHROPIC_API_KEY`` is set and the ``anthropic``
            extra is not installed.
    """
    if config_client is not None:
        config_client = as_llm_client(config_client)
        return ResolvedClient(
            client=config_client,
            owned=False,
            routes_through_orq=client_routes_through_orq(config_client),
        )

    from openai import DEFAULT_MAX_RETRIES, AsyncOpenAI

    retries = max_retries if max_retries is not None else DEFAULT_MAX_RETRIES

    orq_api_key = extra_api_key or env_raw('ORQ_API_KEY')
    if orq_api_key:
        host = (orq_host or env_raw('ORQ_BASE_URL', default_orq_host)).rstrip('/')
        router_url = f'{host}{ORQ_ROUTER_SUFFIX}'
        return ResolvedClient(
            client=AsyncOpenAI(api_key=orq_api_key, base_url=router_url, max_retries=retries),
            owned=True,
            routes_through_orq=True,
        )

    if require_orq:
        raise MissingLLMCredentialsError(
            'ORQ routing is required for this target but no Orq credentials were '
            'found. Set ORQ_API_KEY (optionally ORQ_BASE_URL), or pass a pre-built '
            'Orq-routed client. The OPENAI_API_KEY and ANTHROPIC_API_KEY fallbacks are disabled here: '
            'ORQ agent targets must run on the Orq router.'
        )

    openai_api_key = env_raw('OPENAI_API_KEY')
    if openai_api_key:
        base_url = env_raw('OPENAI_BASE_URL') if honor_openai_base_url else None
        client = (
            AsyncOpenAI(api_key=openai_api_key, base_url=base_url, max_retries=retries)
            if base_url
            else AsyncOpenAI(api_key=openai_api_key, max_retries=retries)
        )
        return ResolvedClient(
            client=client,
            owned=True,
            routes_through_orq=client_routes_through_orq(client),
        )

    anthropic_api_key = env_raw('ANTHROPIC_API_KEY')
    if anthropic_api_key:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ImportError(
                'ANTHROPIC_API_KEY is set but the anthropic package is not installed. '
                "Install it with `pip install 'evaluatorq[anthropic]'`."
            ) from exc
        from evaluatorq.anthropic.client import wrap_anthropic_client

        # Single retry layer: the SDK's own, sized like the OpenAI branches above.
        # The adapter is duck-typed as an `AsyncOpenAI` everywhere downstream.
        return ResolvedClient(
            client=cast(
                'AsyncOpenAI', wrap_anthropic_client(AsyncAnthropic(api_key=anthropic_api_key, max_retries=retries))
            ),
            owned=True,
            routes_through_orq=False,
        )

    raise MissingLLMCredentialsError(
        'Missing LLM credentials. Set ORQ_API_KEY (optionally ORQ_BASE_URL), '
        'OPENAI_API_KEY (optionally OPENAI_BASE_URL) or ANTHROPIC_API_KEY, or pass a pre-built client.'
    )


__all__ = [
    'ORQ_DEFAULT_HOST',
    'ORQ_ROUTER_SUFFIX',
    'LLMClient',
    'MissingLLMCredentialsError',
    'ModelClientMismatchError',
    'ResolvedClient',
    'as_llm_client',
    'client_routes_through_orq',
    'is_anthropic_client',
    'resolve_llm_client',
    'resolve_results_base_url',
]
