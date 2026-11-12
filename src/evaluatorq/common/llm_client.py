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

import contextlib
import weakref
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast, overload

from evaluatorq.common.env_config import DEFAULT_ORQ_BASE_URL, env_raw

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicVertex
    from openai import AsyncOpenAI

    from evaluatorq.anthropic.client import AnthropicLLMClient

    LLMClient = AsyncOpenAI | AsyncAnthropic | AsyncAnthropicBedrock | AsyncAnthropicVertex | AnthropicLLMClient
else:
    LLMClient = Any

ORQ_DEFAULT_HOST = DEFAULT_ORQ_BASE_URL
ORQ_ROUTER_SUFFIX = '/v3/router'

LLM_CREDENTIAL_KEYS: tuple[str, ...] = ('ORQ_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY')
"""The environment variables that can supply an LLM client, in the order `resolve_llm_client` tries them."""

LLMProvider = Literal['orq', 'openai', 'anthropic']


def orq_base_url() -> str:
    """The Orq host a run is talking to: ``ORQ_BASE_URL`` or the prod default.

    Recorded on run/report records (sim, red team, evaluatorq) so a saved run
    remembers which deployment — prod / staging / on-prem — actually served it.
    Only meaningful when the run used an Orq agent/response; callers omit it
    otherwise. See ``dashboard.orq_workspace.classify_host`` for prod/staging/
    on-prem detection from this value.
    """
    return env_raw('ORQ_BASE_URL', DEFAULT_ORQ_BASE_URL).rstrip('/')


class LLMConfigurationError(ValueError):
    """A client, model or request setting that no call can succeed with, raised before any request is sent.

    A run-wide configuration mistake: every call would fail the same way. Judge, classify and signals code re-raises it instead of turning it into a per-item judge error. A per-item handler further up (a red-team attack, a simulation row) still records it on that row and the batch carries on until RES-1879 lands, so the error appears once per row rather than stopping the run.
    """


class ModelClientMismatchError(LLMConfigurationError):
    """A model id that the client it was sent to cannot serve, such as an OpenAI default model reaching an Anthropic client.

    Judge, classify and signals code re-raises it like every `LLMConfigurationError`; a per-item handler further up still records it on each row until RES-1879 lands.
    """


class MissingLLMCredentialsError(ValueError):
    """No usable LLM credentials: none of `LLM_CREDENTIAL_KEYS` is set and no client was injected, or the package that serves the one that is set is not installed.

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


def client_provider(obj: object) -> str | None:
    """The provider marker of an evaluatorq adapter (``'anthropic'`` for `AnthropicLLMClient`), else ``None``."""
    provider = getattr(obj, 'evaluatorq_provider', None)
    return provider if isinstance(provider, str) else None


def is_anthropic_client(obj: object) -> bool:
    """True for an Anthropic SDK adapter (``evaluatorq.anthropic``) or a raw Anthropic SDK client that has not been wrapped yet.

    Reads a class attribute and the module names in the MRO instead of importing anything, so it works when the ``anthropic`` extra is not installed.
    """
    return client_provider(obj) == 'anthropic' or _is_anthropic_sdk_client(obj)


def _is_anthropic_sdk_client(client: object) -> bool:
    """True for an instance of a class from the ``anthropic`` package, or of a subclass of one, that has ``messages``."""
    if not hasattr(client, 'messages'):
        return False
    return any(cls.__module__ == 'anthropic' or cls.__module__.startswith('anthropic.') for cls in type(client).__mro__)


_WRAPPED: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()
"""One adapter per raw Anthropic client, so state such as the warn-once refusal flag survives repeated wrapping."""


@overload
def as_llm_client(client: None) -> None: ...


@overload
def as_llm_client(client: Any) -> AsyncOpenAI: ...


def as_llm_client(client: Any) -> AsyncOpenAI | None:
    """Return ``client`` ready for evaluatorq: a raw Anthropic SDK client is wrapped, anything else passes through.

    Public entry points that accept an injected client call this, and so do the executors in `common.llm_call`, the only code that calls ``.chat`` or ``.responses``; an entry point that forgets still works. ``None``, an OpenAI client and an already wrapped adapter come back unchanged. A raw ``AsyncAnthropic``, ``AsyncAnthropicBedrock`` or ``AsyncAnthropicVertex`` (or a subclass of one) is recognised by the module name of a class in its MRO and its ``messages`` attribute, so nothing here imports ``anthropic``; the adapter module is imported only when a raw client is seen. The same raw client always gets the same adapter.
    """
    if client is None or client_provider(client) is not None:
        return client
    if not _is_anthropic_sdk_client(client):
        return client
    try:
        cached = _WRAPPED.get(client)
    except TypeError:  # not weak-referenceable: wrap without caching
        cached = None
    if cached is not None:
        return cast('AsyncOpenAI', cached)
    from evaluatorq.anthropic.client import wrap_anthropic_client

    wrapped = wrap_anthropic_client(client)
    with contextlib.suppress(TypeError):  # not weak-referenceable: use it uncached
        _WRAPPED[client] = wrapped
    # The adapter is duck-typed as an `AsyncOpenAI` everywhere downstream.
    return cast('AsyncOpenAI', wrapped)


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


def resolve_orq_credential(client: AsyncOpenAI | None = None) -> str | None:
    """Resolve the API key to send to the Orq host chosen by `resolve_results_base_url`.

    Returns ``client.api_key`` only when ``client`` routes through the Orq router,
    because only then is that key an Orq key for that host. Any other client (a
    direct OpenAI or Anthropic client) holds a provider key that must never be
    sent to Orq, so its key is ignored and ``ORQ_API_KEY`` is used instead.
    Returns ``None`` when neither is available, and the caller then makes no
    authenticated Orq call.
    """
    if client is not None and client_routes_through_orq(client):
        key = getattr(client, 'api_key', None)
        if isinstance(key, str) and key:
            return key
    return env_raw('ORQ_API_KEY') or None


def configured_llm_provider(*, extra_api_key: str | None = None, require_orq: bool = False) -> LLMProvider | None:
    """The provider `resolve_llm_client` picks from the environment, or ``None`` when no key is set.

    The one place the precedence lives: ``ORQ_API_KEY`` (or an explicit ``extra_api_key``), then ``OPENAI_API_KEY``, then ``ANTHROPIC_API_KEY``, in the order of `LLM_CREDENTIAL_KEYS`. With ``require_orq`` only the Orq key counts. An injected client is the caller's business and is not considered here.
    """
    if extra_api_key or env_raw('ORQ_API_KEY'):
        return 'orq'
    if require_orq:
        return None
    if env_raw('OPENAI_API_KEY'):
        return 'openai'
    if env_raw('ANTHROPIC_API_KEY'):
        return 'anthropic'
    return None


def require_models_served_by_provider(models: Mapping[str, str]) -> None:
    """Stop a run before any work when the environment picked Anthropic and a model it will use is not a Claude id.

    A leftover ``ANTHROPIC_API_KEY`` with no Orq or OpenAI key makes Anthropic the provider, while the built-in role defaults are ``openai/`` ids it refuses. Without this check the run starts and dies at the first call. Does nothing for any other provider.

    Args:
        models: Each model a command will use, keyed by where it is set (a role variable such as ``EVALUATORQ_SMART_MODEL`` or a flag), so the error can name what to change. The model rule itself stays in the adapter's ``normalize_model``.

    Raises:
        ModelClientMismatchError: Anthropic is the provider and at least one model is not served by it.
    """
    if configured_llm_provider() != 'anthropic':
        return
    from evaluatorq.anthropic.convert import normalize_model

    rejected: list[str] = []
    for source, model in models.items():
        try:
            normalize_model(model)
        except ModelClientMismatchError:  # noqa: PERF203
            rejected.append(f'{source}={model}')
    if rejected:
        raise ModelClientMismatchError(
            'Anthropic was chosen because ANTHROPIC_API_KEY is the only LLM key set, but it cannot serve '
            f'{", ".join(rejected)}. Set those to Claude model ids (for example claude-sonnet-5-5), '
            'or set ORQ_API_KEY or OPENAI_API_KEY to use another provider.'
        )


def _resolve_anthropic_client(max_retries: int | None) -> ResolvedClient:
    """Build the ``AsyncAnthropic`` behind the adapter for ``ANTHROPIC_API_KEY``.

    The SDK's own retry layer is the single one, sized like the OpenAI branches but from Anthropic's default when ``max_retries`` is ``None``. The SDK reads ``ANTHROPIC_BASE_URL`` itself.
    """
    try:
        from anthropic import DEFAULT_MAX_RETRIES, AsyncAnthropic
    except ImportError as exc:
        raise MissingLLMCredentialsError(
            'ANTHROPIC_API_KEY is the only LLM key set, but the anthropic package is not installed. '
            'Install it with: uv add "evaluatorq[anthropic]" (or: python -m pip install "evaluatorq[anthropic]"), '
            'or set ORQ_API_KEY or OPENAI_API_KEY.'
        ) from exc
    from evaluatorq.anthropic.client import wrap_anthropic_client

    retries = max_retries if max_retries is not None else DEFAULT_MAX_RETRIES
    # The adapter is duck-typed as an `AsyncOpenAI` everywhere downstream.
    return ResolvedClient(
        client=cast(
            'AsyncOpenAI',
            wrap_anthropic_client(AsyncAnthropic(api_key=env_raw('ANTHROPIC_API_KEY'), max_retries=retries)),
        ),
        owned=True,
        routes_through_orq=False,
    )


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
            ``ANTHROPIC_API_KEY`` either), or only ``ANTHROPIC_API_KEY`` is
            set and the ``anthropic`` extra is not installed.
    """
    if config_client is not None:
        config_client = as_llm_client(config_client)
        return ResolvedClient(
            client=config_client,
            owned=False,
            routes_through_orq=client_routes_through_orq(config_client),
        )

    provider = configured_llm_provider(extra_api_key=extra_api_key, require_orq=require_orq)

    if provider == 'anthropic':
        return _resolve_anthropic_client(max_retries)

    from openai import DEFAULT_MAX_RETRIES, AsyncOpenAI

    retries = max_retries if max_retries is not None else DEFAULT_MAX_RETRIES

    if provider == 'orq':
        orq_api_key = extra_api_key or env_raw('ORQ_API_KEY')
        host = (orq_host or env_raw('ORQ_BASE_URL', default_orq_host)).rstrip('/')
        router_url = f'{host}{ORQ_ROUTER_SUFFIX}'
        return ResolvedClient(
            client=AsyncOpenAI(api_key=orq_api_key, base_url=router_url, max_retries=retries),
            owned=True,
            routes_through_orq=True,
        )

    if provider == 'openai':
        openai_api_key = env_raw('OPENAI_API_KEY')
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

    if require_orq:
        raise MissingLLMCredentialsError(
            'ORQ routing is required for this target but no Orq credentials were '
            'found. Set ORQ_API_KEY (optionally ORQ_BASE_URL), or pass a pre-built '
            'Orq-routed client. The OPENAI_API_KEY and ANTHROPIC_API_KEY fallbacks are disabled here: '
            'ORQ agent targets must run on the Orq router.'
        )

    raise MissingLLMCredentialsError(
        'Missing LLM credentials. Set ORQ_API_KEY (optionally ORQ_BASE_URL), '
        'OPENAI_API_KEY (optionally OPENAI_BASE_URL) or ANTHROPIC_API_KEY, or pass a pre-built client.'
    )


__all__ = [
    'LLM_CREDENTIAL_KEYS',
    'ORQ_DEFAULT_HOST',
    'ORQ_ROUTER_SUFFIX',
    'LLMClient',
    'LLMProvider',
    'MissingLLMCredentialsError',
    'ModelClientMismatchError',
    'ResolvedClient',
    'as_llm_client',
    'client_routes_through_orq',
    'configured_llm_provider',
    'is_anthropic_client',
    'require_models_served_by_provider',
    'resolve_llm_client',
    'resolve_orq_credential',
    'resolve_results_base_url',
]
