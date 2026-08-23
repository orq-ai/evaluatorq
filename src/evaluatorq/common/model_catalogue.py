"""Orq's model catalogue: per-model prices, provider ids, and endpoint support.

Fetched once per process (per host) from ``GET /v2/models`` and used for three things:

* **Pricing calls the router does not price.** ``/v3/router/responses`` returns
  ``usage.input_cost``/``output_cost``/``total_cost``; ``/v3/router/chat/completions``
  returns token counts only. Judges now default to Responses, but any caller
  still on Chat Completions gets its cost filled in here rather than recording a
  billed-to-nobody call (RES-1295).
* **Qualifying a bare model id.** The router's Responses endpoint requires
  ``provider/model`` (``openai/gpt-5-mini``); Chat Completions accepts the bare
  ``gpt-5-mini`` that configs are written with. The catalogue maps one to the other.
* **Knowing which models serve Responses at all.** Entries carry
  ``metadata.supports_responses_api``, so a caller can ask before it calls
  instead of discovering it from a 400.

A model absent from the catalogue stays unpriced — ``total_cost`` ``None``,
``priced_calls`` 0 — which the dashboard already renders as "no cost data"
rather than "$0.00", and unqualified, which sends the caller back to Chat
Completions rather than guessing a provider.

Payload shape confirmed against live ``my.orq.ai/v2/models`` (2026-08-13): a bare
JSON list of 109 entries, no pagination envelope. Prices are **per 1k tokens** —
``gpt-5-mini`` returns ``input_cost: 0.00025`` / ``output_cost: 0.002``, matching
OpenAI's published $0.25 / $2.00 per 1M. ``input_currency``/``output_currency``
are ``''`` on most entries and ``'usd'`` on the rest; a non-empty non-USD entry
is skipped rather than silently mixed into a USD total.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, NamedTuple

import httpx
from loguru import logger

from evaluatorq.common.llm_client import orq_base_url, resolve_results_base_url

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from evaluatorq.contracts import Usage


class _ModelInfoFields(NamedTuple):
    """Field layout for `ModelInfo`; see that class for the contract."""

    input_cost_per_1k: float
    output_cost_per_1k: float
    provider: str
    supports_responses: bool
    # Accepted ``reasoning.effort`` values. ``None`` means the catalogue does not say
    # — absence is "unknown", never "empty".
    reasoning_efforts: frozenset[str] | None = None


class ModelInfo(_ModelInfoFields):
    """One catalogue entry.

    Costs are USD **per 1000 tokens** — the unit ``/v2/models`` publishes, and the
    reason `price_usage` divides by 1000. The names carry the unit so the
    arithmetic stays self-checking: ``Usage.input_cost`` is absolute USD for a
    call, 1000x away from this field, and the two meet in one expression.

    Split over a ``_ModelInfoFields`` base purely so ``__new__`` can normalize:
    ``typing.NamedTuple`` refuses to let a class body override it.
    """

    __slots__ = ()

    def __new__(  # noqa: PYI034 — a NamedTuple subclass really does construct the subclass
        cls,
        input_cost_per_1k: float,
        output_cost_per_1k: float,
        provider: str,
        supports_responses: bool,  # noqa: FBT001 — positional to match the tuple field order
        reasoning_efforts: frozenset[str] | None = None,
    ) -> ModelInfo:
        """Normalize an empty ``reasoning_efforts`` to ``None``.

        The two are not the same thing and the field's contract allows only one of
        them: ``None`` is "unknown, cannot validate"; an empty set would have to
        mean "this model accepts no effort at all". `validate_reasoning_effort`
        branches on ``is not None``, so an empty set makes it reject *every* value —
        including the defaults — with an empty accepted-values list, killing a run at
        pre-flight over a state that means nothing. `_parse_reasoning_efforts` already
        collapses the two; doing it in the type closes the same hole for
        `register_model`, which takes whatever a caller hands it.
        """
        return super().__new__(
            cls,
            input_cost_per_1k,
            output_cost_per_1k,
            provider,
            supports_responses,
            reasoning_efforts or None,
        )


# host -> (model_id -> ModelInfo). A host is absent until its first fetch, and
# maps to {} when the fetch failed or the account has no models. Keyed on host
# because a long-lived process (jobs runner, dashboard) can serve runs against
# prod and staging in turn, and prices attributed to the wrong deployment are
# worse than no prices.
_catalogues: dict[str, dict[str, ModelInfo]] = {}
# Built lazily: a module-level asyncio.Lock() binds to the loop running at import
# time, and the CLI and test suite both drive several asyncio.run() loops, where
# a lock bound to a dead loop raises "attached to a different loop".
_lock: asyncio.Lock | None = None
# Caller-registered entries, consulted before the fetched catalogue — the only way
# to price or correct a model /v2/models does not list. Keyed on the BARE id:
# `register_model` strips any `provider/` prefix so both spellings collapse to one.
_overrides: dict[str, ModelInfo] = {}
# Consecutive failed fetches per host: retry a few times, then give up for the
# process rather than hammering a host that is genuinely down.
_fetch_failures: dict[str, int] = {}
_MAX_FETCH_FAILURES = 3
# Env var rather than a parameter: `_load_catalogue` is called from pricing paths
# with no config object to thread.
_CATALOGUE_TIMEOUT_S = float(os.environ.get('EVALUATORQ_CATALOGUE_TIMEOUT_S', '30'))


def _catalogue_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


def reset_catalogue_cache() -> None:
    """Clear the process-lifetime catalogues; exists for test isolation.

    Leaves `register_model` overrides in place — they are caller intent, not
    cached state. Use `clear_model_overrides` for those.
    """
    global _lock
    _catalogues.clear()
    _fetch_failures.clear()
    _lock = None


def register_model(model_id: str, info: ModelInfo) -> None:
    """Register or override one catalogue entry for the rest of the process.

    Takes priority over the fetched catalogue, so this both adds a model Orq
    does not list and corrects one it lists wrongly. Without it an unknown model
    is silently unpriced (`price_usage`), silently forced off the Responses
    endpoint (`qualified_model`), and unvalidatable (`validate_reasoning_effort`)
    — three degradations a caller could previously neither see nor fix.

    ``model_id`` is stored **unprefixed**: ``'openai/gpt-x'`` and ``'gpt-x'`` name
    the same model and register the same entry. Normalizing on write rather than
    on read is what makes the two spellings interchangeable — the internal lookups
    ask for the bare id (that is what `qualified_model` produces), so a qualified
    registration stored verbatim would never be found and the override would be a
    silent no-op. A second registration under the other spelling replaces the
    first, which is correct: there is one model.

    ```python
    from evaluatorq.common.model_catalogue import ModelInfo, register_model

    register_model(
        'my-self-hosted-llama',
        ModelInfo(
            input_cost_per_1k=0.0002,
            output_cost_per_1k=0.0008,
            provider='self',
            supports_responses=False,
            reasoning_efforts=None,
        ),
    )
    ```
    """
    _overrides[model_id.split('/', 1)[-1]] = info


def clear_model_overrides() -> None:
    """Drop every `register_model` entry."""
    _overrides.clear()


async def get_model_info(model: str, client: AsyncOpenAI | None = None) -> ModelInfo | None:
    """The catalogue entry for ``model``, or ``None`` if nothing knows it.

    Public counterpart of the internal lookup, so a caller can ask what the
    catalogue believes instead of inferring it from an unpriced result.
    """
    return await _lookup(model, client)


def _entry_metadata(entry: dict[str, object]) -> dict[str, object]:
    """One entry's ``metadata`` mapping, or ``{}`` when it is any other shape.

    Providers have returned ``metadata`` as a list. ``(x or {}).get(...)`` survives
    an empty one and raises ``AttributeError`` on a populated one, which aborts the
    whole catalogue parse over a single malformed entry.
    """
    metadata = entry.get('metadata')
    return metadata if isinstance(metadata, dict) else {}


def _parse_reasoning_efforts(entry: dict[str, object]) -> frozenset[str] | None:
    """Accepted ``reasoning.effort`` values for one catalogue entry, or ``None``.

    Read from ``parameters[parameter == 'reasoningEffort'].config.options[].value``
    rather than the ``metadata.supports_reasoning_effort_*`` booleans: the two
    disagree on 50 of the 148 live entries (``kimi-k3`` advertises low/high in the
    flags and low/high/max in the options), and only the options list grows a new
    level without a schema change. ``None`` when the entry has no such parameter,
    which the caller must treat as "cannot validate", not "nothing allowed".
    """
    parameters = entry.get('parameters')
    if not isinstance(parameters, list):
        return None
    for parameter in parameters:
        if not isinstance(parameter, dict) or parameter.get('parameter') != 'reasoningEffort':
            continue
        config = parameter.get('config')
        options = config.get('options') if isinstance(config, dict) else None
        if not isinstance(options, list):
            return None
        values = {o['value'] for o in options if isinstance(o, dict) and isinstance(o.get('value'), str)}
        return frozenset(values) or None
    return None


def _parse_catalogue(payload: object) -> dict[str, ModelInfo]:
    """Build the model table from a ``/v2/models`` payload.

    Keyed on the bare ``model_id`` (``gpt-5-mini``), which is how callers name
    models; a provider-prefixed id is normalised at lookup time. The same
    model_id appears once per hosting provider (openai, azure, …) at the same
    price, so the entry whose provider matches ``model_developer`` wins — that is
    the one the router resolves ``openai/gpt-5-mini`` to — and otherwise the
    first seen is kept. Verified against the live catalogue: of 9 duplicated
    model_ids, none disagreed on price. A future disagreement is logged rather
    than silently resolved.
    """
    if not isinstance(payload, list):
        logger.warning(
            'Model catalogue had unexpected shape {} (expected a JSON list); costs and Responses routing unavailable',
            type(payload).__name__,
        )
        return {}
    models: dict[str, ModelInfo] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        model_id, provider = entry.get('model_id'), entry.get('provider')
        inp, out = entry.get('input_cost'), entry.get('output_cost')
        if not isinstance(model_id, str) or not isinstance(provider, str):
            continue
        # bool is an int subclass: `input_cost: true` would otherwise price at $1.00/1k.
        if isinstance(inp, bool) or isinstance(out, bool):
            continue
        if not isinstance(inp, (int, float)) or not isinstance(out, (int, float)):
            continue
        if inp < 0 or out < 0:
            continue
        # Currency is '' on most entries and 'usd' on the rest; only a stated
        # non-USD price is skipped, so the common empty case still prices.
        currencies = {str(entry.get('input_currency') or ''), str(entry.get('output_currency') or '')}
        if currencies - {'', 'usd'}:
            continue
        info = ModelInfo(
            input_cost_per_1k=float(inp),
            output_cost_per_1k=float(out),
            provider=provider,
            # `metadata` is provider-shaped and has arrived as a list; `.get` on a
            # non-mapping took the whole catalogue down rather than one model.
            supports_responses=bool(_entry_metadata(entry).get('supports_responses_api')),
            reasoning_efforts=_parse_reasoning_efforts(entry),
        )
        existing = models.get(model_id)
        if existing is not None:
            if (existing.input_cost_per_1k, existing.output_cost_per_1k) != (
                info.input_cost_per_1k,
                info.output_cost_per_1k,
            ):
                logger.debug(
                    'Catalogue prices disagree for {} ({} vs {}); keeping the model_developer entry if there is one',
                    model_id,
                    existing.provider,
                    provider,
                )
            if provider != entry.get('model_developer'):
                continue
        models[model_id] = info
    if isinstance(payload, list) and payload and not models:
        logger.warning(
            'Model catalogue returned {} entries but none parsed (payload shape change?); '
            'judges will run unpriced on chat completions',
            len(payload),
        )
    return models


async def _load_catalogue(client: AsyncOpenAI | None = None) -> dict[str, ModelInfo]:
    """Fetch the Orq model catalogue once per process, per host.

    The host is taken from ``client`` when it routes through the Orq router, so an
    injected staging/on-prem client is not priced against prod; otherwise
    ``ORQ_BASE_URL``.

    Failures are cached as an empty table: a run makes dozens of judge calls and
    must not re-pay a failing HTTP round-trip on each one. Cost simply stays
    unknown for the run, which is the honest reading — but note this also
    disables Responses routing for the run, so it is logged at WARNING with that
    consequence spelled out.
    """
    host = resolve_results_base_url(client) if client is not None else orq_base_url()
    cached = _catalogues.get(host)
    if cached is not None:
        return cached
    async with _catalogue_lock():
        cached = _catalogues.get(host)
        if cached is not None:
            return cached
        # An injected client's own key matches its host; an ambient ORQ_API_KEY for a
        # different workspace would 401 (caching {}) or price against the wrong one.
        api_key = (
            (getattr(client, 'api_key', None) or os.environ.get('ORQ_API_KEY'))
            if client is not None
            else os.environ.get('ORQ_API_KEY')
        )
        if not api_key:
            logger.debug('No ORQ_API_KEY and no client credential; model catalogue unavailable')
            _catalogues[host] = {}
            return _catalogues[host]
        payload: object = None
        try:
            async with httpx.AsyncClient(timeout=_CATALOGUE_TIMEOUT_S) as http_client:
                response = await http_client.get(
                    f'{host}/v2/models',
                    headers={'Authorization': f'Bearer {api_key}'},
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # Narrow on purpose: a bug inside _parse_catalogue must not be cached
            # away as "the network was down", so parsing happens outside the try.
            failures = _fetch_failures[host] = _fetch_failures.get(host, 0) + 1
            give_up = failures >= _MAX_FETCH_FAILURES
            logger.warning(
                'Orq model catalogue unavailable ({}: {}) at {}/v2/models (attempt {} of {}) — '
                'judges will run on chat completions and report no cost {}',
                type(exc).__name__,
                exc,
                host,
                failures,
                _MAX_FETCH_FAILURES,
                'for the rest of this process' if give_up else 'until a later call succeeds',
            )
            if give_up:
                _catalogues[host] = {}
                return _catalogues[host]
            # Not cached: a transient hiccup must not degrade the whole run.
            return {}
        _fetch_failures.pop(host, None)
        _catalogues[host] = _parse_catalogue(payload)
        logger.debug('Loaded catalogue for {} models from {}', len(_catalogues[host]), host)
        return _catalogues[host]


async def _lookup(model: str, client: AsyncOpenAI | None = None) -> ModelInfo | None:
    """Catalogue entry for ``model``, trying the id as given then unprefixed.

    Caller `register_model` overrides win over the fetched catalogue.
    """
    bare = model.split('/', 1)[-1]
    # One probe, not two: `register_model` normalizes to the bare id on write, so
    # the override key space has a single canonical spelling.
    override = _overrides.get(bare)
    if override is not None:
        return override
    catalogue = await _load_catalogue(client)
    return catalogue.get(model) or catalogue.get(bare)


async def qualified_model(model: str, client: AsyncOpenAI | None = None) -> str | None:
    """The ``provider/model`` id to send to the router's Responses endpoint.

    A bare id is qualified from the catalogue and yields ``None`` when the model
    is absent or its entry says it does not serve Responses — the caller must
    then stay on Chat Completions. An already-qualified id (one containing
    ``/``) passes through unchecked, since the caller has named a provider
    explicitly and the catalogue has nothing to add.
    """
    if '/' in model:
        return model
    info = await _lookup(model, client)
    if info is None or not info.supports_responses:
        return None
    return f'{info.provider}/{model}'


async def validate_reasoning_effort(effort: str, model: str, client: AsyncOpenAI | None = None) -> None:
    """Fail a run up front when ``model`` cannot serve ``effort``.

    Raises ``ValueError`` only when the catalogue actually lists the accepted
    values for ``model`` and ``effort`` is not among them. Every other case —
    model absent from the catalogue (an ``agent/<key>`` id always is), catalogue
    fetch failed, entry carries no ``reasoningEffort`` parameter — logs and
    returns, leaving the provider as the authority. The point is to fail before
    strategy generation and attack generation are paid for, not to become a
    second source of truth about what a model accepts.
    """
    info = await _lookup(model, client)
    if info is None or info.reasoning_efforts is None:
        logger.warning(
            'Cannot pre-validate reasoning effort {!r}: model {} is {} — the provider will '
            'reject an unsupported value at call time instead.',
            effort,
            model,
            'absent from the Orq catalogue' if info is None else 'listed without a reasoningEffort parameter',
        )
        return
    if effort not in info.reasoning_efforts:
        raise ValueError(
            f'Reasoning effort {effort!r} is not accepted by {model}. '
            f'Accepted values: {", ".join(sorted(info.reasoning_efforts))}.'
        )


async def price_usage(usage: Usage | None, model: str, client: AsyncOpenAI | None = None) -> Usage | None:
    """Fill in cost on a **single call's** ``usage`` when the provider reported none.

    Returns ``usage`` unchanged when it is ``None``, already priced, spans more
    than one call, or the model is absent from the catalogue. The multi-call
    guard matters: pricing an aggregate at one model's rate would also set
    ``priced_calls == calls``, asserting full pricing for calls this function
    never saw and defeating `Usage.cost_is_partial`.
    """
    if usage is None or usage.total_cost is not None or usage.calls > 1:
        return usage
    info = await _lookup(model, client)
    if info is None:
        logger.debug('Model {} is not in the Orq catalogue; call stays unpriced', model)
        return usage
    # ponytail: flat input rate — cached-read and cache-write tokens are billed at a
    # discount/premium the catalogue does not expose, so a cache-heavy call reads
    # slightly high. Split the rate here if /v2/models ever publishes the tiers.
    input_cost = usage.input_tokens / 1000 * info.input_cost_per_1k
    output_cost = usage.output_tokens / 1000 * info.output_cost_per_1k
    return usage.model_copy(
        update={
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost,
            'priced_calls': usage.calls,
        }
    )


__all__ = [
    'ModelInfo',
    'clear_model_overrides',
    'get_model_info',
    'price_usage',
    'qualified_model',
    'register_model',
    'reset_catalogue_cache',
    'validate_reasoning_effort',
]
