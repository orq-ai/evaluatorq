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


class ModelInfo(NamedTuple):
    """One catalogue entry.

    Costs are USD **per 1000 tokens** — the unit ``/v2/models`` publishes, and the
    reason `price_usage` divides by 1000. The names carry the unit so the
    arithmetic stays self-checking: ``Usage.input_cost`` is absolute USD for a
    call, 1000x away from this field, and the two meet in one expression.
    """

    input_cost_per_1k: float
    output_cost_per_1k: float
    provider: str
    supports_responses: bool


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


def _catalogue_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


def reset_catalogue_cache() -> None:
    """Clear the process-lifetime catalogues; exists for test isolation."""
    global _lock
    _catalogues.clear()
    _lock = None


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
            supports_responses=bool((entry.get('metadata') or {}).get('supports_responses_api')),
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
        # When the host came from an injected client, that client's own key is
        # the credential that matches it — an ambient ORQ_API_KEY for a
        # different workspace/environment must not take priority, or the
        # catalogue silently 401s (caching {}) or prices against the wrong
        # workspace. Only fall back to the env var for the client-less path.
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
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                response = await http_client.get(
                    f'{host}/v2/models',
                    headers={'Authorization': f'Bearer {api_key}'},
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # Narrow on purpose: a bug inside _parse_catalogue must not be cached
            # away as "the network was down", so parsing happens outside the try.
            logger.warning(
                'Orq model catalogue unavailable ({}: {}) at {}/v2/models — judges will run '
                'on chat completions and report no cost for this run',
                type(exc).__name__,
                exc,
                host,
            )
            _catalogues[host] = {}
            return _catalogues[host]
        _catalogues[host] = _parse_catalogue(payload)
        logger.debug('Loaded catalogue for {} models from {}', len(_catalogues[host]), host)
        return _catalogues[host]


async def _lookup(model: str, client: AsyncOpenAI | None = None) -> ModelInfo | None:
    """Catalogue entry for ``model``, trying the id as given then unprefixed."""
    catalogue = await _load_catalogue(client)
    return catalogue.get(model) or catalogue.get(model.split('/', 1)[-1])


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


async def price_usage(usage: Usage | None, model: str, client: AsyncOpenAI | None = None) -> Usage | None:
    """Fill in cost on a **single call's** ``usage`` when the provider reported none.

    Returns ``usage`` unchanged when it is ``None``, already priced, does not
    span exactly one call, or the model is absent from the catalogue. The
    ``calls != 1`` guard matters on both sides: pricing an aggregate (``calls >
    1``) at one model's rate would set ``priced_calls == calls``, asserting full
    pricing for calls this function never saw and defeating
    `Usage.cost_is_partial`; pricing a ``calls == 0`` usage would write a real
    ``total_cost`` alongside ``priced_calls == 0``, which reads back as
    unprovenanced (`Usage.cost_source` is ``None`` below `priced_calls > 0`) —
    the exact unqualified-dollar-figure defect this module exists to prevent.
    On success the returned `Usage` also carries `estimated_calls` equal to the
    priced call, which is what marks the cost's provenance as ``'catalogue'``
    rather than ``'provider'`` once it reaches a span via
    `common.tracing.record_token_usage`.
    """
    if usage is None or usage.total_cost is not None:
        return usage
    if usage.calls != 1:
        if usage.calls == 0:
            logger.warning(
                'price_usage: usage for {} reports calls=0; skipping catalogue pricing rather than '
                'writing a cost with no priced call behind it',
                model,
            )
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
            'estimated_calls': usage.calls,
        }
    )


__all__ = ['ModelInfo', 'price_usage', 'qualified_model', 'reset_catalogue_cache']
