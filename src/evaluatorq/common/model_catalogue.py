"""Orq's model catalogue: per-model prices and provider ids.

Fetched once per process from ``GET /v2/models`` and used for two things:

* **Pricing calls the router does not price.** ``/v3/router/responses`` returns
  ``usage.input_cost``/``output_cost``/``total_cost``; ``/v3/router/chat/completions``
  returns token counts only. Judges now default to Responses, but any caller
  still on Chat Completions gets its cost filled in here rather than recording a
  billed-to-nobody call (RES-1295).
* **Qualifying a bare model id.** The router's Responses endpoint requires
  ``provider/model`` (``openai/gpt-5-mini``); Chat Completions accepts the bare
  ``gpt-5-mini`` that configs are written with. The catalogue maps one to the other.

A model absent from the catalogue stays unpriced — ``total_cost`` ``None``,
``priced_calls`` 0 — which the dashboard already renders as "no cost data"
rather than "$0.00", and unqualified, which sends the caller back to Chat
Completions rather than guessing a provider.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, NamedTuple

from loguru import logger

from evaluatorq.common.llm_client import orq_base_url

if TYPE_CHECKING:
    from evaluatorq.contracts import Usage


class ModelInfo(NamedTuple):
    """One catalogue entry: costs in USD per 1k tokens, plus the routing provider."""

    input_cost: float
    output_cost: float
    provider: str


# model_id -> ModelInfo. None until the first fetch; {} when unavailable.
_catalogue: dict[str, ModelInfo] | None = None
_LOCK = asyncio.Lock()


def reset_catalogue_cache() -> None:
    """Clear the process-lifetime catalogue; exists for test isolation."""
    global _catalogue
    _catalogue = None


def _parse_catalogue(payload: object) -> dict[str, ModelInfo]:
    """Build the model table from a ``/v2/models`` payload.

    Keyed on the bare ``model_id`` (``gpt-5-mini``), which is how callers name
    models; a provider-prefixed id is normalised at lookup time. The same
    model_id appears once per hosting provider (openai, azure, …) at the same
    price, so the entry whose provider matches ``model_developer`` wins — that is
    the one the router resolves ``openai/gpt-5-mini`` to — and otherwise the
    first seen is kept.
    """
    if not isinstance(payload, list):
        logger.warning('Model catalogue had unexpected shape {}; costs unavailable', type(payload).__name__)
        return {}
    models: dict[str, ModelInfo] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        model_id, provider = entry.get('model_id'), entry.get('provider')
        inp, out = entry.get('input_cost'), entry.get('output_cost')
        if not isinstance(model_id, str) or not isinstance(provider, str):
            continue
        if not isinstance(inp, (int, float)) or not isinstance(out, (int, float)):
            continue
        if model_id in models and provider != entry.get('model_developer'):
            continue
        models[model_id] = ModelInfo(float(inp), float(out), provider)
    return models


async def _load_catalogue() -> dict[str, ModelInfo]:
    """Fetch the Orq model catalogue once per process.

    Failures are cached as an empty table: a run makes dozens of judge calls and
    must not re-pay a failing HTTP round-trip on each one. Cost simply stays
    unknown for the run, which is the honest reading.
    """
    global _catalogue
    if _catalogue is not None:
        return _catalogue
    async with _LOCK:
        if _catalogue is not None:
            return _catalogue
        api_key = os.environ.get('ORQ_API_KEY')
        if not api_key:
            logger.debug('No ORQ_API_KEY; model catalogue unavailable')
            _catalogue = {}
            return _catalogue
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f'{orq_base_url()}/v2/models',
                    headers={'Authorization': f'Bearer {api_key}'},
                )
                response.raise_for_status()
                _catalogue = _parse_catalogue(response.json())
        except Exception as exc:  # catalogue is telemetry/routing sugar; never fail the call it decorates
            logger.warning('Could not load the Orq model catalogue ({}); calls will report no cost', exc)
            _catalogue = {}
        else:
            logger.debug('Loaded catalogue for {} models', len(_catalogue))
        return _catalogue


async def _lookup(model: str) -> ModelInfo | None:
    """Catalogue entry for ``model``, trying the id as given then unprefixed."""
    catalogue = await _load_catalogue()
    return catalogue.get(model) or catalogue.get(model.split('/', 1)[-1])


async def qualified_model(model: str) -> str | None:
    """``provider/model`` for the Orq router, or ``None`` if unknown.

    Returns ``model`` unchanged when it is already qualified. A ``None`` means
    the caller must not use the router's Responses endpoint with this id — it
    rejects a bare model name — so it should stay on Chat Completions.
    """
    if '/' in model:
        return model
    info = await _lookup(model)
    return f'{info.provider}/{model}' if info else None


async def price_usage(usage: Usage | None, model: str) -> Usage | None:
    """Fill in cost on ``usage`` when the provider reported none.

    Returns ``usage`` unchanged when it is ``None``, already priced, or the model
    is absent from the catalogue.
    """
    if usage is None or usage.total_cost is not None:
        return usage
    info = await _lookup(model)
    if info is None:
        return usage
    # ponytail: flat input rate — cached-read and cache-write tokens are billed at a
    # discount/premium the catalogue does not expose, so a cache-heavy call reads
    # slightly high. Split the rate here if /v2/models ever publishes the tiers.
    input_cost = usage.input_tokens / 1000 * info.input_cost
    output_cost = usage.output_tokens / 1000 * info.output_cost
    return usage.model_copy(
        update={
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost,
            'priced_calls': usage.calls,
        }
    )


__all__ = ['ModelInfo', 'price_usage', 'qualified_model', 'reset_catalogue_cache']
