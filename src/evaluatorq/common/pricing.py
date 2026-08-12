"""Client-side cost for calls the Orq router does not price.

The router prices the Responses API (``/v3/router/responses`` returns
``usage.input_cost``/``output_cost``/``total_cost``) but NOT Chat Completions —
``/v3/router/chat/completions`` returns token counts only. Target agents run on
Responses and therefore carry cost; the LLM judge and the attack generators run
on Chat Completions and reported none, so every dollar figure in a red-team run
was the target's spend alone (RES-1295).

This module fills that gap by pricing chat-completion usage from Orq's own model
catalogue (``GET /v2/models``, ``input_cost``/``output_cost`` in USD per 1k
tokens). A model missing from the catalogue stays unpriced — ``total_cost``
``None``, ``priced_calls`` 0 — which the dashboard already renders as "no cost
data" rather than "$0.00".
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.common.llm_client import orq_base_url

if TYPE_CHECKING:
    from evaluatorq.contracts import Usage

# model_id -> (input cost, output cost), both USD per 1k tokens.
_price_cache: dict[str, tuple[float, float]] | None = None
_LOCK = asyncio.Lock()


def reset_price_cache() -> None:
    """Clear the process-lifetime price table; exists for test isolation."""
    global _price_cache
    _price_cache = None


def _parse_catalogue(payload: object) -> dict[str, tuple[float, float]]:
    """Build the price table from a ``/v2/models`` payload.

    Keyed on the bare ``model_id`` (``gpt-5-mini``), which is also how callers
    name models; a provider-prefixed id (``openai/gpt-5-mini``) is normalised at
    lookup time. The same model_id appears once per provider (openai, azure, …)
    at the same price, so last-one-wins is fine.
    """
    if not isinstance(payload, list):
        logger.warning('Model catalogue had unexpected shape {}; costs unavailable', type(payload).__name__)
        return {}
    prices: dict[str, tuple[float, float]] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get('model_id')
        inp, out = entry.get('input_cost'), entry.get('output_cost')
        if not isinstance(model_id, str) or not isinstance(inp, (int, float)) or not isinstance(out, (int, float)):
            continue
        prices[model_id] = (float(inp), float(out))
    return prices


async def _load_prices() -> dict[str, tuple[float, float]]:
    """Fetch the Orq model catalogue once per process.

    Failures are cached as an empty table: a run makes dozens of judge calls and
    must not re-pay a failing HTTP round-trip on each one. Cost simply stays
    unknown for the run, which is the honest reading.
    """
    global _price_cache
    if _price_cache is not None:
        return _price_cache
    async with _LOCK:
        if _price_cache is not None:
            return _price_cache
        api_key = os.environ.get('ORQ_API_KEY')
        if not api_key:
            logger.debug('No ORQ_API_KEY; chat-completion calls will report no cost')
            _price_cache = {}
            return _price_cache
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f'{orq_base_url()}/v2/models',
                    headers={'Authorization': f'Bearer {api_key}'},
                )
                response.raise_for_status()
                _price_cache = _parse_catalogue(response.json())
        except Exception as exc:  # pricing is telemetry; never fail the call it decorates
            logger.warning('Could not load Orq model prices ({}); calls will report no cost', exc)
            _price_cache = {}
        else:
            logger.debug('Loaded prices for {} models', len(_price_cache))
        return _price_cache


async def price_usage(usage: Usage | None, model: str) -> Usage | None:
    """Fill in cost on ``usage`` when the provider reported none.

    Returns ``usage`` unchanged when it is ``None``, already priced, or the model
    is absent from the catalogue.
    """
    if usage is None or usage.total_cost is not None:
        return usage
    prices = await _load_prices()
    price = prices.get(model) or prices.get(model.split('/', 1)[-1])
    if price is None:
        return usage
    input_price, output_price = price
    # ponytail: flat input rate — cached-read and cache-write tokens are billed at a
    # discount/premium the catalogue does not expose, so a cache-heavy call reads
    # slightly high. Split the rate here if /v2/models ever publishes the tiers.
    input_cost = usage.input_tokens / 1000 * input_price
    output_cost = usage.output_tokens / 1000 * output_price
    return usage.model_copy(
        update={
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost,
            'priced_calls': usage.calls,
        }
    )


__all__ = ['price_usage', 'reset_price_cache']
