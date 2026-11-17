"""Hand-built model gardens and `/v2/models` cards for offline frontier, preset and pool tests.

Router IDs and per-1M prices are real (read off the live `/v2/models` cards on
2026-11-10); every intelligence index is invented, so nothing here is a claim
about a model.
"""

from __future__ import annotations

from typing import Any

from evaluatorq.common.model_frontier import ScoredModel

# Real card rates, USD per 1M input / output tokens, 2026-11-10.
PRICES: dict[str, tuple[float, float]] = {
    'xai/grok-4.5': (2.0, 6.0),
    'deepseek/deepseek-v4-pro': (0.435, 0.87),
    'deepseek/deepseek-flash': (0.14, 0.28),
    'zai/glm-5.3-flash': (0.075, 0.25),
    'zai/glm-5.2': (1.40, 4.40),
    'openai/gpt-5.4': (2.5, 15.0),
    'openai/gpt-5.4-mini': (0.75, 4.5),
    'openai/gpt-5.6-sol': (4.0, 20.0),
    'openai/gpt-6-luna': (0.1, 0.5),
    'openai/gpt-6-sol': (2.0, 10.0),
    'anthropic/claude-opus-5-5': (4.0, 20.0),
    'anthropic/claude-haiku-4-5-20251001': (1.0, 5.0),
    'google/gemini-3.6-flash': (1.5, 7.5),
    'google-ai/gemini-3.8-flash': (1.0, 3.0),
    'google/eu.claude-sonnet-5': (2.2, 11.0),
    'google/eu.gemini-3.7-flash': (0.825, 4.125),
    'openai/eu.gpt-5.6-luna': (0.22, 1.32),
    'azure/eu.gpt-5.6-luna': (0.20, 1.20),
    'aws/eu.anthropic.claude-haiku-5-5': (0.11, 0.55),
    'baseten/kimi-k3': (3.0, 15.0),
    'greenpt/qwen3.6-35b-a3b': (0.30, 1.80),
    'baseten/deepseek-v4.1-flash': (0.30, 1.20),
    'mistral/mistral-large-4': (0.68, 2.09),
    'tensorix/qwen/qwen3-coder-30b-a3b-instruct': (0.06, 0.25),
}


def scored(router_id: str, index: float, region: str = 'us', **fields: Any) -> ScoredModel:
    """A headline-ranked, enabled card at its real rates."""
    input_price, output_price = PRICES[router_id]
    return ScoredModel(
        router_id=router_id,
        intelligence_index=index,
        input_price=input_price,
        output_price=output_price,
        region=region,
        provider=router_id.split('/')[0],
        enabled=True,
        **fields,
    )


def synthetic_garden() -> dict[str, ScoredModel]:
    """A small garden that derives every pool shape without the network.

    Shaped so that: the open frontier is a three-lineage trio with tiers at
    least MIN_TIER_RATIO apart (grok, deepseek-v4-pro, glm-5.3-flash); OpenAI
    Only seats three OpenAI models none of which is on the open frontier; EU
    Region Only seats three EU-pinned hosts; and Anthropic Only has two
    distinct models, so it lands in the underivable register.
    """
    models = [
        scored('xai/grok-4.5', 53.8),
        scored('deepseek/deepseek-v4-pro', 43.1),
        scored('zai/glm-5.3-flash', 41.8),
        scored('openai/gpt-5.4', 50.0),
        scored('openai/gpt-5.4-mini', 40.0),
        scored('openai/gpt-6-luna', 30.0),
        scored('anthropic/claude-opus-5-5', 51.0),
        scored('anthropic/claude-haiku-4-5-20251001', 24.0),
        scored('google/eu.claude-sonnet-5', 53.0, region='europe'),
        scored('openai/eu.gpt-5.6-luna', 38.0, region='europe'),
        scored('tensorix/qwen/qwen3-coder-30b-a3b-instruct', 20.0, region='europe'),
    ]
    return {m.router_id: m for m in models}


def seated_garden() -> dict[str, ScoredModel]:
    """Every listed router ID a shipped preset seats, at its real rates; seats with no live card come from MANUAL_MODEL_CARDS."""
    return {router_id: scored(router_id, 40.0) for router_id in PRICES}


def card(provider: str, model_id: str, **overrides: Any) -> dict[str, Any]:
    """A raw `/v2/models` card, shaped the way the endpoint returns one."""
    base: dict[str, Any] = {
        'provider': provider,
        'model_id': model_id,
        'input_cost': 0.001,
        'output_cost': 0.004,
        'enabled': True,
        'metadata': {'region': 'us', 'autorouter': {'v2': {'intelligence_index': 40.0}}},
    }
    return base | overrides
