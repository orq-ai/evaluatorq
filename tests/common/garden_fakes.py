"""Hand-built model gardens and public `/v2/model-catalog` cards for offline frontier, preset and pool tests.

Router IDs and per-1M prices are real (read off the live catalogue cards on
2026-11-10, with the rows marked 2026-11-17 from the public `/v2/model-catalog`);
every intelligence index is invented, so nothing here is a claim
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
    'openai/gpt-6.1-sol': (2.0, 10.0),  # public /v2/model-catalog card, 2026-11-17
    'anthropic/claude-opus-5-5': (4.0, 20.0),
    'anthropic/claude-haiku-4-5-20251001': (1.0, 5.0),
    'anthropic/claude-haiku-5-5': (0.10, 0.50),  # public /v2/model-catalog card, 2026-11-17
    'anthropic/claude-sonnet-5-5': (2.0, 10.0),  # public /v2/model-catalog card, 2026-11-17
    'anthropic/claude-fable-5-2': (8.0, 40.0),  # public /v2/model-catalog card, 2026-11-17
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
    # Seats added by the AA index v5 reseat, public /v2/model-catalog cards, 2026-11-17.
    'tencent/deepseek-v4.1-flash': (0.15, 0.60),
    'wafer/MiniMax-M3.1-Flash': (0.15, 0.60),
    'moonshotai/kimi-k3.1-pro': (2.0, 10.0),
    'wafer/GLM-5.3-Flash': (0.05, 0.20),
    'openai/eu.gpt-6-luna': (0.11, 0.55),
    'alibaba/qwen3.6-35b-a3b': (0.248, 1.485),
}


def scored(router_id: str, index: float, region: str | None = None, **fields: Any) -> ScoredModel:
    """A headline-ranked card at its real rates."""
    input_price, output_price = PRICES[router_id]
    return ScoredModel(
        router_id=router_id,
        intelligence_index=index,
        input_price=input_price,
        output_price=output_price,
        region=region,
        provider=router_id.split('/')[0],
        **fields,
    )


def synthetic_garden() -> dict[str, ScoredModel]:
    """A small garden that derives every pool shape without the network.

    Shaped so that: the open frontier is a three-lineage trio with tiers at
    least MIN_TIER_RATIO apart (grok, deepseek-v4-pro, glm-5.3-flash); OpenAI
    Only seats three OpenAI models none of which is on the open frontier; EU
    Region Only seats three EU-pinned hosts; and Anthropic Only seats Fable,
    Sonnet and Haiku 5.5 with Opus in reserve, the shape the v5 catalogue gives.
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
        scored('anthropic/claude-fable-5-2', 62.0),
        scored('anthropic/claude-sonnet-5-5', 45.0),
        scored('anthropic/claude-haiku-5-5', 30.0),
        scored('google/eu.claude-sonnet-5', 53.0, region='europe'),
        scored('openai/eu.gpt-5.6-luna', 38.0, region='europe'),
        scored('tensorix/qwen/qwen3-coder-30b-a3b-instruct', 20.0, region='europe'),
    ]
    return {m.router_id: m for m in models}


def seated_garden() -> dict[str, ScoredModel]:
    """Every router ID a shipped preset seats, at its real rates; seats with no scored card come from MANUAL_MODEL_CARDS."""
    return {router_id: scored(router_id, 40.0) for router_id in PRICES}


def card(provider: str, model_id: str, index: object = 40.0, **overrides: Any) -> dict[str, Any]:
    """A public `/v2/model-catalog` card, shaped the way the endpoint returns one."""
    base: dict[str, Any] = {
        'id': f'{provider}/{model_id}',
        'provider': {'id': provider},
        'pricing': {
            'input': {'cost': 1.0, 'currency': 'USD', 'per': 1_000_000, 'unit': 'tokens'},
            'output': {'cost': 4.0, 'currency': 'USD', 'per': 1_000_000, 'unit': 'tokens'},
        },
        'supported_parameters': ['max_tokens', 'temperature'],
        'location': ['us'],
        'artificial_intelligence': {'artificial_analysis_intelligence_index': index},
        'deprecated': False,
    }
    return base | overrides
