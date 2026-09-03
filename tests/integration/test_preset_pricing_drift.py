"""Do the preset's captured prices still match what the router bills? (RES-1171)

The offline suite recomputes every published $/1k from the committed garden
snapshot, which keeps the docs and the code honest with each other but cannot
notice a provider repricing: both sides read the same capture. This is the half
that can, so it is the half that needs a key and a live gateway, and it is marked
`integration` rather than run in the normal suite.

A failure here is not a bug in the code. It means the snapshot has aged past a
price change: re-run the research repo capture
(`projects/hitl-evaluators/scripts/refresh_model_garden_snapshot.py`), copy it
in, and let the offline cost test tell you which published figures moved.

It reads `/v2/models` directly rather than through `common.model_catalogue`,
which cannot answer the question a preset asks. That catalogue keys on the bare
model id, so it collapses every host of a model onto one entry and keeps the one
whose provider matches the developer. Two seats here are host-pinned decisions:
`azure/eu.gpt-5.6-luna` is seated because Azure bills 0.20/1.20 for the same
weights the OpenAI EU endpoint bills 0.22/1.32 for, and the catalogue answers
with the OpenAI figure. `deepseek/deepseek-v4-pro` is worse than imprecise: a
different host publishes the literal model id `deepseek/deepseek-v4-pro`, so an
exact-key lookup returns tensorix's 1.75/3.50 for a seat that bills 0.435/0.87.
Both would read as a 4x repricing that never happened.
"""

import os

import httpx
import pytest

from evaluatorq.common.model_frontier import load_judge_pricing
from evaluatorq.common.llm_client import orq_base_url
from evaluatorq.jury_presets import all_router_ids

# `/v2/models` publishes USD per 1,000 tokens; captured rates are per 1,000,000.
PER_1K_TO_PER_1M = 1000
# Providers publish rates to the cent per million, and both sides round-trip
# through a float, so compare at that resolution rather than exactly.
TOLERANCE = 0.005


@pytest.mark.integration
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_captured_judge_rates_still_match_the_live_catalogue() -> None:
    """Every seated judge bills what the snapshot says it bills."""
    api_key = os.environ.get('ORQ_API_KEY')
    if not api_key:
        pytest.skip('ORQ_API_KEY is required to read the live model catalogue')

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(
            f'{orq_base_url()}/v2/models',
            headers={'Authorization': f'Bearer {api_key}'},
        )
    response.raise_for_status()
    # Keyed by (host, model id), which is what a router ID names and what a
    # host-pinned seat is chosen on.
    live = {
        (entry['provider'], entry['model_id']): entry
        for entry in response.json()
        if isinstance(entry, dict) and entry.get('provider') and entry.get('model_id')
    }

    captured = load_judge_pricing()
    drifted: dict[str, str] = {}
    unlisted: list[str] = []
    for router_id in sorted(all_router_ids()):
        host, _, model_id = router_id.partition('/')
        entry = live.get((host, model_id))
        if entry is None:
            unlisted.append(router_id)
            continue
        rates = captured[router_id]
        live_input = entry['input_cost'] * PER_1K_TO_PER_1M
        live_output = entry['output_cost'] * PER_1K_TO_PER_1M
        if abs(live_input - rates['input_rate']) > TOLERANCE or abs(live_output - rates['output_rate']) > TOLERANCE:
            drifted[router_id] = (
                f'captured {rates["input_rate"]}/{rates["output_rate"]}, live {live_input}/{live_output}'
            )

    assert not drifted, (
        f'these judges are billed at a rate the snapshot does not carry, so every '
        f'published $/1k that names them is wrong: {drifted}. Re-capture the garden.'
    )
    # Reported, not asserted: a workspace may simply not have a model switched
    # on, which is our own catalog scope and not a fact about the preset. The
    # offline suite is what refuses an unpriced seat.
    if unlisted:
        pytest.skip(f'not listed for this key: {unlisted}; every other seat matched')
