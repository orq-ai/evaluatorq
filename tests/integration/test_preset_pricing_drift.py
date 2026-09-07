"""Do the preset's captured prices still match what the router bills? (RES-1171)

The offline suite recomputes every published $/1k from the committed rate table,
which keeps the docs and the code honest with each other but cannot notice a
provider repricing: both sides read the same capture. This is the half that can,
so it is the half that needs a key and a live gateway, and it is marked
`integration` rather than run in the normal suite.

A failure here is not a bug in the code. It means the committed rates have aged
past a price change: re-run the capture in the research repo that owns the
derivation, copy the new rates in, and let the offline cost test tell you which
published figures moved.

It asks `common.model_catalogue` for the price, host and all, which is the same
module that prices these judges at runtime. Answering this question is what made
that module key host-pinned ids on `provider/model_id` as well as on the bare id:
hosts republish each other's models at their own prices, and collapsing them
returned another host's rate for two of the seats here.
"""

import os

import pytest

from evaluatorq.common.model_catalogue import get_model_info
from evaluatorq.jury_presets import all_router_ids, judge_rates

# `/v2/models` publishes USD per 1,000 tokens; captured rates are per 1,000,000.
PER_1K_TO_PER_1M = 1000
# Providers publish rates to the cent per million, and both sides round-trip
# through a float, so compare at that resolution rather than exactly.
TOLERANCE = 0.005


@pytest.mark.integration
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_captured_judge_rates_still_match_the_live_catalogue() -> None:
    """Every seated judge bills what the committed table says it bills."""
    if not os.environ.get('ORQ_API_KEY'):
        pytest.skip('ORQ_API_KEY is required to read the live model catalogue')

    drifted: dict[str, str] = {}
    unlisted: list[str] = []
    for router_id in sorted(all_router_ids()):
        live = await get_model_info(router_id)
        if live is None:
            unlisted.append(router_id)
            continue
        rates = judge_rates(router_id)
        assert rates is not None, f'{router_id} is seated but carries no captured rates'
        live_input = live.input_cost_per_1k * PER_1K_TO_PER_1M
        live_output = live.output_cost_per_1k * PER_1K_TO_PER_1M
        if abs(live_input - rates.input_rate) > TOLERANCE or abs(live_output - rates.output_rate) > TOLERANCE:
            drifted[router_id] = f'captured {rates.input_rate}/{rates.output_rate}, live {live_input}/{live_output}'

    assert not drifted, (
        f'these judges are billed at a rate the committed table does not carry, so every '
        f'published $/1k that names them is wrong: {drifted}. Re-capture the rates.'
    )
    # Reported, not asserted: a workspace may simply not have a model switched
    # on, which is our own catalog scope and not a fact about the preset. The
    # offline suite is what refuses an unpriced seat.
    if unlisted:
        pytest.skip(f'not listed for this key: {unlisted}; every other seat matched')
