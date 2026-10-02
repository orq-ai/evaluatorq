"""Do the preset's captured rates and efforts still match the live catalogue? (RES-1171)

The offline suite recomputes every published $/1k from the committed rate table,
which keeps the docs and the code honest with each other but cannot notice a
provider repricing or a changed default effort: both sides read the same
capture. This is the half that can, so it needs a key and a live gateway and is
marked `integration` rather than run in the normal suite.

A failure here is not a bug in the code. It means the committed table has aged:
run `scripts/refresh_jury_judge_rates.py` and let the offline cost test tell you
which published figures moved. The comparison is the script's own, so this test
and the monthly refresh cannot disagree about what counts as drift.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from evaluatorq.common.model_catalogue import get_model_info
from evaluatorq.jury_presets import all_router_ids

_spec = importlib.util.spec_from_file_location(
    'refresh_jury_judge_rates',
    Path(__file__).resolve().parents[2] / 'scripts' / 'refresh_jury_judge_rates.py',
)
assert _spec is not None and _spec.loader is not None
refresh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refresh)


@pytest.mark.integration
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_captured_judge_rates_still_match_the_live_catalogue() -> None:
    """Every seated judge bills and runs at what the committed table says."""
    if not os.environ.get('ORQ_API_KEY'):
        pytest.skip('ORQ_API_KEY is required to read the live model catalogue')

    committed = json.loads(refresh.RATES_PATH.read_text())['judges']
    previous: dict[str, dict[str, object]] = {}
    current: dict[str, dict[str, object]] = {}
    unlisted: list[str] = []
    for router_id in sorted(all_router_ids()):
        assert router_id in committed, f'{router_id} is seated but carries no captured rates'
        live = await get_model_info(router_id)
        if live is None or live.input_cost_per_1k is None or live.output_cost_per_1k is None:
            unlisted.append(router_id)
            continue
        previous[router_id] = committed[router_id]
        current[router_id] = refresh.refreshed_row(live)

    drifted = refresh.decisions(previous, current)
    assert not drifted, f'the committed table no longer matches the live catalogue: {drifted}'
    # Reported, not asserted: a workspace may simply not have a model switched
    # on, which is our own catalog scope and not a fact about the preset. The
    # offline suite is what refuses an unpriced seat.
    if unlisted:
        pytest.skip(f'not listed for this key: {unlisted}; every other seat matched')
