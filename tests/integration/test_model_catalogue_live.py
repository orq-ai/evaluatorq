"""Live-server integration test for the model catalogue loader (RES-1295).

Requires ORQ_API_KEY and makes a real ``GET /v2/models`` call against
``ORQ_BASE_URL`` (default prod). Excluded from the default test run (skipped
unless ``-m integration``).

This is the guard that makes a payload-shape or pricing-unit change loud: a
mock-based unit test would keep passing even if Orq changed ``/v2/models`` to
paginate, renamed ``input_cost``, or started publishing prices per-token
instead of per-1k tokens. Only a real response can catch that.
"""

from __future__ import annotations

import os

import pytest

from evaluatorq.common import model_catalogue as pricing


@pytest.mark.integration
class TestModelCatalogueLive:
    @pytest.mark.asyncio
    async def test_live_catalogue_loads_and_prices_a_known_model(self):
        if not os.environ.get('ORQ_API_KEY'):
            pytest.skip('ORQ_API_KEY not set')

        pricing.reset_catalogue_cache()
        try:
            catalogue = await pricing._load_catalogue()  # pyright: ignore[reportPrivateUsage]

            assert catalogue, 'live /v2/models returned no usable entries'

            info = catalogue.get('gpt-5-mini')
            assert info is not None, 'gpt-5-mini missing from the live catalogue'

            # Per-1k prices, not per-token or per-1M: bracket the known OpenAI
            # published rate ($0.25 / $2.00 per 1M) with generous slack so a
            # routine price change doesn't flake this, but a unit-shape bug
            # (10x/1000x off) trips it.
            assert 0.0001 <= info.input_cost_per_1k <= 0.001
            assert 0.001 <= info.output_cost_per_1k <= 0.01
        finally:
            pricing.reset_catalogue_cache()
