"""Router pools stay derived, frontier-true and honest about relaxations (RES-1345).

Every structural check runs twice: offline on the synthetic garden in
`tests/common/garden_fakes.py`, and against the live `/v2/models` garden under
the `integration` marker. Nothing here
pins which models a pool seats; membership is whatever the derivation picks.
"""

import os
import subprocess
import sys
from typing import Any

import pytest
from pydantic import ValidationError

from evaluatorq import jury_presets
from evaluatorq import router_pools as rp
from evaluatorq.common.model_frontier import (
    MIN_TIER_RATIO,
    ScoredModel,
    derived_successor,
    load_garden_sync,
    pareto_frontier,
    seatable,
)
from evaluatorq.jury_presets import (
    ESTIMATED_COMPLETION_TOKENS,
    ESTIMATED_PROMPT_TOKENS,
    judge_family,
    model_identity,
)
from evaluatorq.router_pools import (
    DROPPED_POOLS,
    REVIEWED_UNDERIVABLE_POOLS,
    RouterPool,
    all_pool_model_ids,
    build_pools,
    get_pools,
    get_underivable_pools,
    pool_is_frontier,
)
from tests.common.garden_fakes import synthetic_garden as _synthetic_garden

LIVE = pytest.param('live', marks=[pytest.mark.integration, pytest.mark.allow_network])


# The shapes RES-1345 names, answered either by a shipping pool or by a
# written-down drop.
TICKET_NAMES = {
    'Balanced Pareto Trio',
    'Max Savings',
    'OpenAI Only',
    'Anthropic Only',
    'Bedrock Only',
    'EU Region Only',
    'Open-Weight / Portable',
    'Value Stack',
}


@pytest.fixture
def synthetic_garden() -> dict[str, ScoredModel]:
    return _synthetic_garden()


@pytest.fixture(scope='module', params=['synthetic', LIVE])
def mode(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope='module')
def garden(mode: str) -> dict[str, ScoredModel]:
    return _synthetic_garden() if mode == 'synthetic' else load_garden_sync()


@pytest.fixture(scope='module')
def derived(garden: dict[str, ScoredModel]) -> tuple[dict[str, RouterPool], dict[str, str]]:
    return build_pools(garden)


@pytest.fixture(scope='module')
def pools(derived: tuple[dict[str, RouterPool], dict[str, str]]) -> dict[str, RouterPool]:
    pools, _ = derived
    return pools


@pytest.fixture
def served_from(monkeypatch: pytest.MonkeyPatch):
    """Make get_pools() serve a given garden's derivation instead of the live one."""

    def _serve(garden: dict[str, ScoredModel]) -> None:
        derivation = build_pools(garden)
        monkeypatch.setattr(rp, '_live_pools', lambda: derivation)

    return _serve


class TestPoolMatchesTheGarden:
    def test_every_seat_and_reserve_is_in_the_garden(self, pools, garden):
        for pool in pools.values():
            missing = {*pool.models, *pool.reserve_models} - set(garden)
            assert not missing, f'{pool.name} recommends models the garden does not carry: {missing}'

    def test_published_ranges_match_the_members(self, pools, garden):
        for pool in pools.values():
            indices = [garden[m].default_index for m in pool.models]
            prices = [garden[m].price for m in pool.models]

            assert pool.index_range == (min(indices), max(indices))
            assert pool.price_range == (min(prices), max(prices))

    def test_published_cost_range_matches_the_members(self, pools, garden):
        """Per 1,000 requests at the jury estimate, from each member's own card rates."""
        for pool in pools.values():
            costs = [
                (garden[m].input_price * ESTIMATED_PROMPT_TOKENS + garden[m].output_price * ESTIMATED_COMPLETION_TOKENS)
                / 1_000
                for m in pool.models
            ]

            assert pool.cost_per_1k_range == (round(min(costs), 2), round(max(costs), 2))

    def test_pareto_scores_are_index_per_blended_dollar(self, pools, garden):
        for pool in pools.values():
            expected = {m: garden[m].default_index / garden[m].price for m in pool.models}

            assert pool.pareto_scores == pytest.approx(expected)

    def test_seats_are_distinct_models_not_rehosted_weights(self, pools):
        for pool in pools.values():
            identities = [model_identity(m) for m in pool.models]
            assert len(set(identities)) == len(identities), f'{pool.name} seats the same weights twice'


class TestDerivedPools:
    def test_three_members(self, pools):
        """The RES-1285 clean configuration, and the only size trivially safe
        under positional banding: one model per band, nothing unreachable."""
        for pool in pools.values():
            if not pool.measured:
                assert len(pool.models) == 3, pool.name

    def test_no_member_dominates_another(self, pools, garden):
        for pool in pools.values():
            assert pool_is_frontier(pool, garden) == [], pool.name

    def test_tier_rule_holds_or_the_pool_says_it_could_not(self, pools, garden):
        """`MIN_TIER_RATIO` between adjacent tiers, or the cost profile has nothing to win.
        A constraint that cannot reach it must flag the pool, not pass silently."""
        for pool in pools.values():
            # Judge on unrounded prices: a published 3.0 can round up from 2.96,
            # which is exactly the case the relaxed flag exists to disclose.
            prices = sorted((garden[m].price for m in pool.models), reverse=True)
            ratios = [prices[i] / prices[i + 1] for i in range(len(prices) - 1)]
            if pool.relaxed_tier_rule:
                assert any(ratio < MIN_TIER_RATIO for ratio in ratios), pool.name
            else:
                assert all(ratio >= MIN_TIER_RATIO for ratio in ratios), pool.name

    def test_published_tier_ratios_match_the_members(self, pools, garden):
        for pool in pools.values():
            prices = sorted((garden[m].price for m in pool.models), reverse=True)
            expected = tuple(round(prices[i] / prices[i + 1], 1) for i in range(len(prices) - 1))

            assert pool.tier_ratios == expected, pool.name


class TestSeatability:
    """The floor and the denylist hold for every recommendation this module makes."""

    def test_every_seat_and_reserve_is_seatable(self, pools, garden):
        for pool in pools.values():
            for model_id in (*pool.models, *pool.reserve_models):
                assert seatable(garden[model_id]), f'{pool.name} recommends unseatable {model_id}'

    def test_no_seat_has_a_newer_version_inside_its_pool(self, pools, garden):
        for spec in rp._SPECS:
            if spec.name not in pools:
                continue
            eligible = {r: m for r, m in garden.items() if m.enabled is not False and seatable(m) and spec.predicate(m)}
            pool = pools[spec.name]
            for model_id in (*pool.models, *pool.reserve_models):
                successor = derived_successor(model_id, eligible)
                assert successor is None, f'{spec.name} seats {model_id}, superseded by {successor}'

    def test_measured_default_stays_retired(self, pools):
        """Its seats fail the region and seatability rules, and a measured
        configuration cannot be edited without invalidating the measurement."""
        assert 'Measured Default' in DROPPED_POOLS
        assert 'Measured Default' not in pools
        assert 'RES-1285' in DROPPED_POOLS['Measured Default']

    def test_max_savings_retires_while_the_subdollar_frontier_is_under_three_lineages(self, pools, garden):
        """Counted in lineages, not identities: deepseek-v4-pro and deepseek-v4-flash
        are two identities and one vendor, the correlated-failure shape a pool
        exists to refuse. A wider cheap band is the tripwire to reinstate it."""
        cheap = pareto_frontier([m for m in garden.values() if m.price <= 1.0 and m.region != 'europe' and seatable(m)])
        lineages = {judge_family(m.router_id) for m in cheap}

        if len(lineages) >= 3:
            assert 'Max Savings' in pools, f'the cheap frontier spans {sorted(lineages)}; reinstate Max Savings'
        else:
            assert 'Max Savings' not in pools
            assert 'Max Savings' in DROPPED_POOLS

    def test_bedrock_only_stays_retired_while_its_catalog_is_thin(self, pools, garden):
        """A pool with no seat choices is the catalog restated, not a derivation."""
        assert 'Bedrock Only' not in pools
        assert 'Bedrock Only' in DROPPED_POOLS
        # Counted on the seatable frontier, not the card list: a dominated card
        # (haiku-4-5 under haiku-5-5) adds a row but no choice.
        bedrock = pareto_frontier([m for m in garden.values() if m.router_id.startswith('aws/') and seatable(m)])
        assert len(bedrock) <= 3, f'Bedrock frontier grew to {[m.router_id for m in bedrock]}: reconsider the pool'


class TestPoolConstraints:
    def test_every_shape_the_ticket_asked_for_is_answered(self, derived):
        """Shipped, dropped with the reason written down, or a reviewed data
        blocker, but never silently missing."""
        pools, underivable = derived
        unreviewed = set(underivable) - set(REVIEWED_UNDERIVABLE_POOLS)
        assert not unreviewed, f'pools the garden cannot seat with no review on record: {underivable}'
        answered = set(pools) | set(DROPPED_POOLS) | set(REVIEWED_UNDERIVABLE_POOLS)

        assert answered >= TICKET_NAMES
        assert not set(pools) & set(DROPPED_POOLS)
        for reason in DROPPED_POOLS.values():
            assert reason.strip()

    def test_an_underivable_spec_degrades_instead_of_breaking_the_rest(self, garden, pools):
        """An empty constraint records itself and leaves the other pools standing."""
        thin = {k: v for k, v in garden.items() if v.region != 'europe'}
        thinned, underivable = build_pools(thin)

        assert 'EU Region Only' not in thinned
        assert 'fewer than three' in underivable['EU Region Only']
        assert set(thinned) == set(pools) - {'EU Region Only'}

    def test_a_disabled_model_is_never_seated(self, garden):
        """enabled=False means listed and scored but not servable: seating one
        would ship a pool that 404s."""
        switched_off = ScoredModel(
            router_id='openai/imaginary-flagship',
            intelligence_index=99.0,
            input_price=2.0,
            output_price=2.0,
            enabled=False,
        )
        pools, _ = build_pools(dict(garden) | {switched_off.router_id: switched_off})

        for pool in pools.values():
            assert switched_off.router_id not in pool.models
            assert switched_off.router_id not in pool.reserve_models

    def test_eu_pool_is_entirely_eu(self, pools, garden):
        if 'EU Region Only' not in pools:
            pytest.skip('EU Region Only is not derivable from this garden')
        pool = pools['EU Region Only']
        for model_id in (*pool.models, *pool.reserve_models):
            assert garden[model_id].region == 'europe', f'{model_id} is not EU-served'

    def test_general_pools_seat_no_eu_pinned_models(self, pools, garden):
        """Seating one would silently pin an unconstrained preset to a region."""
        for name in ('Balanced Pareto Trio', 'OpenAI Only', 'Anthropic Only'):
            for model_id in pools[name].models if name in pools else ():
                assert garden[model_id].region != 'europe', f'{name} seats EU-pinned {model_id}'

    def test_openai_only_seats_only_openai_models(self, pools):
        if 'OpenAI Only' not in pools:
            pytest.skip('OpenAI Only is not derivable from this garden')
        for model_id in (*pools['OpenAI Only'].models, *pools['OpenAI Only'].reserve_models):
            assert model_id.startswith('openai/') or 'gpt' in model_id.lower(), f'OpenAI Only seats {model_id}'

    def test_anthropic_only_seats_only_anthropic_models(self, pools):
        if 'Anthropic Only' not in pools:
            pytest.skip('Anthropic Only is not derivable from this garden')
        for model_id in (*pools['Anthropic Only'].models, *pools['Anthropic Only'].reserve_models):
            assert model_id.startswith('anthropic/') or 'claude' in model_id.lower(), f'Anthropic Only seats {model_id}'

    def test_constrained_pools_earn_their_existence(self, pools):
        """A constrained pool that derives to the default's members is padding."""
        default = set(pools['Balanced Pareto Trio'].models)
        for pool in pools.values():
            if pool.name != 'Balanced Pareto Trio':
                assert set(pool.models) != default, f'{pool.name} duplicates the default'

    def test_a_pool_without_reserves_says_why(self, pools):
        for pool in pools.values():
            if not pool.reserve_models:
                assert pool.reserve_gap_reason.strip(), f'{pool.name} has no reserve and no reason'

    def test_reserves_are_not_rehosted_seated_weights(self, pools):
        for pool in pools.values():
            seated = {model_identity(m) for m in pool.models}
            for reserve in pool.reserve_models:
                assert model_identity(reserve) not in seated, (
                    f'{pool.name} reserves {reserve}, the same weights it already seats'
                )


class TestDerivation:
    def test_build_pools_is_deterministic(self, garden, pools):
        rebuilt, _ = build_pools(garden)

        assert {name: pool.models for name, pool in rebuilt.items()} == {
            name: pool.models for name, pool in pools.items()
        }

    def test_anthropic_spec_returns_when_a_third_distinct_model_is_scored(self, synthetic_garden):
        """The underivable register is a data blocker, not a retirement: one more
        distinct model on the constrained frontier brings the pool back."""
        assert 'Anthropic Only' in build_pools(synthetic_garden)[1]
        sonnet = ScoredModel(
            router_id='anthropic/claude-sonnet-4.5',
            intelligence_index=40.0,
            input_price=3.0,
            output_price=15.0,
            region='us',
            enabled=True,
        )
        pools, underivable = build_pools(synthetic_garden | {sonnet.router_id: sonnet})

        assert 'Anthropic Only' not in underivable
        pool = pools['Anthropic Only']
        assert sonnet.router_id in pool.models
        assert all(judge_family(m) == 'anthropic' for m in pool.models)
        assert pool.relaxed_tier_rule


class TestLiveAccessors:
    def test_importing_the_module_needs_no_network_or_key(self):
        """Pools derive on first use, so an import with no key must succeed."""
        env = {k: v for k, v in os.environ.items() if k != 'ORQ_API_KEY'}
        result = subprocess.run(
            [sys.executable, '-c', 'import evaluatorq.router_pools'],
            env=env | {'ORQ_API_KEY': ''},
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr

    def test_get_pools_and_the_register_describe_one_derivation(self, synthetic_garden, served_from):
        served_from(synthetic_garden)

        assert set(get_pools()) == {'Balanced Pareto Trio', 'OpenAI Only', 'EU Region Only'}
        assert set(get_underivable_pools()) == {'Anthropic Only'}

    def test_the_live_derivation_runs_once_per_process(self, synthetic_garden, monkeypatch):
        calls = []

        def counting_build() -> tuple[dict[str, RouterPool], dict[str, str]]:
            calls.append(1)
            return build_pools(synthetic_garden)

        monkeypatch.setattr(rp, 'build_pools', counting_build)
        rp._live_pools.cache_clear()
        get_pools()
        get_underivable_pools()
        rp._live_pools.cache_clear()

        assert len(calls) == 1

    def test_all_pool_model_ids_covers_reserves(self, synthetic_garden, served_from):
        served_from(synthetic_garden)
        ids = all_pool_model_ids()
        for pool in get_pools().values():
            assert set(pool.models) <= ids
            assert set(pool.reserve_models) <= ids


class TestValidators:
    def _pool(self, **overrides: Any) -> dict[str, Any]:
        return {
            'name': 'Test',
            'models': ('a/one', 'b/two'),
            'use_when': 'testing',
            'reserve_models': ('c/three',),
            'constraint': 'none',
            'index_range': (1.0, 2.0),
            'price_range': (1.0, 2.0),
            'cost_per_1k_range': (1.0, 2.0),
            **overrides,
        }

    def test_a_pool_needs_two_models(self):
        with pytest.raises(ValidationError, match='at least two'):
            RouterPool(**self._pool(models=('a/one',)))

    def test_a_reserve_cannot_be_seated(self):
        with pytest.raises(ValidationError, match='cannot also be seated'):
            RouterPool(**self._pool(reserve_models=('a/one',)))

    def test_no_reserve_needs_a_reason(self):
        with pytest.raises(ValidationError, match='why none exists'):
            RouterPool(**self._pool(reserve_models=()))


@pytest.mark.integration
@pytest.mark.allow_network
class TestConstraintsHoldAcrossPresetKinds:
    """A preset constraint binds the jury as well as the pool built on it.

    A jury's judges are hand-picked, so they can violate the predicate, and
    they did: the Open-Weight / Portable jury named `minimax/MiniMax-M2.7` as
    its reserve while `_OPEN_WEIGHT_FAMILIES` did not list minimax. Reserves
    count here because a reserve is a seat the moment a judge retires.
    """

    def test_every_open_weight_judge_and_reserve_is_open_weight(self):
        jury = jury_presets.PRESETS['Open-Weight / Portable']
        garden = load_garden_sync()

        closed = [
            router_id
            for router_id in (*jury.judges, *jury.reserve_judges)
            if (model := garden.get(router_id)) and not rp._is_open_weight(model)
        ]

        assert not closed, (
            f'{closed} sit on the Open-Weight / Portable panel but do not pass the '
            f'open-weight predicate. Either the lineage belongs in '
            f'_OPEN_WEIGHT_FAMILIES or the seat does not belong here.'
        )

    def test_every_eu_judge_and_reserve_is_eu_regioned(self):
        jury = jury_presets.PRESETS['EU Region']
        garden = load_garden_sync()

        elsewhere = [
            router_id
            for router_id in (*jury.judges, *jury.reserve_judges)
            if (model := garden.get(router_id)) and model.region != 'europe'
        ]

        assert not elsewhere, (
            f'{elsewhere} sit on the EU Region panel without an EU region on the card. '
            f'A data-residency preset that seats a model outside the region is the one '
            f'failure it cannot recover from.'
        )
