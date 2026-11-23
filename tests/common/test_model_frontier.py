"""The frontier engine's comparison rules, exercised directly (RES-1345, RES-1346).

The derivation tests in tests/unit/test_router_pools.py run the engine against a garden;
these pin the comparison semantics themselves, in
particular that every model is ranked at its catalogue default effort, where
judges run, and priced on a 3:1 blend of its own rates.
"""

import asyncio
from typing import Any

import httpx
import pytest
from loguru import logger

from evaluatorq.common import model_catalogue, model_frontier, retry
from tests.common.garden_fakes import card

from evaluatorq.common.model_frontier import (
    MAX_BLEND,
    MIN_SEAT_INDEX,
    NEVER_SEAT,
    NO_STRUCTURED_VERDICT,
    SUPERSEDED,
    MANUAL_MODEL_CARDS,
    UNROUTABLE,
    ScoredModel,
    derive_trio,
    derived_successor,
    garden_from_cards,
    headline_contradicts_its_own_rung,
    in_family_upgrades,
    load_garden,
    load_garden_sync,
    pareto_frontier,
    rank_cheapest_top,
    scored_model_from_card,
    seatable,
    unseatable_reason,
    with_aa_fills,
    with_manual_cards,
)


def scored(router_id: str, index: float, price: float, **kwargs: Any) -> ScoredModel:
    return ScoredModel(router_id=router_id, intelligence_index=index, input_price=price, output_price=price, **kwargs)


def graded(router_id: str, default: str | None, ladder: dict[str, float], price: float = 1.0) -> ScoredModel:
    """A card exposing a reasoningEffort parameter, headline deliberately off every rung."""
    return scored(router_id, 99.0, price, effort_parameter=True, default_effort=default, effort_indices=ladder)


def test_derive_trio_never_seats_a_free_model():
    """A frontier assembled outside build_pools may carry models with no price."""
    ranked = [scored('a/top', 60.0, 30.0), scored('b/mid', 50.0, 10.0), scored('c/low', 40.0, 3.0)]
    free = scored('e/free', 65.0, 0.0)

    members, relaxed = derive_trio([*ranked, free], identity_of=lambda router_id: router_id)

    assert [m.router_id for m in members] == ['a/top', 'b/mid', 'c/low']
    assert relaxed is False


class TestDominanceAndFrontier:
    def test_dominates_needs_strictly_better_on_one_axis(self):
        a = scored('x/a', 50.0, 2.0)
        twin = scored('x/b', 50.0, 2.0)
        cheaper = scored('x/c', 50.0, 1.0)

        assert not a.dominates(twin)
        assert cheaper.dominates(a)
        assert not a.dominates(cheaper)

    def test_frontier_keeps_ties_and_drops_the_dominated(self):
        strong = scored('x/strong', 60.0, 5.0)
        cheap = scored('x/cheap', 30.0, 0.5)
        dominated = scored('x/mid', 29.0, 1.0)

        frontier = pareto_frontier([strong, cheap, dominated])

        assert dominated not in frontier
        assert {m.router_id for m in frontier} == {'x/strong', 'x/cheap'}

    def test_single_model_is_its_own_frontier(self):
        only = scored('x/only', 10.0, 1.0)

        assert pareto_frontier([only]) == [only]


class TestDefaultEffortRanking:
    """Seats are ranked where the judge runs: the catalogue default effort."""

    def test_ranks_at_the_default_rung_not_the_best_or_the_headline(self):
        m = graded('x/a', 'medium', {'medium': 38.1, 'max': 51.2})

        assert m.default_index == pytest.approx(38.1)

    def test_a_card_without_an_effort_parameter_ranks_on_its_headline(self):
        m = scored('xai/grok-4-1-fast', 16.9, 0.5, effort_indices={'none': 16.9, 'reasoning': 30.6})

        assert m.default_index == pytest.approx(16.9)

    def test_no_benchmark_at_the_default_ranks_on_the_headline_not_another_rung(self):
        """deepseek-v4-pro: default high, benchmarked only at max and none, headline 36.0."""
        m = scored(
            'deepseek/deepseek-v4-pro',
            36.0,
            0.5,
            effort_parameter=True,
            default_effort='high',
            headline_effort='max',
            effort_indices={'max': 36.5, 'none': 20.4},
        )

        assert m.default_index == pytest.approx(36.0)
        assert m.ranked_on_headline
        assert seatable(m)

    def test_a_headline_measured_at_the_default_ranks_when_the_ladder_omits_it(self):
        """claude-opus-4-6: headline 37.8 at high, default high, ladder lists only max."""
        m = scored(
            'anthropic/claude-opus-4-6',
            37.8,
            1.0,
            effort_parameter=True,
            default_effort='high',
            headline_effort='high',
            effort_indices={'max': 43.7},
        )

        assert m.default_index == pytest.approx(37.8)

    def test_an_unknown_default_ranks_on_the_headline(self):
        m = scored(
            'mistral/mistral-medium-latest',
            33.0,
            1.0,
            effort_parameter=True,
            default_effort=None,
            effort_indices={'none': 30.0, 'high': 40.0},
        )

        assert m.default_index == pytest.approx(33.0)
        assert m.ranked_on_headline

    def test_a_card_scored_at_its_default_is_not_ranked_on_the_headline(self):
        assert not graded('x/a', 'medium', {'medium': 38.1, 'max': 51.2}).ranked_on_headline

    def test_a_free_card_neither_dominates_nor_is_dominated(self):
        free = scored('x/a', 90.0, 0.0)
        ranked = scored('x/b', 40.0, 5.0)

        assert not free.dominates(ranked)
        assert not ranked.dominates(free)
        assert pareto_frontier([free, ranked]) == [ranked]

    def test_dominance_reads_the_default_rung(self):
        """At its best a wins; at its default b does, for the same price."""
        a = graded('x/a', 'medium', {'medium': 38.1, 'max': 51.2})
        b = scored('x/b', 45.0, 1.0)

        assert b.dominates(a)
        assert not a.dominates(b)

    def test_the_floor_applies_at_the_default_rung(self):
        assert not seatable(graded('x/a', 'low', {'low': MIN_SEAT_INDEX - 1, 'high': 40.0}))


class TestPrice:
    def test_price_is_the_three_to_one_blend_of_the_rates(self):
        m = ScoredModel(router_id='x/a', intelligence_index=40.0, input_price=1.0, output_price=5.0)

        assert m.price == pytest.approx(2.0)

    def test_pareto_score_is_default_index_per_blended_dollar(self):
        m = ScoredModel(router_id='x/a', intelligence_index=40.0, input_price=1.0, output_price=5.0)

        assert m.pareto_score == pytest.approx(20.0)

    def test_pareto_score_is_none_only_for_a_free_card(self):
        free = ScoredModel(router_id='x/a', intelligence_index=40.0, input_price=0.0, output_price=5.0)
        unknown_default = ScoredModel(
            router_id='x/b', intelligence_index=40.0, input_price=1.0, output_price=5.0, effort_parameter=True
        )

        assert free.pareto_score is None
        assert unknown_default.pareto_score == pytest.approx(40.0 / unknown_default.price)

    def test_a_card_free_on_either_side_is_excluded_everywhere(self):
        free = ScoredModel(router_id='nvidia/kimi-k3', intelligence_index=57.1, input_price=0.0, output_price=1.0)
        paid = scored('baseten/kimi-k3', 50.0, 9.0)

        def family_of(_router_id):
            return 'moonshot'

        assert not seatable(free)
        assert not free.dominates(paid)
        assert pareto_frontier([free, paid]) == [paid]
        assert in_family_upgrades(paid, [free], family_of) == []


class TestSelfContradictingCards:
    """A headline-ranked card that scores its own headline rung twice ranks nowhere."""

    def test_a_contradicted_card_is_not_seatable(self):
        m = scored('moonshotai/kimi-k2.6', 44.2, 1.712, effort_indices={'none': 34.6})

        assert not seatable(m)

    def test_a_card_ranked_on_its_ladder_cannot_contradict_its_headline(self):
        m = graded('x/a', 'none', {'none': 34.6})

        assert seatable(m)

    def test_in_family_upgrades_never_promotes_a_contradicted_card(self):
        incumbent = scored('openai/gpt-5.4-mini', 29.8, 1.0)
        pretender = scored('openai/gpt-5-mini', 44.0, 0.8, effort_indices={'none': 20.0})

        def family_of(_router_id):
            return 'openai'

        assert in_family_upgrades(incumbent, [pretender], family_of) == []


class TestInFamilyUpgrades:
    def test_in_family_upgrades_still_fires_on_a_real_upgrade(self):
        incumbent = scored('openai/gpt-5.4-mini', 29.8, 1.0)
        successor = scored('openai/gpt-5.6-mini', 35.0, 0.9)

        def family_of(_router_id):
            return 'openai'

        upgrades = in_family_upgrades(incumbent, [successor], family_of)

        assert [m.router_id for m in upgrades] == ['openai/gpt-5.6-mini']


class TestSeatable:
    """Each rejection branch, which nothing until now constructed a model for.

    The library exercises `seatable` constantly, and every model it has ever
    passed came back True, so all three refusals were carried by inspection.
    """

    def test_a_model_on_both_axes_is_seatable(self):
        assert seatable(scored('openai/some-model', MIN_SEAT_INDEX + 1, 0.5))

    def test_the_index_floor_is_a_floor_and_not_a_ceiling_to_reach(self):
        assert not seatable(scored('openai/tiny', MIN_SEAT_INDEX - 0.1, 0.01))
        assert seatable(scored('openai/tiny', MIN_SEAT_INDEX, 0.01))

    def test_an_unroutable_model_stays_out_however_well_it_scores(self):
        router_id = next(iter(UNROUTABLE))

        assert not seatable(scored(router_id, 99.0, 0.01))

    def test_a_banned_family_stays_out_under_any_host(self):
        family = next(iter(NEVER_SEAT))

        assert not seatable(scored(f'some-new-host/{family}-turbo', 99.0, 0.01))

    def test_the_contributor_tier_stays_out_and_the_standard_tier_stays_seatable(self):
        assert not seatable(scored('meta/muse-spark-1.3-contributor', 99.0, 0.01))
        assert seatable(scored('meta/muse-spark-1.3', 99.0, 0.01))


class TestRankCheapestTop:
    """The cheap-aggregate derivation: best model per family under a blend ceiling."""

    @staticmethod
    def rank(garden: dict[str, ScoredModel], max_blend: float = 1.0) -> list[str]:
        return [m.router_id for m in rank_cheapest_top(garden, lambda rid: rid.split('/')[0], max_blend)]

    @staticmethod
    def garden(*models: ScoredModel) -> dict[str, ScoredModel]:
        return {m.router_id: m for m in models}

    def test_orders_by_index_one_model_per_family(self):
        garden = self.garden(
            scored('a/big', 40.0, 0.5), scored('a/small', 30.0, 0.1), scored('b/x', 35.0, 0.2), scored('c/y', 20.0, 0.1)
        )

        assert self.rank(garden) == ['a/big', 'b/x', 'c/y']

    def test_blend_ceiling_is_strict(self):
        garden = self.garden(scored('a/x', 40.0, 1.0), scored('b/y', 30.0, 0.99))

        assert self.rank(garden) == ['b/y']

    def test_ties_break_to_the_lower_price_then_router_id(self):
        garden = self.garden(scored('a/dear', 30.0, 0.5), scored('a/cheap', 30.0, 0.2), scored('b/z', 30.0, 0.2))

        assert self.rank(garden) == ['a/cheap', 'b/z']

    def test_unseatable_unrouteable_eu_free_and_floor_models_are_excluded(self):
        garden = self.garden(
            scored('ok/m', 20.0, 0.2),
            scored('minimax/MiniMax-M2.7', 40.0, 0.2),
            scored('eu/m', 40.0, 0.2, region='europe'),
            scored('free/m', 40.0, 0.0),
            scored('floor/m', MIN_SEAT_INDEX - 1, 0.2),
        )

        assert self.rank(garden) == ['ok/m']

    def test_a_card_with_an_unknown_default_effort_ranks_on_its_headline(self):
        garden = self.garden(graded('a/x', None, {'high': 50.0}, price=0.2), scored('b/y', 20.0, 0.2))

        assert self.rank(garden) == ['a/x', 'b/y']


def test_default_ceiling_is_the_published_max_blend():
    garden = {m.router_id: m for m in (scored('a/in', 30.0, MAX_BLEND - 0.01), scored('b/out', 40.0, MAX_BLEND))}

    assert [m.router_id for m in rank_cheapest_top(garden, lambda rid: rid.split('/')[0])] == ['a/in']


def test_a_host_that_answers_schemas_in_prose_loses_the_cheap_seat_to_one_that_does_not(monkeypatch):
    monkeypatch.setitem(NO_STRUCTURED_VERDICT, 'vendor/m-1', 'answers json_schema in prose')
    garden = {m.router_id: m for m in (scored('vendor/m-1', 30.0, 0.3), scored('host/m-1', 30.0, 0.3))}

    assert [m.router_id for m in rank_cheapest_top(garden, lambda rid: 'm')] == ['host/m-1']


class TestManualModelCards:
    def test_every_entry_is_comparable(self):
        for router_id, model in with_manual_cards({}).items():
            assert model.router_id in MANUAL_MODEL_CARDS
            assert not model.free, router_id

    def test_the_live_catalogue_leads_when_it_lists_the_route(self):
        router_id = next(iter(MANUAL_MODEL_CARDS))
        live = scored(router_id, 99.0, 0.5)

        assert with_manual_cards({router_id: live})[router_id] is live


def test_scored_model_from_card_reads_rates_per_million_and_the_default_effort():
    card_ = card(
        'p',
        'm',
        index=30.0,
        pricing={
            'input': {'cost': 0.2, 'currency': 'USD', 'per': 1_000_000, 'unit': 'tokens'},
            'output': {'cost': 1.0, 'currency': 'usd', 'per': 1_000_000, 'unit': 'tokens'},
        },
        supported_parameters=['reasoning_effort'],
        default_effort='high',
        default_effort_scored=True,
        reasoning_efforts={'high': {'intelligence_index': 30.0, 'index_version': 'v5', 'index_is_estimated': False}},
    )

    model = scored_model_from_card(card_)

    assert model is not None
    assert (model.router_id, model.provider, model.input_price, model.output_price) == ('p/m', 'p', 0.2, 1.0)
    assert (model.default_effort, model.default_index, model.headline_effort) == ('high', 30.0, 'high')
    assert model.region is None
    assert scored_model_from_card({'id': 'p/n'}) is None


@pytest.mark.parametrize(
    'rate',
    [
        None,
        {'cost': 1.0, 'currency': 'EUR', 'per': 1_000_000, 'unit': 'tokens'},
        {'cost': 1.0, 'currency': 'USD', 'per': 1, 'unit': 'requests'},
        {'cost': 1.0, 'currency': 'USD', 'per': 0, 'unit': 'tokens'},
        {'cost': 'n/a', 'currency': 'USD', 'per': 1_000_000, 'unit': 'tokens'},
    ],
)
def test_a_card_without_a_usd_token_rate_is_left_out(rate):
    pricing = {'input': rate, 'output': {'cost': 1.0, 'currency': 'USD', 'per': 1_000_000, 'unit': 'tokens'}}

    assert scored_model_from_card(card('p', 'm', pricing=pricing)) is None


def test_a_per_thousand_rate_is_scaled_to_per_million():
    rate = {'cost': 0.001, 'currency': 'USD', 'per': 1000, 'unit': 'tokens'}
    model = scored_model_from_card(card('p', 'm', pricing={'input': rate, 'output': rate}))

    assert model is not None
    assert (model.input_price, model.output_price) == (1.0, 1.0)


@pytest.mark.parametrize(
    ('location', 'region'), [(['eu'], 'europe'), (['eu', 'us'], None), (['us'], None), (None, None)]
)
def test_only_an_eu_only_card_is_region_pinned(location, region):
    model = scored_model_from_card(card('p', 'm', location=location))

    assert model is not None
    assert model.region == region


def test_a_default_effort_without_the_parameter_is_ignored():
    model = scored_model_from_card(card('p', 'm', default_effort='high', default_effort_scored=True))

    assert model is not None
    assert (model.effort_parameter, model.default_effort, model.headline_effort) == (False, None, 'high')


class TestSuperseded:
    def test_a_higher_version_of_the_same_tier_supersedes_across_hosts_and_claude_dashes(self):
        garden = {
            m.router_id: m
            for m in (
                scored('openai/gpt-5.6-luna', 25.0, 0.2),
                scored('openai/gpt-6-luna', 30.0, 0.2),
                scored('openai/gpt-6.1-sol', 50.0, 3.0),
                scored('anthropic/claude-haiku-4-5-20251001', 24.0, 2.0),
                scored('anthropic/claude-haiku-5-5', 35.0, 0.2),
            )
        }

        assert derived_successor('openai/gpt-5.6-luna', garden) == 'openai/gpt-6-luna'
        assert derived_successor('anthropic/claude-haiku-4-5-20251001', garden) == 'anthropic/claude-haiku-5-5'
        # A different tier word is a different product line, never a successor.
        assert derived_successor('openai/gpt-6-luna', garden) is None

    def test_a_newer_card_ranked_on_its_headline_is_a_successor(self):
        garden = {m.router_id: m for m in (scored('a/m-1', 30.0, 0.2), graded('a/m-2', 'high', {'max': 40.0}))}

        assert derived_successor('a/m-1', garden) == 'a/m-2'

    def test_an_unseatable_newer_card_does_not_retire_a_working_model(self):
        """A free successor cannot take the seat, so the older version keeps its place in the cheap ranking."""
        garden = {m.router_id: m for m in (scored('a/m-1', 30.0, 0.2), scored('a/m-2', 40.0, 0.0))}

        assert derived_successor('a/m-1', garden) is None
        assert [m.router_id for m in rank_cheapest_top(garden, lambda rid: rid.split('/')[0], 1.0)] == ['a/m-1']

    @pytest.mark.parametrize(
        ('old', 'new'),
        [
            ('aws/us-east-1.openai.gpt-5.4', 'aws/us-east-1.openai.gpt-5.6'),
            ('aws/eu.anthropic.claude-haiku-4-5-20251001-v1:0', 'aws/eu.anthropic.claude-haiku-5-5'),
        ],
    )
    def test_bedrock_routes_have_a_lineage(self, old, new):
        """In-region and dotted-vendor Bedrock ids parse, so a newer one retires the older."""
        garden = {m.router_id: m for m in (scored(old, 30.0, 0.2), scored(new, 40.0, 0.2))}

        assert derived_successor(old, garden) == new

    def test_superseded_models_are_refused_and_dropped_from_the_cheap_ranking(self):
        garden = {m.router_id: m for m in (scored('p/old-1', 40.0, 0.2), scored('p/old-2', 30.0, 0.2))}
        assert [m.router_id for m in rank_cheapest_top(garden, lambda rid: rid.split('/')[0] + rid[-1], 1.0)] == [
            'p/old-2'
        ]

        old = next(iter(SUPERSEDED))
        assert (unseatable_reason(scored(old, 40.0, 0.2)) or '').startswith('superseded')


class TestAaFills:
    """The register is empty on the v5 garden; these pin the rules for its next entry."""

    @pytest.fixture(autouse=True)
    def one_fill(self, monkeypatch):
        monkeypatch.setattr(model_frontier, 'AA_FILLS', {'p/m': {'effort': 'max', 'index': 30.0, 'source': 'test'}})

    def test_fills_only_the_missing_default_rung_and_never_overrides_live(self):
        empty = graded('p/m', 'max', {})
        measured = graded('p/m', 'max', {'max': 10.0})

        assert with_aa_fills({'p/m': empty})['p/m'].default_index == 30.0
        assert with_aa_fills({'p/m': measured})['p/m'].default_index == 10.0

    def test_a_fill_never_lands_on_a_rung_other_than_the_default(self):
        other = graded('p/m', 'high', {})

        filled = with_aa_fills({'p/m': other})['p/m']
        assert filled.effort_indices == {}
        assert filled.default_index == other.intelligence_index


def _effort_card(index: float, **fields: Any) -> dict[str, Any]:
    return card('p', 'm', index=index, supported_parameters=['reasoning_effort'], **fields)


class TestEffortCapture:
    """A host default must not relabel an AA benchmark's operating point."""

    @pytest.mark.parametrize('ladder', [['high'], 'high', {'high': 'not-a-rung', 'low': None}])
    def test_a_malformed_ladder_reads_as_absent(self, ladder):
        """External data: a bad `reasoning_efforts` must not take the whole garden down."""
        model = scored_model_from_card(_effort_card(40.0, reasoning_efforts=ladder))

        assert model is not None
        assert model.effort_indices == {}
        assert model.headline_effort is None

    @pytest.mark.parametrize('flag', [{}, {'default_effort_scored': False}, {'default_effort_scored': 'true'}])
    def test_a_host_default_does_not_prove_the_headline_was_measured_there(self, flag):
        model = scored_model_from_card(_effort_card(57.1, default_effort='max', **flag))

        assert model is not None
        assert model.default_effort == 'max'
        assert model.headline_effort == 'high'

    def test_a_headline_is_placed_on_the_one_rung_with_its_index(self):
        """Without a scored default, an equal index is the only evidence of where the headline was measured."""
        rungs = {'none': {'intelligence_index': 11.7, 'index_version': 'v5'}, 'reasoning': {'intelligence_index': 14.8, 'index_version': 'v5'}}
        model = scored_model_from_card(card('p', 'm', index=14.8, reasoning_efforts=rungs))

        assert model is not None
        assert model.headline_effort == 'reasoning'
        assert not headline_contradicts_its_own_rung(model)

    def test_a_scored_default_wins_over_an_equal_index_on_another_rung(self):
        rungs = {'medium': {'intelligence_index': 30.0, 'index_version': 'v5'}, 'high': {'intelligence_index': 40.0, 'index_version': 'v5'}}
        model = scored_model_from_card(
            _effort_card(40.0, default_effort='medium', default_effort_scored=True, reasoning_efforts=rungs)
        )

        assert model is not None
        assert model.headline_effort == 'medium'

    def test_a_scored_default_with_no_v5_rung_is_unscored(self):
        """The headline names no version, so a default with no v5 rung cannot rank on it."""
        model = scored_model_from_card(_effort_card(40.0, default_effort='medium', default_effort_scored=True))

        assert model is None

    def test_two_rungs_on_the_headline_index_place_it_nowhere(self):
        rungs = {'high': {'intelligence_index': 30.0, 'index_version': 'v5'}, 'max': {'intelligence_index': 30.0, 'index_version': 'v5'}}
        model = scored_model_from_card(_effort_card(30.0, default_effort='medium', reasoning_efforts=rungs))

        assert model is None

    def test_measured_default_and_other_rungs_remain_distinct(self):
        model = scored_model_from_card(
            _effort_card(
                39.7820971749522,
                default_effort='medium',
                default_effort_scored=True,
                reasoning_efforts={
                    'medium': {'intelligence_index': 39.7820971749522, 'index_version': 'v5'},
                    'max': {'intelligence_index': 47.5276426437724, 'index_version': 'v5'},
                },
            )
        )

        assert model is not None
        assert model.effort_indices == {'medium': 39.7820971749522, 'max': 47.5276426437724}
        assert model.headline_effort == 'medium'
        assert model.default_index == 39.7820971749522


NO_LADDER: dict[str, Any] = {'reasoning_efforts': None}


async def _no_sleep(_seconds: float) -> None:
    return None


V5_RUNG = {'reasoning_efforts': {'medium': {'intelligence_index': 40.0, 'index_version': 'v5'}}}


def _serve(monkeypatch: pytest.MonkeyPatch, pages: list[Any], calls: list[httpx.Request] | None = None) -> None:
    """Point the public catalogue fetch at canned pages, served in order."""
    queue = list(pages)

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        page = queue.pop(0)
        if isinstance(page, Exception):
            raise page
        return page if isinstance(page, httpx.Response) else httpx.Response(200, json=page)

    real = httpx.AsyncClient
    monkeypatch.setenv('ORQ_BASE_URL', 'https://catalog.test')
    monkeypatch.setattr(
        model_frontier.httpx, 'AsyncClient', lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )


class TestLiveGarden:
    """The garden is read from the public catalogue; these stub the HTTP transport, never the parsing."""

    def test_unscored_cards_are_left_out_of_the_garden(self):
        cards = [card('a', 'one'), card('b', 'unscored', index=None)]

        assert set(garden_from_cards(cards)) == {'a/one'}

    @pytest.mark.asyncio
    async def test_pages_are_followed_and_deprecated_cards_dropped(self, monkeypatch):
        monkeypatch.setenv('ORQ_API_KEY', 'a-key-the-public-read-must-not-send')
        calls: list[httpx.Request] = []
        _serve(
            monkeypatch,
            [
                {'data': [card('a', 'one', **V5_RUNG), card('a', 'old', deprecated=True)], 'has_more': True},
                {'data': [card('b', 'two'), 'not a card', {'deprecated': False}], 'has_more': False},
            ],
            calls,
        )

        garden = await load_garden()

        assert set(garden) == {'a/one', 'b/two'}
        assert [str(r.url) for r in calls] == [
            'https://catalog.test/v2/model-catalog',
            'https://catalog.test/v2/model-catalog?starting_after=a%2Fold',
        ]
        assert all('authorization' not in r.headers for r in calls)
        assert list(await model_frontier.public_listings()) == ['a/one', 'b/two']

    @pytest.mark.asyncio
    async def test_the_cards_are_fetched_once_per_process(self, monkeypatch):
        calls: list[httpx.Request] = []
        _serve(monkeypatch, [{'data': [card('a', 'one')], 'has_more': False}], calls)

        first = await model_frontier.public_cards()
        listings = await model_frontier.public_listings()

        assert first is await model_frontier.public_cards()
        assert list(listings) == ['a/one']
        assert len(calls) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('pages', 'status'),
        [
            ([httpx.Response(200, text='not json')], None),
            ([{'data': [], 'has_more': True}], None),
            ([{'data': [], 'has_more': False}], None),
            ([{'data': [{'no': 'id'}], 'has_more': True}], None),
            ([{'data': 7, 'has_more': False}], None),
            ([[1, 2]], None),
            ([{'data': [card('a', 'x')], 'has_more': True}] * 2, None),
        ],
    )
    async def test_a_bad_fetch_raises_and_is_not_cached(self, monkeypatch, pages, status):
        """A garden from a partial roster would retire live seats, so every failure is loud."""
        _serve(monkeypatch, [*pages, {'data': [card('a', 'one')], 'has_more': False}])

        with pytest.raises(model_catalogue.ModelCatalogueError) as raised:
            await model_frontier.public_cards()

        assert raised.value.status_code == status
        assert [c['id'] for c in await model_frontier.public_cards()] == ['a/one']

    @pytest.mark.asyncio
    @pytest.mark.parametrize('failure', [httpx.Response(503), httpx.Response(429), httpx.ConnectError('down')])
    async def test_a_transient_failure_is_retried(self, monkeypatch, failure):
        monkeypatch.setattr(retry.asyncio, 'sleep', _no_sleep)
        _serve(monkeypatch, [failure, {'data': [card('a', 'one')], 'has_more': False}])

        assert [c['id'] for c in await model_frontier.public_cards()] == ['a/one']

    @pytest.mark.asyncio
    async def test_a_persistent_503_raises_and_is_not_cached(self, monkeypatch):
        monkeypatch.setattr(retry.asyncio, 'sleep', _no_sleep)
        _serve(monkeypatch, [httpx.Response(503)] * retry.MAX_RETRY_ATTEMPTS + [{'data': [card('a', 'one')], 'has_more': False}])

        with pytest.raises(httpx.HTTPStatusError):
            await model_frontier.public_cards()

        assert [c['id'] for c in await model_frontier.public_cards()] == ['a/one']

    @pytest.mark.asyncio
    async def test_concurrent_cold_callers_share_one_fetch(self, monkeypatch):
        calls: list[httpx.Request] = []
        _serve(monkeypatch, [{'data': [card('a', 'one')], 'has_more': False}], calls)

        first, second = await asyncio.gather(model_frontier.public_cards(), model_frontier.public_cards())

        assert first is second
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_a_catalogue_with_no_scored_usd_card_raises(self, monkeypatch):
        """Every card dropped reads as every seat retired, so an empty garden is a failed read."""
        _serve(
            monkeypatch,
            [
                {
                    'data': [card('a', 'eur', pricing={'input': {'cost': 1.0, 'currency': 'EUR'}}, **V5_RUNG)],
                    'has_more': False,
                }
            ],
        )

        with pytest.raises(model_catalogue.ModelCatalogueError, match='none with a USD price'):
            await load_garden()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'rungs',
        [{}, {'reasoning_efforts': {'high': {'intelligence_index': 51.2, 'index_version': 'v4.3'}}}],
        ids=['no-ladders', 'v4.3-ladder'],
    )
    async def test_a_catalogue_not_on_index_v5_raises(self, monkeypatch, rungs):
        """Prod before the platform release: scored cards, but the floor would rank them on the wrong version."""
        _serve(
            monkeypatch,
            [{'data': [card('a', 'one', **NO_LADDER | rungs), card('b', 'two', **NO_LADDER)], 'has_more': False}],
        )

        with pytest.raises(model_catalogue.ModelCatalogueError, match='no AA index v5 rung'):
            await load_garden()

    @pytest.mark.asyncio
    async def test_a_card_left_on_an_older_index_is_dropped_not_ranked(self, monkeypatch):
        """Mid-migration: one card still on v4.3 must not be ranked against v5 scores."""
        old = {'reasoning_efforts': {'high': {'intelligence_index': 51.2, 'index_version': 'v4.3'}}}
        _serve(
            monkeypatch,
            [{'data': [card('a', 'new', **V5_RUNG), card('b', 'old', **old), card('c', 'plain')], 'has_more': False}],
        )

        assert set(await load_garden()) == {'a/new', 'c/plain'}

    @pytest.mark.asyncio
    async def test_a_kept_card_ranks_on_its_v5_rungs_only(self, monkeypatch):
        """{medium: v5, high: v4.3} on a medium default ranks on the v5 rung, never the v4.3 one."""
        mixed = {
            'reasoning_efforts': {
                'medium': {'intelligence_index': 24.6, 'index_version': 'v5'},
                'high': {'intelligence_index': 34.5, 'index_version': 'v4.3'},
            },
            'supported_parameters': ['reasoning_effort'],
            'default_effort': 'medium',
        }
        _serve(monkeypatch, [{'data': [card('a', 'mixed', index=34.5, **mixed)], 'has_more': False}])

        mixed_model = (await load_garden())['a/mixed']
        assert mixed_model.effort_indices == {'medium': 24.6}
        assert mixed_model.default_index == 24.6

    @pytest.mark.asyncio
    async def test_a_card_whose_default_rung_is_off_version_is_left_out(self, monkeypatch):
        """{medium: v5, high: v4.3} on a high default, headline 34.5: the headline must not stand in for the v4.3 rung."""
        mixed = {
            'reasoning_efforts': {
                'medium': {'intelligence_index': 24.6, 'index_version': 'v5'},
                'high': {'intelligence_index': 34.5, 'index_version': 'v4.3'},
            },
            'supported_parameters': ['reasoning_effort'],
            'default_effort': 'high',
        }
        _serve(
            monkeypatch,
            [{'data': [card('a', 'mixed', index=34.5, **mixed), card('b', 'new', **V5_RUNG)], 'has_more': False}],
        )
        messages: list[str] = []
        sink_id = logger.add(lambda message: messages.append(message.record['message']), level='WARNING')
        try:
            garden = await load_garden()
        finally:
            logger.remove(sink_id)

        assert set(garden) == {'b/new'}
        assert any('a/mixed' in m and "'high'" in m and "'v4.3'" in m for m in messages)

    @pytest.mark.asyncio
    async def test_cards_left_out_for_price_or_an_unversioned_score_are_logged(self, monkeypatch):
        eur = {'input': {'cost': 1.0, 'currency': 'EUR', 'per': 1_000_000, 'unit': 'tokens'}, 'output': {}}
        _serve(
            monkeypatch,
            [
                {
                    'data': [
                        card('a', 'ladder', **V5_RUNG),
                        card('b', 'eur', pricing=eur),
                        card('c', 'free-text', pricing={}),
                        card('d', 'unscored', index=None, pricing={}),
                        card('e', 'plain', **NO_LADDER),
                        card('f', 'off-rung', reasoning_efforts={'high': {'intelligence_index': 12.0, 'index_version': 'v5'}}),
                    ],
                    'has_more': False,
                }
            ],
        )
        messages: list[str] = []
        sink_id = logger.add(lambda message: messages.append(message.record['message']), level='WARNING')
        try:
            garden = await load_garden()
        finally:
            logger.remove(sink_id)

        assert set(garden) == {'a/ladder', 'e/plain'}
        assert any('non-USD price' in m and 'b/eur' in m and 'c/free-text' not in m for m in messages)
        assert any('no USD per-token price' in m and 'c/free-text' in m for m in messages)
        assert not any('d/unscored' in m for m in messages)
        assert any('no AA index v5 score' in m and 'f/off-rung' in m and 'e/plain' not in m for m in messages)
        assert any('no effort ladder' in m and 'e/plain' in m and 'a/ladder' not in m for m in messages)

    def test_the_sync_reader_refuses_inside_a_running_loop(self):
        import asyncio

        async def inside() -> None:
            with pytest.raises(RuntimeError, match='await load_garden'):
                load_garden_sync()

        asyncio.run(inside())


class TestRankCheapestTopSuccessors:
    """A newer version retires an older one only when the newer one could itself be seated."""

    @staticmethod
    def rank(garden: dict[str, ScoredModel], max_blend: float = 1.0) -> list[str]:
        return [m.router_id for m in rank_cheapest_top(garden, lambda rid: rid.split('/')[0], max_blend)]

    def test_an_eu_pinned_successor_retires_nothing(self):
        garden = {
            'a/foo-1': scored('a/foo-1', 50, 0.5),
            'a/foo-2': scored('a/foo-2', 60, 0.5, region='europe'),
        }
        assert self.rank(garden) == ['a/foo-1']

    def test_a_successor_over_the_ceiling_retires_nothing(self):
        garden = {
            'a/foo-1': scored('a/foo-1', 50, 0.5),
            'a/foo-2': scored('a/foo-2', 60, 5.0),
        }
        assert self.rank(garden) == ['a/foo-1']

    def test_an_eligible_successor_still_retires_the_older_version(self):
        garden = {
            'a/foo-1': scored('a/foo-1', 50, 0.5),
            'a/foo-2': scored('a/foo-2', 60, 0.5),
        }
        assert self.rank(garden) == ['a/foo-2']


class TestMalformedScores:
    """External card data never takes the whole garden down, and never reads as a score by accident."""

    @staticmethod
    def card_with(headline: object, rungs: dict[str, Any] | None = None) -> dict[str, Any]:
        extra = {} if rungs is None else {'reasoning_efforts': rungs}
        return card('p', 'm', index=headline, **extra)

    @pytest.mark.parametrize('headline', [float('nan'), float('inf'), True, 'n/a'])
    def test_a_non_finite_or_non_numeric_headline_is_unscored(self, headline):
        assert model_frontier.scored_model_from_card(self.card_with(headline)) is None

    @pytest.mark.parametrize('rung', ['n/a', True, float('nan')])
    def test_a_bad_rung_is_dropped_and_the_card_survives(self, rung):
        card_ = self.card_with(40, {'low': {'intelligence_index': rung, 'index_version': 'v5'}, 'high': {'intelligence_index': 40, 'index_version': 'v5'}})
        garden = model_frontier.garden_from_cards([card_, card('q', 'other')])
        assert set(garden) == {'p/m', 'q/other'}
        assert garden['p/m'].effort_indices == {'high': 40.0}


class TestSmallRobustness:
    def test_a_misspelled_manual_card_field_is_refused(self):
        with pytest.raises(ValueError, match='defualt_effort'):
            ScoredModel.model_validate({
                'router_id': 'x/y',
                'intelligence_index': 1.0,
                'input_price': 1.0,
                'output_price': 1.0,
                'defualt_effort': 'low',
            })

    @pytest.mark.parametrize(
        ('stamp', 'expected'),
        [
            ('2026-01-01T00:00:00Z', 1767225600000),
            (1767225600, 1767225600000),
            (1767225600000, 1767225600000),
            ('not a date', None),
            (True, None),
            (None, None),
        ],
    )
    def test_created_stamps_read_as_milliseconds_or_none(self, stamp, expected):
        assert model_frontier._created_ms(stamp) == expected

    def test_a_cheap_model_of_unknown_lineage_is_left_out_not_fatal(self):
        def family_of(router_id: str) -> str:
            if router_id.startswith('mystery'):
                raise ValueError('unknown lineage')
            return router_id.split('/')[0]

        garden = {
            'mystery/m-1': scored('mystery/m-1', 50, 0.5),
            'a/ok-1': scored('a/ok-1', 40, 0.5),
        }
        assert [m.router_id for m in rank_cheapest_top(garden, family_of, 1.0)] == ['a/ok-1']
