"""The frontier engine's comparison rules, exercised directly (RES-1345, RES-1346).

The derivation tests in tests/unit/test_router_pools.py run the engine against a garden;
these pin the comparison semantics themselves, in
particular that every model is ranked at its catalogue default effort, where
judges run, and priced on a 3:1 blend of its own rates.
"""

from typing import Any

import pytest
from loguru import logger

from evaluatorq.common import model_catalogue, model_frontier
from tests.common.garden_fakes import card

from evaluatorq.common.model_frontier import (
    AA_FILLS,
    MAX_BLEND,
    MIN_SEAT_INDEX,
    NEVER_SEAT,
    SUPERSEDED,
    UNLISTED_ROUTES,
    UNROUTABLE,
    ScoredModel,
    catalog_scope,
    derive_trio,
    derived_successor,
    dominated_by,
    garden_from_cards,
    in_family_upgrades,
    load_garden,
    load_garden_sync,
    pareto_frontier,
    rank_cheapest_top,
    scored_model_from_card,
    seatable,
    unseatable_reason,
    upgrade_verdict,
    with_aa_fills,
    with_unlisted_routes,
)


def scored(router_id: str, index: float, price: float, **kwargs: Any) -> ScoredModel:
    return ScoredModel(router_id=router_id, intelligence_index=index, input_price=price, output_price=price, **kwargs)


def graded(router_id: str, default: str | None, ladder: dict[str, float], price: float = 1.0) -> ScoredModel:
    """A card exposing a reasoningEffort parameter, headline deliberately off every rung."""
    return scored(router_id, 99.0, price, effort_parameter=True, default_effort=default, effort_indices=ladder)


def test_derive_trio_never_seats_an_unranked_or_free_model():
    """A frontier assembled outside build_pools may carry models with no default-rung score or no price."""
    ranked = [scored('a/top', 60.0, 30.0), scored('b/mid', 50.0, 10.0), scored('c/low', 40.0, 3.0)]
    unranked = graded('d/unranked', 'medium', {'high': 70.0}, price=1.0)
    free = scored('e/free', 65.0, 0.0)

    members, relaxed = derive_trio([*ranked, unranked, free], identity_of=lambda router_id: router_id)

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

    def test_dominated_by_names_every_winner_strongest_first(self):
        loser = scored('x/loser', 30.0, 2.0)
        strong = scored('x/strong', 50.0, 1.0)
        mid = scored('x/mid', 40.0, 2.0)
        unrelated = scored('x/pricier', 60.0, 5.0)

        winners = dominated_by(loser, [strong, mid, unrelated, loser])

        assert [m.router_id for m in winners] == ['x/strong', 'x/mid']


class TestDefaultEffortRanking:
    """Seats are ranked where the judge runs: the catalogue default effort."""

    def test_ranks_at_the_default_rung_not_the_best_or_the_headline(self):
        m = graded('x/a', 'medium', {'medium': 38.1, 'max': 51.2})

        assert m.default_index == pytest.approx(38.1)

    def test_a_card_without_an_effort_parameter_ranks_on_its_headline(self):
        m = scored('xai/grok-4-1-fast', 16.9, 0.5, effort_indices={'none': 16.9, 'reasoning': 30.6})

        assert m.default_index == pytest.approx(16.9)

    def test_no_benchmark_at_the_default_is_unrankable_not_borrowed(self):
        """gemini-3.6-flash: default medium, benchmarked only at high."""
        m = graded('google/gemini-3.6-flash', 'medium', {'high': 50.1})

        assert m.default_index is None
        assert not seatable(m)
        assert 'medium' in (unseatable_reason(m) or '')

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

    def test_a_disabled_card_is_unseatable(self):
        m = scored('x/a', 40.0, 1.0, enabled=False)

        assert unseatable_reason(m) == 'disabled in the model catalogue'

    def test_an_unknown_default_is_unrankable(self):
        m = graded('mistral/mistral-medium-latest', None, {'none': 30.0, 'high': 40.0})

        assert m.default_index is None
        assert not seatable(m)

    def test_an_unrankable_card_neither_dominates_nor_is_dominated(self):
        unrankable = graded('x/a', 'medium', {'high': 90.0}, price=0.01)
        ranked = scored('x/b', 40.0, 5.0)

        assert not unrankable.dominates(ranked)
        assert not ranked.dominates(unrankable)
        assert pareto_frontier([unrankable, ranked]) == [ranked]

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

    def test_pareto_score_is_none_where_the_card_is_not_comparable(self):
        free = ScoredModel(router_id='x/a', intelligence_index=40.0, input_price=0.0, output_price=5.0)
        unknown_default = ScoredModel(
            router_id='x/b', intelligence_index=40.0, input_price=1.0, output_price=5.0, effort_parameter=True
        )

        assert free.pareto_score is None
        assert unknown_default.pareto_score is None

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

    def test_upgrade_verdict_sends_a_contradicted_successor_to_a_human(self):
        incumbent = scored('x/old', 40.0, 2.0)
        successor = scored('x/new', 44.2, 1.0, effort_indices={'none': 34.6})

        verdict, reason = upgrade_verdict(incumbent, successor)

        assert verdict == 'trade'
        assert 'cannot hold a seat' in reason

    def test_upgrade_verdict_never_seats_a_comparable_but_unseatable_successor(self):
        """Both axes defined is not enough: a never-seat family is still refused."""
        incumbent = scored('x/old', 40.0, 2.0)
        successor = scored(f'x/{next(iter(NEVER_SEAT))}-new', 60.0, 1.0)

        verdict, reason = upgrade_verdict(incumbent, successor)

        assert verdict == 'trade'
        assert 'never seated' in reason

    def test_upgrade_verdict_replaces_a_dead_incumbent(self):
        """A disabled incumbent is replaced even by a successor that does not dominate it."""
        incumbent = scored('x/old', 50.0, 1.0, enabled=False)
        successor = scored('x/new', 45.0, 2.0)

        verdict, reason = upgrade_verdict(incumbent, successor)

        assert verdict == 'seat'
        assert 'disabled' in reason

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


class TestRankCheapestTop:
    """The Cheap Aggregate derivation: best model per family under a blend ceiling."""

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

    def test_unseatable_unrouteable_eu_disabled_free_and_floor_models_are_excluded(self):
        garden = self.garden(
            scored('ok/m', 20.0, 0.2),
            scored('minimax/MiniMax-M2.7', 40.0, 0.2),
            scored('eu/m', 40.0, 0.2, region='europe'),
            scored('off/m', 40.0, 0.2, enabled=False),
            scored('free/m', 40.0, 0.0),
            scored('floor/m', MIN_SEAT_INDEX - 1, 0.2),
        )

        assert self.rank(garden) == ['ok/m']

    def test_a_card_with_an_unknown_default_effort_is_not_ranked(self):
        garden = self.garden(graded('a/x', None, {'high': 50.0}, price=0.2), scored('b/y', 20.0, 0.2))

        assert self.rank(garden) == ['b/y']


def test_default_ceiling_is_the_published_max_blend():
    garden = {m.router_id: m for m in (scored('a/in', 30.0, MAX_BLEND - 0.01), scored('b/out', 40.0, MAX_BLEND))}

    assert [m.router_id for m in rank_cheapest_top(garden, lambda rid: rid.split('/')[0])] == ['a/in']


class TestUnlistedRoutes:
    def test_every_entry_ranks_at_its_default_effort(self):
        """Or is a reviewed unbenchmarked hold, which must not be rankable by accident."""
        from evaluatorq.jury_presets import REVIEWED_UNBENCHMARKED_SEATS

        for router_id, model in with_unlisted_routes({}).items():
            assert model.router_id in UNLISTED_ROUTES
            if router_id in REVIEWED_UNBENCHMARKED_SEATS:
                assert model.default_index is None, router_id
                continue
            assert model.comparable, router_id

    def test_the_live_catalogue_leads_when_it_lists_the_route(self):
        router_id = next(iter(UNLISTED_ROUTES))
        live = scored(router_id, 99.0, 0.5)

        assert with_unlisted_routes({router_id: live})[router_id] is live


def test_scored_model_from_card_reads_rates_per_million_and_the_default_effort():
    card: dict[str, object] = {
        'provider': 'p',
        'model_id': 'm',
        'input_cost': 0.0002,
        'output_cost': 0.001,
        'enabled': True,
        'metadata': {
            'region': 'rest',
            'autorouter': {
                'v2': {
                    'intelligence_index': 30.0,
                    'aa_slug': 'm-high',
                    'reasoning_efforts': {'high': {'aa_slug': 'm-high', 'intelligence_index': 30.0}},
                }
            },
        },
        'parameters': [{'parameter': 'reasoningEffort', 'config': {'default': 'high', 'options': [{'value': 'high'}]}}],
    }

    model = scored_model_from_card(card)

    assert model is not None
    assert (model.router_id, model.input_price, model.output_price) == ('p/m', 0.2, 1.0)
    assert (model.default_effort, model.default_index, model.headline_effort) == ('high', 30.0, 'high')
    assert scored_model_from_card({'provider': 'p', 'model_id': 'n'}) is None


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

    def test_an_unrankable_newer_card_does_not_retire_a_working_model(self):
        garden = {m.router_id: m for m in (scored('a/m-1', 30.0, 0.2), graded('a/m-2', 'high', {'max': 40.0}))}

        assert derived_successor('a/m-1', garden) is None

    def test_superseded_models_are_refused_and_dropped_from_the_cheap_ranking(self):
        garden = {m.router_id: m for m in (scored('p/old-1', 40.0, 0.2), scored('p/old-2', 30.0, 0.2))}
        assert [m.router_id for m in rank_cheapest_top(garden, lambda rid: rid.split('/')[0] + rid[-1], 1.0)] == [
            'p/old-2'
        ]

        old = next(iter(SUPERSEDED))
        assert (unseatable_reason(scored(old, 40.0, 0.2)) or '').startswith('superseded')


class TestAaFills:
    def test_fills_only_the_missing_default_rung_and_never_overrides_live(self):
        router_id = next(iter(AA_FILLS))
        effort = AA_FILLS[router_id]['effort']
        empty = graded(router_id, effort, {})
        measured = graded(router_id, effort, {effort: 10.0})

        assert with_aa_fills({router_id: empty})[router_id].default_index == AA_FILLS[router_id]['index']
        assert with_aa_fills({router_id: measured})[router_id].default_index == 10.0

    def test_a_fill_never_lands_on_a_rung_other_than_the_default(self):
        router_id = next(iter(AA_FILLS))
        other = graded(router_id, 'high', {})

        assert with_aa_fills({router_id: other})[router_id].default_index is None


def _v2_card(v2: dict[str, Any]) -> dict[str, object]:
    return {
        'provider': 'p',
        'model_id': 'm',
        'input_cost': 0.001,
        'output_cost': 0.004,
        'metadata': {'autorouter': {'v2': v2}},
    }


class TestEffortCapture:
    """A host default must not relabel an AA benchmark's operating point."""

    @pytest.mark.parametrize('host_default', ['none', 'max'])
    def test_a_host_default_does_not_prove_the_headline_was_measured_there(self, host_default):
        model = scored_model_from_card(
            _v2_card({'aa_slug': 'kimi-k3', 'intelligence_index': 57.1, 'default_effort': host_default})
        )

        assert model is not None
        assert model.effort_indices == {}
        assert model.headline_effort is None

    def test_a_non_reasoning_high_headline_is_not_folded_into_an_adaptive_ladder(self):
        model = scored_model_from_card(
            _v2_card({
                'aa_slug': 'claude-sonnet-4-6',
                'aa_name': 'Claude Sonnet 4.6 (Non-reasoning, High Effort)',
                'intelligence_index': 35.9,
                'default_effort': 'high',
                'reasoning_efforts': {'max': {'aa_slug': 'claude-sonnet-4-6-adaptive', 'intelligence_index': 47.2}},
            })
        )

        assert model is not None
        assert model.effort_indices == {'max': 47.2}
        assert model.headline_effort is None

    def test_measured_default_and_other_rungs_remain_distinct(self):
        model = scored_model_from_card(
            _v2_card({
                'aa_slug': 'gpt-6-sol-medium',
                'intelligence_index': 39.7820971749522,
                'default_effort': 'medium',
                'reasoning_efforts': {
                    'medium': {'aa_slug': 'gpt-6-sol-medium', 'intelligence_index': 39.7820971749522},
                    'max': {'aa_slug': 'gpt-6-sol', 'intelligence_index': 47.5276426437724},
                },
            })
        )

        assert model is not None
        assert model.effort_indices == {'medium': 39.7820971749522, 'max': 47.5276426437724}
        assert model.headline_effort == 'medium'


class TestLiveGarden:
    """The garden is read from /v2/models; these stub the fetch, never the parsing."""

    def test_a_catalogue_with_a_disabled_card_is_workspace_scoped(self):
        cards = [card('a', 'one'), card('b', 'two', enabled=False)]

        assert catalog_scope(cards) == 'workspace'
        assert catalog_scope(cards[:1]) == 'project'

    def test_unscored_cards_are_left_out_of_the_garden(self):
        cards = [card('a', 'one'), card('b', 'unscored', metadata={})]

        assert set(garden_from_cards(cards)) == {'a/one'}

    def test_a_non_usd_card_is_left_out_rather_than_mixed_in(self):
        """The catalogue's own price reader decides, so the garden and the pricing table agree."""
        cards = [card('a', 'one'), card('b', 'eur', input_currency='eur')]

        assert set(garden_from_cards(cards)) == {'a/one'}

    @pytest.mark.asyncio
    async def test_load_garden_ranks_the_fetched_cards(self, monkeypatch):
        async def fetch(_client=None):
            return (card('a', 'one'), card('b', 'two', enabled=False))

        monkeypatch.setattr(model_frontier, 'model_cards', fetch)

        garden = await load_garden()

        assert set(garden) == {'a/one', 'b/two'}
        assert garden['b/two'].enabled is False

    @pytest.mark.asyncio
    async def test_a_project_scoped_catalogue_is_used_with_a_warning(self, monkeypatch):
        """Decision RES-1346: warn and carry on, since a project key is what most callers hold."""
        cards = (card('a', 'one'),)

        async def fetch(_client=None):
            return cards

        monkeypatch.setattr(model_frontier, 'model_cards', fetch)
        messages: list[str] = []
        sink = logger.add(messages.append, level='WARNING')
        try:
            garden = await load_garden()
        finally:
            logger.remove(sink)

        assert set(garden) == {'a/one'}
        assert any('project-scoped' in m for m in messages)

    def test_the_sync_reader_refuses_inside_a_running_loop(self):
        import asyncio

        async def inside() -> None:
            with pytest.raises(RuntimeError, match='await load_garden'):
                load_garden_sync()

        asyncio.run(inside())

    @pytest.mark.asyncio
    async def test_reading_cards_without_a_key_says_why(self, monkeypatch):
        """A garden from an empty table would retire every seat, so the raw reader is always strict."""
        from evaluatorq.common.llm_client import MissingLLMCredentialsError

        monkeypatch.delenv('ORQ_API_KEY', raising=False)
        monkeypatch.setattr(model_catalogue, '_catalogues', {})

        with pytest.raises(MissingLLMCredentialsError):
            await model_catalogue.model_cards()

    @pytest.mark.asyncio
    async def test_raw_cards_share_the_parsed_tables_fetch_and_cache(self, monkeypatch):
        """One GET per process and credential serves both readers, so they describe one catalogue."""
        monkeypatch.setenv('ORQ_API_KEY', 'test-key')
        monkeypatch.setattr(model_catalogue, '_catalogues', {})
        calls: list[str] = []

        async def fetch(host, api_key, oauth_fetch):
            calls.append(host)
            return [card('a', 'one'), 'not a card']

        monkeypatch.setattr(model_catalogue, '_fetch_catalogue_payload', fetch)

        first = await model_catalogue.model_cards()
        second = await model_catalogue.model_cards()
        info = await model_catalogue.get_model_info('a/one')

        assert first is second
        assert [c['model_id'] for c in first] == ['one']
        assert info is not None
        assert len(calls) == 1
