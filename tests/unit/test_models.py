"""`evaluatorq.models`: card parsing, dominance, the frontier, one-per-model, the CLI and the dashboard page."""

from __future__ import annotations

import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any, cast, get_args

import pytest
from typer.testing import CliRunner

from evaluatorq import models as models_mod
from evaluatorq.common import model_frontier
from evaluatorq.common.model_frontier import (
    INDEX_AXES,
    LIST_MIX,
    PRICE_MIXES,
    X_AXIS_SPECS,
    PriceMix,
    ScoredModel,
    available_efforts,
    available_indices,
    matches_vendor,
    model_vendor,
    parse_mix,
    scored_model_from_card,
    x_axis_spec,
)
from evaluatorq.models import (
    dominated_by,
    dominates,
    find_model,
    one_per_model,
    pareto,
    pareto_frontier,
    score_of,
    x_of,
)
from evaluatorq.dashboard import workspace_models
from tests.common.garden_fakes import card


_real_can_manage_models = workspace_models.can_manage_models


@pytest.fixture(autouse=True)
def _no_live_admin_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every /models page asks Orq whether the sign-in may switch models; a test that wants a yes says so."""
    async def not_admin(auth: object) -> bool:
        del auth
        return False

    monkeypatch.setattr(workspace_models, 'can_manage_models', not_admin)
    monkeypatch.setattr(workspace_models, '_admin_answers', {})


def model(
    router_id: str,
    index: float,
    input_price: float,
    output_price: float,
    *,
    enabled: bool = True,
    catalogue_id: str | None = None,
    **fields: Any,
) -> ScoredModel:
    return ScoredModel(
        router_id=router_id,
        intelligence_index=index,
        input_price=input_price,
        output_price=output_price,
        provider=router_id.split('/')[0],
        enabled=enabled,
        catalogue_id=catalogue_id or f'id-{router_id}',
        **fields,
    )


# Blended prices: strong 8.0, mid 2.0, cheap 0.5, bad 4.0 (dominated by mid), free 0.
STRONG = model('a/strong', 50.0, 5.0, 17.0)
MID = model('b/mid', 40.0, 1.0, 5.0)
CHEAP = model('c/cheap', 20.0, 0.25, 1.25)
BAD = model('d/bad', 35.0, 2.0, 10.0)
FREE = model('e/free', 60.0, 0.0, 0.0)


class TestCardParsing:
    def test_reads_categories_speed_and_estimate_flag(self) -> None:
        v2 = {
            'intelligence_index': 40.0,
            'aa_slug': 'm',
            'aa_index_is_estimated': True,
            'aa_evaluations': {
                'artificial_analysis_intelligence_index': 40.0,
                'artificial_analysis_legal_index': 33.5,
                'artificial_analysis_coding_index': None,
                'artificial_analysis_economics_index': 'n/a',
            },
            'aa_median_output_tokens_per_second': 120.0,
            'aa_median_time_to_first_token_seconds': 0.8,
        }
        parsed = scored_model_from_card(
            card('openai', 'm', id='uuid-1', model_developer='openai', display_name='M', metadata={'autorouter': {'v2': v2}})
        )
        assert parsed is not None
        assert parsed.catalogue_id == 'uuid-1'
        assert parsed.category_indices == {'legal': 33.5}
        assert parsed.is_estimated is True
        assert parsed.output_tokens_per_second == 120.0
        assert parsed.time_to_first_token_seconds == 0.8
        assert (parsed.developer, parsed.display_name, parsed.aa_slug) == ('openai', 'M', 'm')

    def test_malformed_evaluations_read_as_absent(self) -> None:
        v2 = {'intelligence_index': 40.0, 'aa_evaluations': ['not', 'a', 'mapping']}
        parsed = scored_model_from_card(card('openai', 'm', metadata={'autorouter': {'v2': v2}}))
        assert parsed is not None
        assert parsed.category_indices == {}
        assert parsed.is_estimated is False


class TestDominance:
    def test_cheaper_and_smarter_dominates(self) -> None:
        assert dominates(MID, BAD)
        assert not dominates(BAD, MID)

    def test_trade_off_is_not_dominance(self) -> None:
        assert not dominates(STRONG, MID)
        assert not dominates(MID, STRONG)

    def test_a_model_never_dominates_itself(self) -> None:
        assert not dominates(MID, MID)

    def test_free_models_have_no_place_on_price(self) -> None:
        assert x_of(FREE) is None
        assert not dominates(FREE, CHEAP)

    def test_higher_speed_is_better(self) -> None:
        fast = model('a/fast', 30.0, 1.0, 1.0, output_tokens_per_second=200.0)
        slow = model('b/slow', 30.0, 0.1, 0.1, output_tokens_per_second=50.0)
        assert dominates(fast, slow, x='output_speed')
        assert dominates(slow, fast, x='price')

    def test_lower_time_to_first_token_is_better(self) -> None:
        quick = model('a/quick', 30.0, 9.0, 9.0, time_to_first_token_seconds=0.4)
        laggy = model('b/laggy', 30.0, 0.1, 0.1, time_to_first_token_seconds=2.5)
        unmeasured = model('c/unmeasured', 90.0, 0.1, 0.1)
        assert dominates(quick, laggy, x='time_to_first_token')
        assert not dominates(laggy, quick, x='time_to_first_token')
        result = pareto([quick, laggy, unmeasured], x='time_to_first_token')
        assert [p.model.router_id for p in result.frontier] == ['a/quick']
        assert result.points[1].dominated_by == 'a/quick'
        assert result.points[1].score_per_usd is None
        assert [(e.router_id, e.reason) for e in result.excluded] == [('c/unmeasured', 'no time to first token data')]

    def test_unknown_index_is_refused_for_a_pair(self) -> None:
        with pytest.raises(ValueError, match='unknown index'):
            dominates(MID, BAD, index='vibes')  # ty: ignore[invalid-argument-type]


def _old_dominates(a: ScoredModel, b: ScoredModel) -> bool:
    """`ScoredModel.dominates` as it was before it delegated to the shared kernel."""
    if a.free or b.free:
        return False
    at_least_as_good = a.default_index >= b.default_index and a.price <= b.price
    return at_least_as_good and (a.default_index > b.default_index or a.price < b.price)


def _old_frontier(models: list[ScoredModel]) -> list[ScoredModel]:
    """`common.model_frontier.pareto_frontier` as it was before it read the shared kernel."""
    comparable = [m for m in models if not m.free]
    frontier = [m for m in comparable if not any(_old_dominates(o, m) for o in comparable)]
    return sorted(frontier, key=lambda m: -m.default_index)


class TestKernelAgreement:
    """Seating (router pools, juries, in-family upgrades) reads one kernel; it must decide as the old rule did."""

    @pytest.mark.parametrize('seed', range(5))
    def test_seating_rule_matches_the_old_one(self, seed: int) -> None:
        rng = random.Random(seed)
        garden = []
        for i in range(40):
            # A coarse grid forces ties on both axes; a few free cards and effort ladders ride along.
            index = rng.choice([10.0, 20.0, 20.0, 30.0, 40.0])
            ladder = {'low': index - 5, 'high': index + 5} if rng.random() < 0.3 else {}
            garden.append(
                model(
                    f'p{i % 4}/m{i}',
                    index,
                    rng.choice([0.0, 0.5, 1.0, 1.0, 2.0]),
                    rng.choice([0.0, 1.0, 2.0, 2.0, 4.0]),
                    effort_parameter=bool(ladder),
                    default_effort=rng.choice(['low', 'high']) if ladder else None,
                    effort_indices=ladder,
                )
            )
        for a in garden:
            for b in garden:
                assert a.dominates(b) is _old_dominates(a, b)
                assert dominates(a, b) is _old_dominates(a, b)
        assert model_frontier.pareto_frontier(garden) == _old_frontier(garden)
        # The public helper and the view agree with seating on the same blend.
        assert {m.router_id for m in pareto_frontier(garden)} == {
            p.model.router_id for p in pareto(garden, min_score=None).frontier
        }


class TestAxisTable:
    def test_one_row_per_literal_value(self) -> None:
        assert tuple(INDEX_AXES) == get_args(models_mod.Index)
        assert tuple(X_AXIS_SPECS) == get_args(models_mod.XAxis)

    def test_direction_and_format(self) -> None:
        assert [k for k, s in X_AXIS_SPECS.items() if s.lower_is_better] == ['price', 'time_to_first_token']
        assert X_AXIS_SPECS['price'].format(0.4375) == '0.438'
        assert X_AXIS_SPECS['output_speed'].format(None) == '—'

    def test_available_indices_and_efforts_come_from_the_models(self) -> None:
        laddered = model('a/l', 30.0, 1.0, 1.0, effort_indices={'high': 40.0, 'low': 20.0}, category_indices={'legal': 3.0})
        flat = model('b/f', 30.0, 1.0, 1.0, headline_effort='medium')
        assert available_indices([laddered, flat]) == ['intelligence', 'legal']
        assert available_indices([]) == ['intelligence']
        assert available_efforts([laddered, flat]) == ['low', 'medium', 'high']


class TestPareto:
    def test_frontier_and_nearest_dominator(self) -> None:
        result = pareto([STRONG, MID, CHEAP, BAD, FREE])
        assert [p.model.router_id for p in result.frontier] == ['a/strong', 'b/mid', 'c/cheap']
        [bad] = result.dominated
        assert bad.dominated_by == 'b/mid'
        assert bad.dominators == ['b/mid']
        assert [e.router_id for e in result.excluded] == ['e/free']

    def test_score_per_usd_is_score_per_dollar(self) -> None:
        point = next(p for p in pareto([MID], mix='list', min_score=0).points)
        assert point.score_per_usd == pytest.approx(40.0 / 2.0)
        # The default blend: no cache rate on MID, so reads bill at input: (7*1 + 2*1 + 1*5) / 10 = 1.4.
        assert pareto([MID], min_score=0).points[0].score_per_usd == pytest.approx(40.0 / 1.4)

    def test_score_per_usd_subtracts_the_floor(self) -> None:
        point = next(p for p in pareto([MID], mix='list', min_score=10).points)
        assert point.score_per_usd == pytest.approx((40.0 - 10) / 2.0)

    def test_score_per_usd_is_zero_at_the_floor(self) -> None:
        point = next(p for p in pareto([MID], mix='list', min_score=40.0).points)
        assert point.score_per_usd == 0.0

    def test_nearest_dominator_is_the_cheapest_frontier_model_at_least_as_smart(self) -> None:
        weak = model('z/weak', 10.0, 10.0, 10.0)
        assert dominated_by(weak, [STRONG, MID, CHEAP]) == CHEAP

    def test_dominated_dominator_is_not_named(self) -> None:
        # BAD dominates WORSE, but MID dominates BAD: the answer is the frontier model.
        worse = model('z/worse', 30.0, 3.0, 11.0)
        assert dominated_by(worse, [MID, BAD]) == MID

    def test_effort_reads_that_rung_and_excludes_models_without_it(self) -> None:
        laddered = model(
            'a/l', 30.0, 1.0, 1.0, effort_parameter=True, default_effort='low', effort_indices={'low': 30.0, 'high': 45.0}
        )
        flat = model('b/f', 40.0, 1.0, 1.0)
        assert score_of(laddered) == 30.0
        assert score_of(laddered, effort='high') == 45.0
        result = pareto([laddered, flat], effort='high')
        assert [p.model.router_id for p in result.points] == ['a/l']
        assert result.excluded[0].reason == 'no intelligence score at effort high'

    def test_domain_index_axis(self) -> None:
        legal = model('a/l', 30.0, 1.0, 1.0, category_indices={'legal': 50.0})
        result = pareto([legal, MID], index='legal')
        assert [p.score for p in result.points] == [50.0]
        assert result.excluded[0].router_id == 'b/mid'

    def test_domain_index_refuses_an_effort(self) -> None:
        with pytest.raises(ValueError, match='one effort Artificial Analysis benchmarked'):
            pareto([MID], index='legal', effort='high')

    def test_unknown_axis_is_refused(self) -> None:
        with pytest.raises(ValueError, match='unknown x axis'):
            pareto([MID], x='vibes')  # ty: ignore[invalid-argument-type]

    def test_exclude_estimated_drops_estimates(self) -> None:
        guess = model('z/guess', 99.0, 0.1, 0.1, is_estimated=True)
        result = pareto([MID, guess], exclude_estimated=True)
        assert [p.model.router_id for p in result.points] == ['b/mid']
        assert result.excluded[0].reason == 'estimated score'
        assert pareto_frontier([MID, guess]) == [guess]

    def test_default_floor_excludes_low_scorers_with_the_reason(self) -> None:
        junk = model('z/junk', 2.7, 0.05, 0.05)
        result = pareto([STRONG, MID, CHEAP, junk])
        assert result.min_score == model_frontier.MIN_SEAT_INDEX == 10.0
        assert [p.model.router_id for p in result.frontier] == ['a/strong', 'b/mid', 'c/cheap']
        [gone] = result.excluded
        assert gone.router_id == 'z/junk'
        assert gone.reason == 'score 2.7 is below min_score 10'
        assert gone.below_min_score is True
        assert result.below_min_score == [gone]

    def test_floor_zero_keeps_every_model(self) -> None:
        junk = model('z/junk', 2.7, 0.05, 0.05)
        result = pareto([STRONG, MID, CHEAP, junk], min_score=0)
        assert [p.model.router_id for p in result.frontier] == ['a/strong', 'b/mid', 'c/cheap', 'z/junk']
        assert result.excluded == []

    def test_floor_applies_to_the_chosen_domain_index(self) -> None:
        weak_legal = model('a/wl', 50.0, 1.0, 1.0, category_indices={'legal': 9.0})
        strong_legal = model('b/sl', 10.0, 1.0, 1.0, category_indices={'legal': 30.0})
        result = pareto([weak_legal, strong_legal], index='legal')
        assert [p.model.router_id for p in result.points] == ['b/sl']
        assert result.excluded[0].reason == 'score 9 is below min_score 10'

    def test_missing_axis_reason_wins_over_the_floor(self) -> None:
        free_junk = model('z/free', 2.0, 0.0, 0.0)
        [gone] = pareto([free_junk]).excluded
        assert (gone.reason, gone.below_min_score) == ('free (no list price)', False)

    def test_dominated_by_ignores_the_floor(self) -> None:
        junk = model('z/junk', 2.7, 0.5, 0.5)
        dearer_junk = model('z/dear', 2.0, 9.0, 9.0)
        assert dominated_by(dearer_junk, [junk, STRONG]) == junk

    def test_dominated_by_applies_no_floor_to_a_negative_score(self) -> None:
        negative = model('n/neg', -1.0, 9.0, 9.0)
        assert dominated_by(negative, [MID]) == MID

    def test_floor_none_keeps_a_negative_score(self) -> None:
        negative = model('n/neg', -1.0, 0.01, 0.01)
        result = pareto([MID, negative], min_score=None)
        assert result.min_score is None
        assert [p.model.router_id for p in result.points] == ['b/mid', 'n/neg']

    @pytest.mark.parametrize('floor', [float('nan'), float('inf'), float('-inf')])
    def test_non_finite_floor_is_refused(self, floor: float) -> None:
        with pytest.raises(ValueError, match='min_score must be a finite number'):
            pareto([MID], min_score=floor)

    def test_a_repeated_model_is_placed_once(self) -> None:
        result = pareto([MID, MID, BAD, model('b/mid', 1.0, 9.0, 9.0)])
        assert [(p.model.router_id, p.on_frontier) for p in result.points] == [('b/mid', True), ('d/bad', False)]
        assert result.points[0].model is MID


# Opus 5.5's card rates: input 4, output 20, cache read 0.2, cache write 5 (USD per 1M, 2026-11-20).
OPUS = model('anthropic/opus', 48.0, 4.0, 20.0, cache_read_price=0.2, cache_write_price=5.0, aa_cache_read_price=0.2)


class TestPriceMix:
    def test_agentic_is_the_artificial_analysis_blend(self) -> None:
        # (7 * 0.2 + 2 * 4 + 1 * 20) / 10
        assert PRICE_MIXES['agentic'].price(OPUS) == pytest.approx(2.94)
        assert x_of(OPUS) == OPUS.price == pytest.approx(2.94)
        assert x_of(OPUS, mix='list') == LIST_MIX.price(OPUS) == 8.0

    @pytest.mark.parametrize('seed', range(3))
    def test_agentic_mix_is_the_seating_price_float_for_float(self, seed: int) -> None:
        rng = random.Random(seed)
        for _ in range(200):
            m = model('a/m', 30.0, rng.uniform(0.01, 30), rng.uniform(0.01, 90), cache_read_price=rng.uniform(0, 1))
            assert PRICE_MIXES['agentic'].price(m) == m.price

    def test_explicit_ratio_with_a_write_share(self) -> None:
        mix = parse_mix('7:1:1:1')
        assert (mix.read, mix.write, mix.fresh, mix.output, mix.name) == (7, 1, 1, 1, None)
        # (7 * 0.2 + 1 * 5 + 1 * 4 + 1 * 20) / 10
        assert mix.price(OPUS) == pytest.approx(3.04)

    def test_names_and_equivalent_ratios_resolve_to_the_named_blend(self) -> None:
        assert parse_mix('agentic') is PRICE_MIXES['agentic']
        assert parse_mix(' List ') is PRICE_MIXES['list']
        assert parse_mix('7:0:2:1') is PRICE_MIXES['agentic']
        assert parse_mix('70:0:20:10') is PRICE_MIXES['agentic']
        assert parse_mix('0:0:3:1') is PRICE_MIXES['list']
        custom = PriceMix(read=1, fresh=1, output=1)
        assert parse_mix(custom) is custom

    @pytest.mark.parametrize(
        ('text', 'message'),
        [
            ('cheap', 'unknown price mix'),
            ('7:2:1', 'unknown price mix'),
            ('7:x:2:1', 'not a number'),
            ('7:-1:2:1', 'negative or non-finite'),
            ('inf:0:2:1', 'negative or non-finite'),
            ('nan:0:2:1', 'negative or non-finite'),
            ('0:0:0:0', 'no non-zero share'),
        ],
    )
    def test_bad_mixes_are_refused(self, text: str, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_mix(text)

    def test_a_mix_built_directly_is_validated_too(self) -> None:
        with pytest.raises(ValueError):
            PriceMix(read=-1, fresh=1, output=1)
        with pytest.raises(ValueError):
            PriceMix(fresh=0, output=0)

    def test_labels_name_the_blend(self) -> None:
        assert x_axis_spec('price', PRICE_MIXES['agentic']).short_label == 'Price $/1M (7:2:1 cached)'
        assert x_axis_spec('price', PRICE_MIXES['list']).short_label == 'Price $/1M (3:1 list)'
        custom = parse_mix('7:1:1:1')
        assert x_axis_spec('price', custom).short_label == 'Price $/1M (7:1:1:1)'
        assert x_axis_spec('price', custom).label == 'Price, USD per 1M tokens (7:1:1:1 read:write:fresh:output)'
        assert x_axis_spec('output_speed', custom) is X_AXIS_SPECS['output_speed']
        assert pareto([MID]).x_spec.label == 'Price, USD per 1M tokens (7:2:1 cached)'


class TestCacheFallbacks:
    def test_host_rate_wins(self) -> None:
        assert OPUS.cache_read_rate() == (0.2, 'host')
        assert OPUS.cache_write_rate() == (5.0, 'host')

    def test_artificial_analysis_price_when_the_host_lists_none(self) -> None:
        berget = model('berget/kimi', 46.5, 3.0, 12.0, aa_cache_read_price=0.2)
        assert berget.cache_read_rate() == (0.2, 'artificial_analysis')
        # (7 * 0.2 + 2 * 3 + 12) / 10
        assert PRICE_MIXES['agentic'].price(berget) == pytest.approx(1.94)

    def test_artificial_analysis_price_is_capped_at_the_host_input_rate(self) -> None:
        cheap_host = model('h/m', 30.0, 0.1, 0.4, aa_cache_read_price=0.5)
        assert cheap_host.cache_read_rate() == (0.1, 'artificial_analysis')

    def test_input_rate_when_the_card_has_no_cache_price(self) -> None:
        assert MID.cache_read_rate() == (1.0, 'input')
        assert MID.cache_write_rate() == (1.0, 'input')

    def test_points_carry_the_source_and_the_result_lists_fallbacks(self) -> None:
        aa = model('berget/kimi', 46.5, 3.0, 12.0, aa_cache_read_price=0.2)
        result = pareto([OPUS, aa, MID])
        sources = {p.model.router_id: (p.cache_read_source, p.cache_write_source) for p in result.points}
        assert sources == {
            'anthropic/opus': ('host', None),
            'berget/kimi': ('artificial_analysis', None),
            'b/mid': ('input', None),
        }
        assert {p.model.router_id for p in result.price_fallbacks} == {'berget/kimi', 'b/mid'}

    def test_write_share_reports_the_write_source(self) -> None:
        result = pareto([OPUS, MID], mix='7:1:1:1')
        sources = {p.model.router_id: (p.cache_read_source, p.cache_write_source) for p in result.points}
        assert sources == {'anthropic/opus': ('host', 'host'), 'b/mid': ('input', 'input')}

    def test_no_source_without_a_cache_share_or_off_the_price_axis(self) -> None:
        assert all(p.cache_read_source is None and not p.price_fallback for p in pareto([MID], mix='list').points)
        assert pareto([MID], mix='list').price_fallbacks == []
        timed = model('a/t', 30.0, 1.0, 1.0, time_to_first_token_seconds=0.5)
        result = pareto([timed], x='time_to_first_token')
        assert result.mix is None
        assert result.points[0].cache_read_source is None

    def test_card_rates_come_from_the_catalogue_pricing_parser(self) -> None:
        v2 = {
            'intelligence_index': 40.0,
            'aa_pricing': {'price_1m_cache_hit_tokens': 0.25, 'price_1m_cache_write_tokens': None},
        }
        metadata = {
            'autorouter': {'v2': v2},
            'million_tokens_cache_read_cost': 0.1,
            'million_tokens_cache_write_cost': 1.25,
        }
        parsed = scored_model_from_card(card('anthropic', 'm', metadata=metadata))
        assert parsed is not None
        assert (parsed.cache_read_price, parsed.cache_write_price, parsed.aa_cache_read_price) == (0.1, 1.25, 0.25)

    def test_missing_or_malformed_cache_fields_read_as_absent(self) -> None:
        v2 = {'intelligence_index': 40.0, 'aa_pricing': {'price_1m_cache_hit_tokens': 'n/a'}}
        metadata = {'autorouter': {'v2': v2}, 'million_tokens_cache_read_cost': -1}
        parsed = scored_model_from_card(card('openai', 'm', metadata=metadata))
        assert parsed is not None
        assert (parsed.cache_read_price, parsed.cache_write_price, parsed.aa_cache_read_price) == (None, None, None)


class TestSeatingUsesTheAgenticBlend:
    def test_cache_rates_change_seating_and_the_view_alike(self) -> None:
        # Same 3:1 price; only the cache rate differs, so the cached model is cheaper everywhere by default.
        cached = model('a/cached', 40.0, 2.0, 8.0, cache_read_price=0.1)
        uncached = model('b/uncached', 41.0, 2.0, 8.0)
        assert LIST_MIX.price(cached) == LIST_MIX.price(uncached)
        assert cached.price < uncached.price
        assert not uncached.dominates(cached)
        assert not dominates(uncached, cached)
        assert dominates(uncached, cached, mix='list')
        both = ['b/uncached', 'a/cached']
        assert [m.router_id for m in model_frontier.pareto_frontier([cached, uncached])] == both
        assert [p.model.router_id for p in pareto([cached, uncached]).frontier] == both
        assert [p.model.router_id for p in pareto([cached, uncached], mix='list').frontier] == ['b/uncached']


class TestVendor:
    def test_one_key_for_host_and_developer_spellings(self) -> None:
        gemini = model('google-ai/gemini', 40.0, 1.0, 1.0, developer='Google')
        assert model_vendor(gemini) == 'google'
        assert matches_vendor(gemini, ['google'])
        assert matches_vendor(gemini, ['google-ai'])
        assert not matches_vendor(gemini, ['openai', ''])

    def test_host_or_developer_matches(self) -> None:
        azure_gpt = model('azure/gpt-x', 40.0, 1.0, 1.0, developer='openai')
        assert matches_vendor(azure_gpt, ['azure'])
        assert matches_vendor(azure_gpt, ['OpenAI'])


class TestSummary:
    def test_summary_is_the_model_and_the_point(self) -> None:
        [point] = pareto([MID], mix='list').points
        row = point.summary()
        assert set(ScoredModel.model_fields) <= set(row)
        for key in (
            'price',
            'score',
            'x',
            'score_per_usd',
            'on_frontier',
            'dominated_by',
            'dominators',
            'cache_read_source',
            'cache_write_source',
        ):
            assert key in row
        assert 'model' not in row
        assert row['price'] == MID.price
        assert row['score_per_usd'] == pytest.approx((40.0 - 10) / 2.0)
        json.dumps(row)


class TestOnePerModel:
    def test_prefers_the_developer_listing(self) -> None:
        own = model('openai/gpt-x', 40.0, 2.0, 8.0, developer='openai', aa_slug='gpt-x')
        azure = model('azure/gpt-x', 40.0, 1.0, 4.0, developer='openai', aa_slug='gpt-x')
        assert one_per_model([azure, own]) == [own]

    def test_falls_back_to_the_cheapest_host(self) -> None:
        a = model('h1/kimi', 40.0, 2.0, 8.0, developer='moonshotai', aa_slug='kimi')
        b = model('h2/kimi', 40.0, 1.0, 4.0, developer='moonshotai', aa_slug='kimi')
        assert one_per_model([a, b]) == [b]

    def test_google_ai_counts_as_google(self) -> None:
        own = model('google-ai/gemini', 40.0, 2.0, 8.0, developer='google', aa_slug='g')
        other = model('vertex/gemini', 40.0, 1.0, 4.0, developer='google', aa_slug='g')
        assert one_per_model([other, own]) == [own]


class TestLoadModels:
    @pytest.mark.asyncio
    async def test_keeps_contributor_listings_beside_the_standard_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_cards(client: object = None) -> list[dict[str, Any]]:
            v2 = {'intelligence_index': 48.0, 'aa_slug': 'muse'}
            meta = {'region': 'us', 'autorouter': {'v2': v2}}
            return [
                card('meta', 'muse-spark-1.3', model_developer='meta', metadata=meta),
                card('meta', 'muse-spark-1.3-contributor', model_developer='meta', input_cost=0.0001, metadata=meta),
            ]

        monkeypatch.setattr(model_frontier, 'model_cards', fake_cards)
        loaded = await models_mod.load_models()
        assert [m.router_id for m in loaded] == ['meta/muse-spark-1.3', 'meta/muse-spark-1.3-contributor']

    def test_pareto_excludes_contributor_listings_with_the_reason(self) -> None:
        contributor = model('meta/muse-contributor', 48.0, 0.1, 0.2)
        result = pareto([MID, contributor], min_score=None)
        assert [p.model.router_id for p in result.points] == [MID.router_id]
        [gone] = result.contributors
        assert gone.router_id == 'meta/muse-contributor'
        assert 'train on your prompts' in gone.reason


class TestFindModel:
    def test_exact_then_unique_bare_id(self) -> None:
        pool = [MID, model('x/mid', 1.0, 1.0, 1.0), CHEAP]
        assert find_model(pool, model_id='b/mid') == MID
        assert find_model(pool, model_id='cheap') == CHEAP
        assert find_model(pool, model_id='mid') is None  # two hosts share the bare id
        assert find_model(pool, model_id='nope') is None

    def test_three_part_router_id(self) -> None:
        coder = model('tensorix/qwen/qwen3-coder-30b-a3b-instruct', 9.6, 0.05, 0.2)
        pool = [MID, coder, CHEAP]
        assert find_model(pool, model_id='tensorix/qwen/qwen3-coder-30b-a3b-instruct') == coder
        assert find_model(pool, model_id='qwen/qwen3-coder-30b-a3b-instruct') == coder
        assert find_model(pool, model_id='qwen3-coder-30b-a3b-instruct') == coder

    def test_ambiguous_short_ids_are_refused(self) -> None:
        a = model('tensorix/qwen/qwen3-coder', 9.6, 0.05, 0.2)
        b = model('other/qwen/qwen3-coder', 9.6, 0.06, 0.2)
        c = model('deepinfra/qwen3-coder', 9.6, 0.07, 0.2)
        assert find_model([a, b], model_id='qwen/qwen3-coder') is None
        assert find_model([a, b], model_id='qwen3-coder') is None
        # The id after the provider decides before the last segment is tried.
        assert find_model([a, c], model_id='qwen3-coder') == c
        assert find_model([a, c], model_id='qwen/qwen3-coder') == a


@pytest.fixture
def fake_catalogue(monkeypatch: pytest.MonkeyPatch) -> list[ScoredModel]:
    pool = [STRONG, MID, CHEAP, BAD, FREE]

    async def fake_load(client: object = None, *, include_disabled: bool = False, all_hosts: bool = False) -> list[ScoredModel]:
        del client, include_disabled, all_hosts
        return pool

    monkeypatch.setattr(models_mod, 'load_models', fake_load)
    return pool


_ANSI = re.compile(r'\x1b\[[0-9;]*m')


def _flat(output: str) -> str:
    """Rich wraps a usage error inside a box and colours it on CI; read it as one plain line."""
    return ' '.join(_ANSI.sub('', output).replace('│', ' ').split())


class TestCli:
    def _run(self, *args: str, columns: int | None = None) -> Any:
        from evaluatorq.cli import _register_subapps, app

        _register_subapps(app)
        env = {'COLUMNS': str(columns)} if columns else None
        return CliRunner().invoke(app, ['models', 'pareto', *args], env=env)

    def test_json_frontier(self, fake_catalogue: list[ScoredModel]) -> None:
        result = self._run('--json')
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert [p['router_id'] for p in payload['points']] == ['a/strong', 'b/mid', 'c/cheap']
        assert payload['min_score'] == 10.0
        assert payload['excluded'] == [{'router_id': 'e/free', 'reason': 'free (no list price)', 'below_min_score': False, 'contributor': False}]

    def test_min_score_hides_and_zero_shows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        junk = model('z/junk', 2.7, 0.05, 0.05)

        async def fake_load(client: object = None, *, include_disabled: bool = False, all_hosts: bool = False) -> list[ScoredModel]:
            del client, include_disabled, all_hosts
            return [STRONG, MID, junk]

        monkeypatch.setattr(models_mod, 'load_models', fake_load)
        table = self._run()
        assert table.exit_code == 0, table.output
        assert 'z/junk' not in table.stdout
        assert '1 model(s) hidden for scoring below --min-score 10. Pass --min-score 0 to show every model.' in _flat(table.stdout)
        assert 'not compared' not in table.stdout
        shown = json.loads(self._run('--min-score', '0', '--json').stdout)
        assert 'z/junk' in [p['router_id'] for p in shown['points']]
        assert shown['excluded'] == []
        named = self._run('--model', 'junk')
        assert 'z/junk is not plotted: score 2.7 is below min_score 10.' in _flat(named.output)

    def test_negative_min_score_is_a_usage_error(self, fake_catalogue: list[ScoredModel]) -> None:
        assert self._run('--min-score', '-1').exit_code == 2

    def test_dominated_table_names_the_winner(self, fake_catalogue: list[ScoredModel]) -> None:
        result = self._run('--dominated')
        assert result.exit_code == 0, result.output
        assert 'd/bad' in result.stdout
        assert 'b/mid' in result.stdout

    def test_model_filter_shows_a_dominated_model_without_the_flag(self, fake_catalogue: list[ScoredModel]) -> None:
        payload = json.loads(self._run('--model', 'bad', '--json').stdout)
        assert [(p['router_id'], p['dominated_by']) for p in payload['points']] == [('d/bad', 'b/mid')]

    def test_unknown_model_is_a_usage_error(self, fake_catalogue: list[ScoredModel]) -> None:
        result = self._run('--model', 'nope')
        assert result.exit_code == 2

    def test_bad_index_is_a_usage_error(self, fake_catalogue: list[ScoredModel]) -> None:
        assert self._run('--index', 'vibes').exit_code == 2

    def test_index_no_loaded_model_has_is_a_usage_error(self, fake_catalogue: list[ScoredModel]) -> None:
        result = self._run('--index', 'economics')
        assert result.exit_code == 2
        assert 'no model in the catalogue has a economics score' in _flat(result.output)

    def test_coding_and_agentic_are_not_indices(self, fake_catalogue: list[ScoredModel]) -> None:
        assert 'coding' not in model_frontier.INDICES
        assert 'agentic' not in model_frontier.INDICES
        assert 'choose one of' in _flat(self._run('--index', 'coding').output)

    def test_effort_no_loaded_model_has_is_a_usage_error(self, fake_catalogue: list[ScoredModel]) -> None:
        result = self._run('--effort', 'turbo')
        assert result.exit_code == 2
        assert "no model in the catalogue has a score at effort 'turbo'" in _flat(result.output)

    def test_effort_a_loaded_model_has_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        laddered = model('a/l', 30.0, 1.0, 1.0, effort_indices={'high': 45.0})

        async def fake_load(client: object = None, *, include_disabled: bool = False, all_hosts: bool = False) -> list[ScoredModel]:
            del client, include_disabled, all_hosts
            return [laddered, MID]

        monkeypatch.setattr(models_mod, 'load_models', fake_load)
        payload = json.loads(self._run('--effort', 'high', '--json').stdout)
        assert [(p['router_id'], p['score']) for p in payload['points']] == [('a/l', 45.0)]

    def test_provider_filter_matches_host_or_developer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        azure_gpt = model('azure/gpt-x', 40.0, 1.0, 1.0, developer='openai')
        gemini = model('google-ai/gemini', 40.0, 1.0, 1.0, developer='google')

        async def fake_load(client: object = None, *, include_disabled: bool = False, all_hosts: bool = False) -> list[ScoredModel]:
            del client, include_disabled, all_hosts
            return [azure_gpt, gemini, MID]

        monkeypatch.setattr(models_mod, 'load_models', fake_load)
        payload = json.loads(self._run('--provider', 'openai', '--provider', 'google', '--json').stdout)
        assert sorted(p['router_id'] for p in payload['points']) == ['azure/gpt-x', 'google-ai/gemini']

    def test_default_mix_is_cache_aware_and_list_is_one_flag_away(self, fake_catalogue: list[ScoredModel]) -> None:
        default = self._run(columns=200)
        assert default.exit_code == 0, default.output
        assert 'Price $/1M (7:2:1 cached)' in default.stdout
        assert 'Cache read $/1M' in default.stdout
        assert '(input)' in default.stdout  # the fakes list no cache rate
        listed = self._run('--mix', 'list', columns=200)
        assert 'Price $/1M (3:1 list)' in listed.stdout
        assert 'Cache read $/1M' not in listed.stdout

    def test_json_names_the_mix_and_the_rate_source(self, fake_catalogue: list[ScoredModel]) -> None:
        payload = json.loads(self._run('--json', '--mix', '7:1:1:1').stdout)
        assert payload['mix']['read'] == 7.0
        assert (payload['mix']['write'], payload['mix']['name']) == (1.0, None)
        assert payload['x_label'] == 'Price, USD per 1M tokens (7:1:1:1 read:write:fresh:output)'
        mid = next(p for p in payload['points'] if p['router_id'] == 'b/mid')
        assert (mid['cache_read_source'], mid['cache_write_source']) == ('input', 'input')
        assert mid['x'] == pytest.approx((7 + 1 + 1 + 5) / 10)

    def test_bad_mix_and_mix_off_the_price_axis_are_usage_errors(self, fake_catalogue: list[ScoredModel]) -> None:
        bad = self._run('--mix', '7:2:1')
        assert bad.exit_code == 2
        assert 'unknown price mix' in _flat(bad.output)
        off_axis = self._run('--x', 'output_speed', '--mix', 'list')
        assert off_axis.exit_code == 2
        assert 'a price mix applies only to --x price' in _flat(off_axis.output)

    def test_model_is_found_and_compared_outside_the_provider_filter(self, fake_catalogue: list[ScoredModel]) -> None:
        result = self._run('--provider', 'b', '--model', 'd/bad', '--json')
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert [(p['router_id'], p['dominated_by']) for p in payload['points']] == [('d/bad', 'b/mid')]
        assert payload['left_out_by_provider'] == 3  # a/strong, c/cheap and e/free

    def test_table_counts_the_models_the_provider_filter_left_out(self, fake_catalogue: list[ScoredModel]) -> None:
        result = self._run('--provider', 'b')
        assert result.exit_code == 0, result.output
        assert '4 model(s) from other providers left out by --provider.' in _flat(result.stdout)

    @pytest.mark.parametrize('floor', ['nan', 'inf'])
    def test_non_finite_min_score_is_a_usage_error(self, fake_catalogue: list[ScoredModel], floor: str) -> None:
        result = self._run('--min-score', floor)
        assert result.exit_code == 2
        assert 'must be a finite number' in _flat(result.output)

    def test_table_goes_to_stdout_with_the_axis_label(self, monkeypatch: pytest.MonkeyPatch) -> None:
        timed = model('a/t', 30.0, 1.0, 1.0, time_to_first_token_seconds=0.75)

        async def fake_load(client: object = None, *, include_disabled: bool = False, all_hosts: bool = False) -> list[ScoredModel]:
            del client, include_disabled, all_hosts
            return [timed, MID]

        monkeypatch.setattr(models_mod, 'load_models', fake_load)
        result = self._run('--x', 'time_to_first_token')
        assert result.exit_code == 0, result.output
        assert 'TTFT s' in result.stdout
        assert '0.75' in result.stdout
        assert 'a/t' in result.stdout


class TestDashboardPage:
    @pytest.fixture
    def client(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
        from starlette.testclient import TestClient

        from evaluatorq.dashboard import app as app_module
        from evaluatorq.dashboard.app import build_app

        monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'dashboard-settings.json'))
        monkeypatch.setenv('EVALUATORQ_USER_SETTINGS', str(tmp_path / 'user-settings.json'))
        monkeypatch.setattr(app_module, 'list_oauth_sessions', lambda: ())
        return TestClient(build_app(roots=[tmp_path]), follow_redirects=False)

    def test_without_a_key_the_page_says_how_to_connect(self, client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('ORQ_API_KEY', raising=False)
        response = client.get('/models')
        assert response.status_code == 200
        assert 'models-alert' in response.text
        assert 'ORQ_API_KEY is not set' in response.text
        assert 'data-needs-orq="1"' in response.text

    def test_contributor_listing_shows_as_excluded_with_the_reason(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        from evaluatorq.dashboard import models_views

        monkeypatch.setenv('ORQ_API_KEY', 'test-key')
        contributor = model('meta/muse-contributor', 48.0, 0.1, 0.2, display_name='Muse Contributor')

        async def fake_load(client: object = None, *, include_disabled: bool = False) -> list[ScoredModel]:
            del client, include_disabled
            return [*fake_catalogue, contributor]

        monkeypatch.setattr(models_views, 'load_models', fake_load)
        text = client.get('/models').text
        assert 'Excluded · 1' in text
        assert 'is-excluded' in text and 'Muse Contributor' in text
        assert 'title="contributor listing: the discount is paid for' in text
        assert 'Excluded ·' not in client.get('/models?frontier=1').text

    def test_renders_chart_table_and_dominator(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        from evaluatorq.dashboard import models_views

        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        async def fake_load(client: object = None, *, include_disabled: bool = False) -> list[ScoredModel]:
            del client, include_disabled
            return fake_catalogue

        monkeypatch.setattr(models_views, 'load_models', fake_load)
        response = client.get('/models?sort=score')
        assert response.status_code == 200
        text = response.text
        assert 'data-vega-for="models-pareto"' in text
        assert '<span>3 on the frontier</span><span>1 dominated</span>' in text
        assert '1 not plotted' in text
        assert 'title="b/mid"' in text  # d/bad's nearest dominator, named with its gap
        assert 'e/free' in text  # listed under "not plotted"

    def _models_text(self, client: Any, monkeypatch: pytest.MonkeyPatch, pool: list[ScoredModel], path: str) -> str:
        from evaluatorq.dashboard import models_views

        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        async def fake_load(client: object = None, *, include_disabled: bool = False) -> list[ScoredModel]:
            del client, include_disabled
            return pool

        monkeypatch.setattr(models_views, 'load_models', fake_load)
        return client.get(path).text

    def _as_admin(self, monkeypatch: pytest.MonkeyPatch, *, admin: bool) -> list[tuple[str, bool]]:
        """Sign the page in as someone who may (or may not) manage models; returns the switches made."""
        from evaluatorq.dashboard import models_views
        from evaluatorq.dashboard.auth import DashboardAuth

        calls: list[tuple[str, bool]] = []

        async def fake_can(auth: object) -> bool:
            del auth
            return admin

        async def fake_set(auth: object, model_id: str, *, enabled: bool) -> None:
            del auth
            calls.append((model_id, enabled))

        signed_in = DashboardAuth(method='environment', api_key='test-key', base_url='https://my.orq.ai')
        monkeypatch.setattr(models_views, '_auth', lambda req: (signed_in, None))
        monkeypatch.setattr(workspace_models, 'can_manage_models', fake_can)
        monkeypatch.setattr(models_views, 'set_model_enabled', fake_set)
        return calls

    def test_admins_get_an_on_off_switch(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        self._as_admin(monkeypatch, admin=True)
        off = model('f/off', 45.0, 1.5, 6.0, enabled=False, catalogue_id='uuid-off')
        text = self._models_text(client, monkeypatch, [*fake_catalogue, off], '/models?disabled=1')
        assert text.count('aria-checked="false"') == 1 and 'aria-checked="true"' in text
        assert '>disabled<' not in text  # the switch shows the state; no tag beside it
        assert 'name="model_id" value="uuid-off"' in text

    def test_switches_need_admin(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        self._as_admin(monkeypatch, admin=False)
        off = model('f/off', 45.0, 1.5, 6.0, enabled=False, catalogue_id='uuid-off')
        text = self._models_text(client, monkeypatch, [*fake_catalogue, off], '/models?disabled=1')
        assert '>disabled<' in text and 'models-switch' not in text
        workspace_models._admin_answers.clear()  # a new sign-in
        self._as_admin(monkeypatch, admin=True)
        on = model('f/on', 45.0, 1.5, 6.0, enabled=True, catalogue_id='uuid-on')
        assert 'models-switch' in self._models_text(client, monkeypatch, [*fake_catalogue, on], '/models?x=price')

    def test_a_failed_admin_check_hides_the_switches_instead_of_failing_the_page(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        self._as_admin(monkeypatch, admin=True)

        async def broken(auth: object) -> bool:
            raise RuntimeError(auth)

        monkeypatch.setattr(workspace_models, 'can_manage_models', broken)
        on = model('f/on', 45.0, 1.5, 6.0, enabled=True, catalogue_id='uuid-on')
        text = self._models_text(client, monkeypatch, [*fake_catalogue, on], '/models?x=price')
        assert 'models-page' in text and 'models-switch' not in text

    def test_a_switch_drops_the_process_catalogue_too(self, client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """An API-key sign-in also caches /v2/models for the process; a reload after a switch must refetch it."""
        from evaluatorq.dashboard.security import CSRF_FIELD, csrf_token

        self._as_admin(monkeypatch, admin=True)
        forgotten: list[object] = []
        monkeypatch.setattr('evaluatorq.dashboard.models_views.forget_catalogue', forgotten.append)
        client.post('/models/toggle', data={'model_id': 'f/off', 'action': 'enable', CSRF_FIELD: csrf_token()})
        assert len(forgotten) == 1

    def test_forget_catalogue_drops_only_that_sign_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from evaluatorq.common import model_catalogue

        source = ('https://my.orq.ai', 'mine')
        monkeypatch.setattr(model_catalogue, '_catalogue_source', lambda client: (None, None, None, source))
        other = ('https://my.orq.ai', 'other')
        for key in (source, other):
            monkeypatch.setitem(model_catalogue._raw_catalogues, key, ())
            monkeypatch.setitem(model_catalogue._catalogues, key, {})
        model_catalogue.forget_catalogue(None)
        assert source not in model_catalogue._raw_catalogues and source not in model_catalogue._catalogues
        assert other in model_catalogue._raw_catalogues

    def test_toggle_route_switches_for_admins_only(self, client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from evaluatorq.dashboard.security import CSRF_FIELD, csrf_token

        enable = {'model_id': 'f/off', 'action': 'enable', CSRF_FIELD: csrf_token()}
        calls = self._as_admin(monkeypatch, admin=False)
        assert 'Only workspace admins' in client.post('/models/toggle', data=enable).text
        assert not calls
        workspace_models._admin_answers.clear()  # a "no" is remembered for a minute; this is a new sign-in
        calls = self._as_admin(monkeypatch, admin=True)
        assert 'Stale or missing form token' in client.post('/models/toggle', data={'model_id': 'f/off'}).text
        assert not calls
        text = client.post('/models/toggle', data=enable).text
        assert 'aria-checked="true"' in text and 'models-toggle-note' not in text
        text = client.post('/models/toggle', data={'model_id': 'f/off', 'action': 'disable', CSRF_FIELD: csrf_token()}).text
        assert 'aria-checked="false"' in text and 'models-toggle-note' not in text
        assert calls == [('f/off', True), ('f/off', False)]

    def test_a_visit_without_filters_goes_back_to_the_last_ones(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        assert 'models-page' in self._models_text(client, monkeypatch, fake_catalogue, '/models')  # nothing saved yet
        self._models_text(client, monkeypatch, fake_catalogue, '/models?x=latency&mix=list')
        bare = client.get('/models', follow_redirects=False)
        assert bare.status_code == 303 and bare.headers['location'] == '/models?x=latency&mix=list'
        self._models_text(client, monkeypatch, fake_catalogue, '/models?x=price')
        assert client.get('/models', follow_redirects=False).headers['location'] == '/models?x=price'
        reset = client.get('/models?reset=1', follow_redirects=False)
        assert reset.headers['location'] == '/models' and 'eq_models_query=""' in reset.headers['set-cookie']
        client.cookies.set('eq_models_query', 'javascript:alert(1)//', path='/models')
        assert client.get('/models', follow_redirects=False).headers['location'].startswith('/models?')

    def test_the_admin_answer_is_reused_across_pages_and_switches(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        from evaluatorq.dashboard.security import CSRF_FIELD, csrf_token

        self._as_admin(monkeypatch, admin=True)
        checks: list[object] = []

        async def counting_can(auth: object) -> bool:
            checks.append(auth)
            return True

        monkeypatch.setattr(workspace_models, 'can_manage_models', counting_can)
        for path in ('/models?disabled=1', '/models?disabled=1&x=latency'):
            self._models_text(client, monkeypatch, fake_catalogue, path)
        client.post('/models/toggle', data={'model_id': 'f/off', 'action': 'enable', CSRF_FIELD: csrf_token()})
        assert len(checks) == 1

    def test_oauth_cards_are_reused_until_forgotten(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A filter change re-reads the catalogue; an OAuth read is kept a minute, and a switch drops it."""
        from evaluatorq.common import model_catalogue

        fetches: list[int] = []

        class OAuthClient:
            async def get_model_catalogue(self) -> list[dict[str, object]]:
                fetches.append(1)
                return [card('a', 'm')]

        oauth = OAuthClient()
        monkeypatch.setattr(model_catalogue, 'resolve_results_base_url', lambda client: 'https://my.orq.ai')
        first = asyncio.run(model_catalogue.model_cards(oauth))  # ty: ignore[invalid-argument-type]
        assert asyncio.run(model_catalogue.model_cards(oauth)) is first  # ty: ignore[invalid-argument-type]
        built: list[int] = []

        def build(cards: tuple[dict[str, object], ...]) -> int:
            built.append(1)
            return len(cards)

        assert model_catalogue.derived_from_cards(first, build) == model_catalogue.derived_from_cards(first, build) == 1
        assert fetches == [1] and built == [1]
        model_catalogue.forget_catalogue(oauth)  # ty: ignore[invalid-argument-type]
        second = asyncio.run(model_catalogue.model_cards(oauth))  # ty: ignore[invalid-argument-type]
        assert second is not first and fetches == [1, 1]
        model_catalogue.derived_from_cards(second, build)
        assert built == [1, 1]

    def test_toggle_route_shows_why_orq_refused(self, client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from evaluatorq.dashboard import models_views
        from evaluatorq.dashboard.security import CSRF_FIELD, csrf_token
        from evaluatorq.dashboard.workspace_models import WorkspaceModelError

        self._as_admin(monkeypatch, admin=True)

        async def refuse(auth: object, model_id: str, *, enabled: bool) -> None:
            raise WorkspaceModelError('HTTP 404 Not Found: Model not found')

        monkeypatch.setattr(models_views, 'set_model_enabled', refuse)
        text = client.post('/models/toggle', data={'model_id': 'x/y', 'action': 'enable', CSRF_FIELD: csrf_token()}).text
        assert 'models-toggle-note' in text and 'Model not found' in text and 'aria-checked="false"' in text

    def test_cards_name_the_best_score_and_the_best_score_per_dollar(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        tiny = model('z/tiny', 10.5, 0.05, 0.05)
        text = self._models_text(client, monkeypatch, [*fake_catalogue, tiny], '/models')
        assert text.count('class="stat-tile models-card') == 3
        assert 'Best score' in text and '50.0' in text and 'a/strong' in text
        # c/cheap: (20 - 10) / $0.35 blended under the default mix = 28.6, shown as 29.
        assert 'per $' in text
        assert re.search(r'Highest intelligence per \$</div><svg[^>]*>.*?</svg></div><div class="stat-value">29<span class="stat-unit"> per \$</span>', text)
        # b/mid sits above the line from c/cheap to a/strong, so the third card is the knee, not the cheapest model.
        cards = text.split('models-cards', 1)[1].split('models-intro', 1)[0]
        assert 'Balanced' in cards and 'b/mid' in cards and 'Cheapest' not in cards
        # Each card sketches the frontier and marks its own model once.
        assert cards.count('models-card-spark') == 3 and cards.count('class="is-mark"') == 3

    def test_balanced_card_names_the_knee_when_the_cheapest_is_the_best_value(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        text = self._models_text(client, monkeypatch, fake_catalogue, '/models')
        cards = text.split('models-cards', 1)[1].split('models-intro', 1)[0]
        assert cards.count('class="stat-tile models-card') == 3
        assert 'Balanced' in cards and 'b/mid' in cards and 'Cheapest' not in cards

    def test_balanced_is_the_frontier_model_furthest_above_the_line(self) -> None:
        from types import SimpleNamespace

        from evaluatorq.dashboard.models_views import _balanced

        def pt(score: float, x: float) -> Any:
            return SimpleNamespace(score=score, x=x, score_per_usd=1.0)

        value, best = pt(20, 0.1), pt(50, 10)
        # On log price the line from value to best passes 35 at $1 and 42.5 at $3.16.
        low, knee, flat = pt(30, 0.3), pt(45, 1.0), pt(47, 3.16)
        result = cast(Any, SimpleNamespace(x='price', frontier=[value, low, knee, flat, best]))
        assert _balanced(result, best, value) is knee
        # A frontier that bends the other way has no knee, so the page shows the cheapest card instead.
        sagging = cast(Any, SimpleNamespace(x='price', frontier=[value, pt(25, 1.0), best]))
        assert _balanced(sagging, best, value) is None
        assert _balanced(cast(Any, SimpleNamespace(x='output_speed', frontier=[value, knee, best])), best, value) is None

    def test_speed_axis_shows_no_score_per_dollar_card(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        pool = [
            model(m.router_id, m.intelligence_index or 0, 1.0, 1.0, output_tokens_per_second=100.0 - i * 10)
            for i, m in enumerate(fake_catalogue)
        ]
        text = self._models_text(client, monkeypatch, pool, '/models?x=output_speed')
        assert 'models-cards' in text
        assert 'per $' not in text
        assert 'Fastest' in text

    def test_guides_are_off_until_the_toggle_is_ticked(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        assert 'guide_label' not in self._models_text(client, monkeypatch, fake_catalogue, '/models')
        assert 'guide_label' in self._models_text(client, monkeypatch, fake_catalogue, '/models?guides=1')

    def test_guide_rows_follow_score_equals_floor_plus_k_x(self, fake_catalogue: list[ScoredModel]) -> None:
        from evaluatorq.dashboard.models_views import _guide_rows

        result = pareto(fake_catalogue, min_score=10)
        rows = _guide_rows(result, 10.0)
        xs = [p.x for p in result.points]
        scores = [p.score for p in result.points]
        # Score per $ spans 6.45 (a/strong) to 28.6 (c/cheap), so the 1-2-5 steps strictly inside are 10 and 20.
        assert sorted({r['k'] for r in rows}, key=float) == ['10', '20']
        for r in rows:
            assert r['score'] == pytest.approx(10.0 + float(r['k']) * r['x'], abs=1e-9)
            assert min(xs) - 1e-9 <= r['x'] <= max(xs) + 1e-9
            assert min(scores) <= r['score'] <= max(scores)

    @pytest.mark.parametrize(
        ('points', 'floor'),
        [
            pytest.param([(1.0, 30.0)], 10.0, id='single-point'),
            pytest.param([(1.0, 30.0), (2.0, 50.0)], 10.0, id='equal-score-per-dollar'),
            pytest.param([(0.5, 50.0), (2.0, 90.0)], 50.0, id='a-point-at-the-floor'),
            pytest.param([(0.1, 30.0), (4.0, 60.0)], 0.0, id='floor-zero'),
        ],
    )
    def test_guide_rows_edge_cases(self, points: list[tuple[float, float]], floor: float) -> None:
        from types import SimpleNamespace

        from evaluatorq.dashboard.models_views import _guide_rows

        stubs = [SimpleNamespace(x=x, score=s, score_per_usd=(s - floor) / x) for x, s in points]
        rows = _guide_rows(cast(Any, SimpleNamespace(x='price', points=stubs)), floor)
        positive = [p.score_per_usd for p in stubs if p.score_per_usd > 0]
        for r in rows:
            assert 'e' not in r['k']
            assert float(r['k'].replace(',', '')) > min(positive)

    def test_a_model_at_the_floor_does_not_stretch_the_guides(self) -> None:
        from types import SimpleNamespace

        from evaluatorq.dashboard.models_views import _guide_rows

        # Score per $ 0 (at the floor of 50), 5 and 20. Counting the 0 stretched the steps down six decades, to flat
        # lines on the floor labelled 1e-05 per $; only 10 lies strictly inside the positive range.
        stubs = [
            SimpleNamespace(x=1.0, score=50.0, score_per_usd=0.0),
            SimpleNamespace(x=0.5, score=52.5, score_per_usd=5.0),
            SimpleNamespace(x=2.0, score=90.0, score_per_usd=20.0),
        ]
        rows = _guide_rows(cast(Any, SimpleNamespace(x='price', points=stubs)), 50.0)
        assert {r['k'] for r in rows} == {'10'}

    def test_no_guides_off_the_price_axis(self, fake_catalogue: list[ScoredModel]) -> None:
        from evaluatorq.dashboard.models_views import _guide_rows

        pool = [
            model(m.router_id, m.intelligence_index or 0, 1.0, 1.0, output_tokens_per_second=100.0 - i * 10)
            for i, m in enumerate(fake_catalogue)
        ]
        assert _guide_rows(pareto(pool, x='output_speed', min_score=10), 10.0) == []

    def test_table_has_a_score_per_dollar_column_on_price_only(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        text = self._models_text(client, monkeypatch, fake_catalogue, '/models?sort=value')
        head = text.split('<thead>', 1)[1].split('</thead>', 1)[0]
        assert 'Score per $' in head
        # Right after the price (x) column, before the input / output rates.
        assert head.index('sort=x') < head.index('Score per $') < head.index('sort=rates')
        body = text.split('</thead>', 1)[1]
        # Highest Score per $ first: c/cheap (28.6), b/mid (21.4), d/bad (8.9), a/strong (6.5).
        order = [body.index(f'title="{rid}"') for rid in ('c/cheap', 'b/mid', 'd/bad', 'a/strong')]
        assert order == sorted(order)
        assert '<td class="num">28.6</td>' in body and '<td class="num">21.4</td>' in body
        pool = [
            model(m.router_id, m.intelligence_index or 0, 1.0, 1.0, output_tokens_per_second=100.0 - i * 10)
            for i, m in enumerate(fake_catalogue)
        ]
        assert 'Score per $' not in self._models_text(client, monkeypatch, pool, '/models?x=output_speed')

    def test_no_cards_without_a_frontier(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        text = self._models_text(client, monkeypatch, fake_catalogue, '/models?min_score=100')
        assert 'models-cards' not in text

    def test_tooltip_names_the_estimate_flag(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        from evaluatorq.dashboard import models_views

        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        async def fake_load(client: object = None, *, include_disabled: bool = False) -> list[ScoredModel]:
            del client, include_disabled
            return fake_catalogue

        monkeypatch.setattr(models_views, 'load_models', fake_load)
        text = client.get('/models').text
        assert 'Measured or estimated' in text

    def test_a_catalogue_failure_asks_for_a_retry_not_a_login(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from evaluatorq.dashboard import models_views
        from evaluatorq.models import ModelCatalogueError

        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        async def failing_load(client: object = None, *, include_disabled: bool = False) -> list[ScoredModel]:
            del client, include_disabled
            raise ModelCatalogueError('catalogue returned 503')

        monkeypatch.setattr(models_views, 'load_models', failing_load)
        response = client.get('/models')
        assert response.status_code == 200
        assert 'The Orq model catalogue could not be loaded: catalogue returned 503' in response.text
        assert 'Reload the page to try again.' in response.text
        assert 'to connect an Orq account' not in response.text

    def test_an_unexpected_failure_is_logged_and_shown_not_a_500(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from loguru import logger

        from evaluatorq.dashboard import models_views

        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        async def broken_load(client: object = None, *, include_disabled: bool = False) -> list[ScoredModel]:
            del client, include_disabled
            raise RuntimeError('client refused to start')

        monkeypatch.setattr(models_views, 'load_models', broken_load)
        logged: list[str] = []
        sink = logger.add(lambda message: logged.append(str(message)), level='ERROR')
        try:
            response = client.get('/models')
        finally:
            logger.remove(sink)
        assert response.status_code == 200
        assert 'The Pareto Frontier page failed with an unexpected RuntimeError.' in response.text
        assert 'The dashboard log has the details.' in response.text
        assert any('The /models page failed to load the catalogue' in line and 'client refused to start' in line for line in logged)

    def test_domain_index_disables_the_effort_select(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        from evaluatorq.dashboard import models_views

        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        async def fake_load(client: object = None, *, include_disabled: bool = False) -> list[ScoredModel]:
            del client, include_disabled
            return fake_catalogue

        fake_catalogue.append(model('f/legal', 30.0, 1.0, 1.0, category_indices={'legal': 50.0}))
        monkeypatch.setattr(models_views, 'load_models', fake_load)
        text = client.get('/models?index=legal&effort=high').text
        assert 'title="Domain indices exist only at the one effort Artificial Analysis benchmarked"' in text
        assert '<span class="models-field-off" inert>' in text
        assert '<span>1 on the frontier</span><span>0 dominated</span>' in text
        assert '5 not plotted' in text

    def test_min_score_control_hides_and_zero_shows(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        from evaluatorq.dashboard import models_views

        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        async def fake_load(client: object = None, *, include_disabled: bool = False) -> list[ScoredModel]:
            del client, include_disabled
            return fake_catalogue

        fake_catalogue.append(model('z/junk', 2.7, 0.05, 0.05))
        monkeypatch.setattr(models_views, 'load_models', fake_load)
        text = client.get('/models').text
        assert 'name="min_score" min="0" max="100" step="1" value="10"' in text
        assert '<span>3 on the frontier</span><span>1 dominated</span>' in text
        assert '2 not plotted' in text
        assert '1 below score 10</b><br>z/junk' in text
        text = client.get('/models?min_score=0').text
        assert 'value="0"' in text
        assert '<span>4 on the frontier</span><span>1 dominated</span>' in text
        assert '1 not plotted' in text
        assert 'value="10"' in client.get('/models?min_score=nope').text

    def test_index_dropdown_lists_only_indices_a_model_has(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        from evaluatorq.dashboard import models_views

        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        async def fake_load(client: object = None, *, include_disabled: bool = False) -> list[ScoredModel]:
            del client, include_disabled
            return fake_catalogue

        fake_catalogue.append(model('f/legal', 30.0, 1.0, 1.0, category_indices={'legal': 50.0}))
        monkeypatch.setattr(models_views, 'load_models', fake_load)
        text = client.get('/models?index=coding').text
        assert 'data-model="legal"' in text
        assert 'data-model="coding"' not in text
        assert 'data-model="economics"' not in text
        assert 'name="index" value="intelligence"' in text  # an empty index falls back to the default

    def test_mix_control_switches_the_blend_and_marks_fallbacks(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        from evaluatorq.dashboard import models_views

        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        async def fake_load(client: object = None, *, include_disabled: bool = False) -> list[ScoredModel]:
            del client, include_disabled
            return fake_catalogue

        fake_catalogue.append(OPUS)
        monkeypatch.setattr(models_views, 'load_models', fake_load)
        text = client.get('/models').text
        assert 'name="mix" value="agentic"' in text
        assert 'Agentic: 7:2:1 cached' in text
        assert 'Price $/1M (7:2:1 cached)' in text
        assert 'Cache read $/1M' in text
        assert 'No cache price on this card' in text
        assert '4 priced with a cache rate their host does not list' in text
        listed = client.get('/models?mix=list').text
        assert 'name="mix" value="list"' in listed
        assert 'Price $/1M (3:1 list)' in listed
        assert 'No cache price on this card' not in listed
        custom = client.get('/models?mix=7:1:1:1').text
        assert 'name="mix" value="7:1:1:1"' in custom
        assert 'Custom: 7:1:1:1' in custom
        assert 'name="mix" value="agentic"' in client.get('/models?mix=nope').text
        off_price = client.get('/models?x=output_speed').text
        assert 'title="The mix applies only to the price axis"' in off_price
        assert '<span class="models-field-off" inert>' in off_price
        assert '<span class="models-field-off" inert>' not in text

    def test_several_labs_narrow_to_their_union(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        text = self._models_text(client, monkeypatch, fake_catalogue, '/models?lab=a&lab=b')
        assert '<span>2 on the frontier</span><span>0 dominated</span>' in text
        assert 'c/cheap' not in text and 'd/bad' not in text
        assert '<span class="filter-dd-value is-engaged">2 selected</span>' in text
        # The sort links keep every chosen lab, not only the last one.
        assert 'lab=a&amp;lab=b' in text
        one = self._models_text(client, monkeypatch, fake_catalogue, '/models?lab=a')
        assert '<span>1 on the frontier</span><span>0 dominated</span>' in one
        client.cookies.clear()  # else the bare address goes back to lab=a
        every = self._models_text(client, monkeypatch, fake_catalogue, '/models')
        assert '<span class="filter-dd-value">All</span>' in every
        # Every box ticked is the same as no lab filter, so the sort links do not carry the whole list.
        ticked = '&'.join(f'lab={v}' for v in 'abcde')
        all_ticked = self._models_text(client, monkeypatch, fake_catalogue, f'/models?{ticked}')
        assert '<span class="filter-dd-value">All</span>' in all_ticked
        assert 'lab=a' not in all_ticked
        assert 'class="facet-search"' in every  # the lab list is searchable

    def test_including_disabled_models_with_every_lab_ticked_keeps_the_new_labs(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        from evaluatorq.dashboard import models_views

        self._as_admin(monkeypatch, admin=False)
        monkeypatch.setenv('ORQ_API_KEY', 'test-key')
        hidden = model('z/off', 45.0, 0.1, 0.1)

        async def fake_load(client: object = None, *, include_disabled: bool = False) -> list[ScoredModel]:
            del client
            return [*fake_catalogue, hidden] if include_disabled else fake_catalogue

        monkeypatch.setattr(models_views, 'load_models', fake_load)
        # The toolbar drops the lab values when every box is ticked (dashboard.js), so the request carries none.
        text = client.get('/models?disabled=1').text
        assert 'value="z" checked' in text and 'z/off' in text
        assert '<span class="filter-dd-value">All</span>' in text
        js = (Path(models_views.__file__).parent / 'static' / 'dashboard.js').read_text()
        assert "delete evt.detail.parameters.lab" in js

    def test_min_score_readout_is_plain_at_the_page_default(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        default = self._models_text(client, monkeypatch, fake_catalogue, '/models')
        assert 'data-default="10"' in default
        assert '<span class="filter-slider-readout">≥ 10</span>' in default
        zero = self._models_text(client, monkeypatch, fake_catalogue, '/models?min_score=0')
        assert '<span class="filter-slider-readout is-engaged">≥ 0</span>' in zero
        assert '<span class="filter-dd-more-badge">1</span>' in zero

    def test_single_model_frontier_on_a_speed_axis_shows_one_card(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        only = model('a/only', 40.0, 1.0, 1.0, output_tokens_per_second=80.0)
        text = self._models_text(client, monkeypatch, [only], '/models?x=output_speed')
        assert text.count('class="stat-tile models-card') == 1
        assert 'Fastest' not in text.split('models-cards', 1)[1].split('models-intro', 1)[0]

    def test_score_per_dollar_card_keeps_a_decimal_below_ten(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # (10.4 - 10) / $1.00 = 0.4 per $; whole numbers would show it as 0.
        near_floor = model('a/near', 10.4, 1.0, 1.0)
        text = self._models_text(client, monkeypatch, [near_floor], '/models')
        assert re.search(r'Highest intelligence per \$</div><svg[^>]*>.*?</svg></div><div class="stat-value">0\.4<span class="stat-unit"> per \$</span>', text)

    def test_more_filters_labels_match_the_toolbar_labels(self) -> None:
        import tinycss2

        from evaluatorq.dashboard.styles import DASHBOARD_CSS

        declarations = {
            d.name: tinycss2.serialize(d.value).strip()
            for rule in tinycss2.parse_stylesheet(DASHBOARD_CSS, skip_whitespace=True, skip_comments=True)
            if rule.type == 'qualified-rule'
            and tinycss2.serialize(rule.prelude).strip() == '.models-filters .filter-dd-more-body .filter-label'
            for d in tinycss2.parse_declaration_list(rule.content, skip_whitespace=True, skip_comments=True)
            if d.type == 'declaration'
        }
        assert declarations.get('text-transform') == 'none'
        assert declarations.get('letter-spacing') == '0'

    def test_guide_steps_keep_every_1_2_5_step_on_a_real_range(self) -> None:
        from evaluatorq.dashboard.models_views import _guide_ks

        # The live catalogue's Score per $ runs from about 1.5 to 630: every step shows, 20 and 50 included.
        assert _guide_ks(1.5, 630) == pytest.approx([2, 5, 10, 20, 50, 100, 200, 500])

    def test_guide_steps_fall_back_to_eight_evenly_spaced(self) -> None:
        from evaluatorq.dashboard.models_views import _guide_ks

        # 0.002 to 500,000 holds 26 1-2-5 steps; every other step still leaves 13 (0.002 to 200,000), so eight are picked
        # evenly across those, ends included.
        ks = _guide_ks(0.0011, 1e6)
        assert len(ks) == 8
        assert ks == sorted(ks)
        assert ks[0] == pytest.approx(0.002) and ks[-1] == pytest.approx(200_000)
        assert all(0.0011 < k < 1e6 for k in ks)

    def test_more_filters_badge_counts_active_filters(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        text = self._models_text(client, monkeypatch, fake_catalogue, '/models?frontier=1&min_score=20')
        assert '<details id="filter-dd-more" class="filter-dd filter-dd-more">' in text
        assert '<span class="filter-dd-more-badge">2</span>' in text
        assert 'class="filter-chip is-active"' in text
        default = self._models_text(client, monkeypatch, fake_catalogue, '/models')
        assert 'filter-dd-more-badge' not in default  # the default score floor is not a filter the reader set

    def test_typing_in_a_search_box_does_not_submit_the_filters(
        self, client: Any, monkeypatch: pytest.MonkeyPatch, fake_catalogue: list[ScoredModel]
    ) -> None:
        # A search box fires a bubbling `change` on blur; submitting then would swap the page and lose the text.
        text = self._models_text(client, monkeypatch, fake_catalogue, '/models')
        assert 'onchange="if (!event.target.matches(\'.facet-search\')) this.requestSubmit()"' in text


class TestWorkspaceAdmin:
    """`is_workspace_admin` maps the signed-in account to its workspace roles."""

    @pytest.mark.parametrize(
        ('roles', 'expected'), [(['01J3HZ9VGNP5867SAGRPS8JT4X'], True), (['01J3HZA12M14T6Z82BZ35B170K'], False)]
    )
    def test_admin_role_on_the_signed_in_account(
        self, monkeypatch: pytest.MonkeyPatch, roles: list[str], *, expected: bool
    ) -> None:
        import asyncio

        from evaluatorq.common import cli_oauth

        monkeypatch.setattr(cli_oauth, 'oauth_subject', lambda server_url: {'user_id': 'acct-1'})
        users = {'users': [{'id': 'wsu-1', 'account_id': 'acct-1', 'roles': roles}, {'account_id': 'other', 'roles': []}]}

        async def fake_call(self: object, args: list[str], *a: object, **k: object) -> object:
            assert args == ['users', 'list']
            return users

        monkeypatch.setattr(cli_oauth.shutil, 'which', lambda name: '/bin/orq')
        monkeypatch.setattr(cli_oauth._CLI, 'call', fake_call)
        assert asyncio.run(cli_oauth.is_workspace_admin('https://my.orq.ai')) is expected

    def test_a_refused_users_list_is_not_an_admin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio

        from evaluatorq.common import cli_oauth

        monkeypatch.setattr(cli_oauth, 'oauth_subject', lambda server_url: {'user_id': 'acct-1'})

        async def refuse(self: object, args: list[str], *a: object, **k: object) -> object:
            raise cli_oauth.OrqCLIError('HTTP 403 Forbidden')

        monkeypatch.setattr(cli_oauth.shutil, 'which', lambda name: '/bin/orq')
        monkeypatch.setattr(cli_oauth._CLI, 'call', refuse)
        assert asyncio.run(cli_oauth.is_workspace_admin('https://my.orq.ai')) is False


class TestManagementKey:
    """An API key may manage models when Orq lets it; the probe is an empty enable that changes nothing."""

    def _auth(self) -> Any:
        from evaluatorq.dashboard.auth import DashboardAuth

        return DashboardAuth(method='environment', api_key='mk', base_url='https://my.orq.ai')

    @pytest.mark.parametrize(('status', 'expected'), [(200, True), (403, False), (401, False)])
    def test_admin_check_lists_the_workspace_users(
        self, monkeypatch: pytest.MonkeyPatch, status: int, *, expected: bool
    ) -> None:
        import asyncio

        import httpx

        from evaluatorq.dashboard import workspace_models

        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(status)

        real = httpx.AsyncClient
        monkeypatch.setattr(
            workspace_models.httpx, 'AsyncClient', lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
        )
        assert asyncio.run(_real_can_manage_models(self._auth())) is expected
        assert seen[0].method == 'GET' and seen[0].url.path == '/v3/users'
        assert seen[0].headers['authorization'] == 'Bearer mk'

    def test_disable_deletes_the_encoded_model_id_and_surfaces_refusals(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio

        import httpx

        from evaluatorq.dashboard import workspace_models

        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(204 if request.method == 'DELETE' else 403, text='requires permission')

        from orq_ai_sdk import Orq

        monkeypatch.setattr(
            workspace_models,
            'resolve_orq_client',
            lambda *, api_key, base_url: Orq(
                api_key=api_key,
                server_url=base_url,
                async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            ),
        )
        asyncio.run(workspace_models.set_model_enabled(self._auth(), 'openai/gpt-4o', enabled=False))
        assert seen[0].method == 'DELETE' and seen[0].url.raw_path == b'/v2/workspace-models/openai%2Fgpt-4o'
        with pytest.raises(workspace_models.WorkspaceModelError, match='403'):
            asyncio.run(workspace_models.set_model_enabled(self._auth(), 'openai/gpt-4o', enabled=True))
