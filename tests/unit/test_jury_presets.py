"""Jury preset definitions and the `llm_jury(preset=...)` seam (RES-1171, RES-1346).

Offline: prices and default efforts come from `tests.common.garden_fakes.seated_garden`,
real card rates for every seated router ID. The live recomputation against
`/v2/models` is in `tests/integration/test_preset_freshness.py`.
"""

from typing import Any, cast

import pytest

from evaluatorq import jury_presets, llm_jury
from evaluatorq.common.llm_client import MissingLLMCredentialsError
from evaluatorq.llm_jury import _apply_preset
from evaluatorq.jury_presets import (
    DROPPED_PRESETS,
    ESTIMATED_COMPLETION_TOKENS,
    ESTIMATED_PROMPT_TOKENS,
    PRESETS,
    WITHHELD_PRESETS,
    JuryPreset,
    all_router_ids,
    captured_at,
    get_preset,
    judge_family,
    judge_rates,
    model_identity,
)
from evaluatorq.common.model_frontier import with_manual_cards
from tests.common.garden_fakes import seated_garden


@pytest.fixture
def garden():
    return seated_garden()


def _preset(**overrides: Any) -> dict[str, Any]:
    base = {
        'name': 'Test Panel',
        'judges': ('openai/gpt-5.6-luna', 'anthropic/claude-opus-5-5', 'google/gemini-3.6-flash'),
        'reserve_judges': ('deepseek/deepseek-v4-pro',),
        'use_when': 'testing',
        'estimated_cost_per_1k': 1.0,
    }
    return base | overrides


class TestPanelInvariants:
    """The rules that apply to every preset, enforced at construction."""

    def test_even_panel_is_rejected(self):
        with pytest.raises(ValueError, match='panel size 2 is even'):
            JuryPreset(**_preset(judges=('openai/gpt-5.6-luna', 'google/gemini-3.6-flash')))

    def test_reserve_already_on_the_panel_is_rejected(self):
        """Substituting a seated judge shrinks the panel, which is the failure being avoided."""
        with pytest.raises(ValueError, match='reserve judge is already on the panel'):
            JuryPreset(**_preset(reserve_judges=('openai/gpt-5.6-luna',)))

    def test_duplicate_judge_is_rejected(self):
        """A repeated judge is one voter casting two votes, so majority stops meaning majority."""
        with pytest.raises(ValueError, match='duplicate judge'):
            JuryPreset(**_preset(judges=('openai/gpt-5.6-luna', 'openai/gpt-5.6-luna', 'google/gemini-3.6-flash')))

    def test_reserves_sharing_a_lineage_is_rejected(self):
        """Two reserves of one lineage cannot both be seated without recreating the clash."""
        with pytest.raises(ValueError, match='reserves share a lineage'):
            JuryPreset(**_preset(reserve_judges=('deepseek/deepseek-v4-pro', 'deepseek/deepseek-v4-flash')))

    def test_odd_panel_with_a_distinct_reserve_is_accepted(self):
        preset = JuryPreset(**_preset())

        assert len(preset.judges) % 2 == 1
        assert not set(preset.reserve_judges) & set(preset.judges)


class TestShippedPresets:
    """Properties every published preset must hold, not just the ones hand-checked."""

    @pytest.mark.parametrize('preset', PRESETS.values(), ids=lambda p: p.name)
    def test_panel_has_no_duplicate_judges(self, preset):
        assert len(set(preset.judges)) == len(preset.judges)

    def test_the_registry_cannot_be_edited_in_place(self):
        """PRESETS is process-wide, so a stray write would reseat every later caller."""
        with pytest.raises(TypeError):
            cast('dict[str, JuryPreset]', cast('object', PRESETS))['Balanced Trio'] = PRESETS['Strong Jury']

    def test_a_name_is_matched_after_trimming(self):
        assert get_preset('  Balanced Trio ') is PRESETS['Balanced Trio']

    def test_router_ids_cover_judges_and_reserves(self):
        ids = all_router_ids()

        for preset in PRESETS.values():
            assert set(preset.judges) <= ids
            assert set(preset.reserve_judges) <= ids

    def test_get_preset_names_the_alternatives(self):
        with pytest.raises(ValueError, match="available: \\['Balanced Trio'"):
            get_preset('No Such Panel')

    def test_a_retired_preset_answers_with_its_reason(self):
        """A user coming back to a name they used deserves the reason it went."""
        with pytest.raises(ValueError, match='was retired'):
            get_preset('Value Trio')

    def test_a_withheld_preset_says_what_is_holding_it(self, monkeypatch):
        """Derived and costed but not seated is a third state, and it is not 'unknown'."""
        monkeypatch.setitem(WITHHELD_PRESETS, 'Future Panel', 'Not shipped yet: its third seat cannot vote.')

        with pytest.raises(ValueError, match='is not shipped yet'):
            get_preset('Future Panel')

        for name in WITHHELD_PRESETS:
            assert name not in DROPPED_PRESETS
            assert name not in PRESETS

    def test_cheap_aggregate_ships(self):
        """Withheld while a MiniMax seat answered schemas in prose; the derived panel no longer seats one."""
        preset = get_preset('Cheap Aggregate')

        assert not any('minimax' in judge.lower() for judge in preset.judges)
        assert 'Cheap Aggregate' not in WITHHELD_PRESETS


class TestPublishedCost:
    """The $/1k in the table is recomputed from card rates, never trusted."""

    @pytest.mark.parametrize('preset', PRESETS.values(), ids=lambda p: p.name)
    def test_preset_cost_matches_its_published_figure(self, preset, garden):
        assert preset.cost_per_1k(garden) == preset.estimated_cost_per_1k

    @pytest.mark.parametrize('router_id', sorted(all_router_ids()))
    def test_every_preset_router_id_is_priced(self, router_id, garden):
        """A preset must never ship a judge that silently costs zero."""
        model = with_manual_cards(garden).get(router_id)

        assert model is not None
        assert model.input_price > 0
        assert model.output_price > 0

    def test_a_repriced_judge_moves_the_published_figure(self, garden):
        """The point of recomputing: a price change cannot leave the table standing."""
        preset = PRESETS['Balanced Trio']
        seat = preset.judges[0]
        repriced = garden | {seat: garden[seat].model_copy(update={'input_price': garden[seat].input_price * 2})}

        assert preset.cost_per_1k(repriced) != preset.estimated_cost_per_1k

    def test_an_unpriced_judge_raises_rather_than_costing_nothing(self, garden):
        preset = PRESETS['Balanced Trio']
        missing = {k: v for k, v in garden.items() if k != preset.judges[0]}

        with pytest.raises(KeyError, match='no live card for'):
            preset.cost_per_1k(missing)

    def test_the_estimate_is_three_input_tokens_to_one_output(self):
        """The ratio the garden's blended price ranks on, so cost and rank describe one workload."""
        assert (ESTIMATED_PROMPT_TOKENS, ESTIMATED_COMPLETION_TOKENS) == (1500, 500)

    def test_the_published_tokens_are_the_ones_the_figures_were_costed_at(self, garden):
        """The figures mean nothing without the item they are per, so tie the two together."""
        router_id = PRESETS['Balanced Trio'].judges[0]
        seat = garden[router_id]
        alone = JuryPreset(
            **_preset(
                judges=(router_id,),
                reserve_judges=PRESETS['Balanced Trio'].reserve_judges,
                estimated_cost_per_1k=0.0,
            )
        )
        expected = (
            seat.input_price * ESTIMATED_PROMPT_TOKENS + seat.output_price * ESTIMATED_COMPLETION_TOKENS
        ) / 1_000

        assert alone.cost_per_1k(garden) == pytest.approx(expected, abs=0.005)

    def test_seated_efforts_read_the_catalogue_default(self, garden):
        preset = PRESETS['Balanced Trio']
        seat = preset.judges[1]
        graded = garden | {seat: garden[seat].model_copy(update={'effort_parameter': True, 'default_effort': 'low'})}

        assert preset.seated_efforts(graded)[seat] == 'low'
        assert preset.seated_efforts(garden)[seat] is None


class TestDeprecatedRates:
    """The v1.86 snapshot readers keep working, now backed by the live garden."""

    def test_judge_rates_reads_the_garden_card(self, garden):
        with pytest.deprecated_call():
            rates = judge_rates('openai/gpt-6-sol', garden)
        assert rates is not None
        assert (rates.input_rate, rates.output_rate) == (2.0, 10.0)

    def test_judge_rates_is_none_without_a_card(self, garden):
        with pytest.deprecated_call():
            assert judge_rates('openai/not-a-model', garden) is None

    @pytest.mark.parametrize('error', [RuntimeError('loop running'), MissingLLMCredentialsError('no key')])
    def test_judge_rates_is_none_when_no_live_catalogue_can_be_read(self, monkeypatch, error):
        def unreadable():
            raise error

        monkeypatch.setattr(jury_presets, 'load_garden_sync', unreadable)
        with pytest.deprecated_call():
            assert judge_rates('openai/gpt-6-sol') is None

    def test_captured_at_is_an_iso_date(self):
        with pytest.deprecated_call():
            stamp = captured_at()
        assert len(stamp) == 10 and stamp[4] == '-' and stamp[7] == '-'


class TestModelIdentity:
    @pytest.mark.parametrize(
        ('router_id', 'identity'),
        [
            ('aws/us-east-1.openai.gpt-5.4', 'gpt-5.4'),
            ('aws/eu-central-2.anthropic.claude-haiku-5-5', 'claude-haiku-5-5'),
            ('aws/eu.anthropic.claude-haiku-4-5-20251001-v1:0', 'claude-haiku-4-5-20251001'),
            ('aws/openai.gpt-5.5', 'gpt-5.5'),
            ('openai/gpt-5.6-terra', 'gpt-5.6-terra'),
            ('azure/global.gpt-5.6-terra', 'gpt-5.6-terra'),
            ('aws/jp.anthropic.claude-haiku-5-5', 'claude-haiku-5-5'),
            ('munito/zai-org/glm-5.3-flash@eu', 'glm-5.3-flash'),
        ],
    )
    def test_host_region_vendor_and_version_are_stripped(self, router_id, identity):
        assert model_identity(router_id) == identity


class TestFamilyExclusion:
    """The exclusion rule needs a defined input to be implementable."""

    def test_lineage_beats_hosting_and_licence(self):
        """Self-preference follows who trained the model, not who serves it."""
        assert judge_family('groq/openai/gpt-oss-120b') == 'openai'
        assert judge_family('google/zai-org/glm-5-maas') == 'zhipu'
        assert judge_family('google/eu.claude-sonnet-5') == 'anthropic'
        assert judge_family('baseten/kimi-k3') == 'moonshot'

    def test_unknown_lineage_raises_rather_than_guessing(self):
        """An unclassified judge would never match a generator, silently voiding the rule."""
        with pytest.raises(ValueError, match='unknown lineage'):
            judge_family('someprovider/unheard-of-model-9')

    @pytest.mark.parametrize('preset', PRESETS.values(), ids=lambda p: p.name)
    def test_every_shipped_judge_and_reserve_classifies(self, preset):
        for router_id in (*preset.judges, *preset.reserve_judges):
            assert judge_family(router_id)

    @pytest.mark.parametrize('preset', PRESETS.values(), ids=lambda p: p.name)
    def test_duplicated_lineage_is_surfaced_not_hidden(self, preset):
        """A repeated lineage is allowed but must be visible, since its errors correlate."""
        for family, judges in preset.duplicated_lineages().items():
            assert len(judges) > 1
            assert all(judge_family(j) == family for j in judges)

    def test_single_provider_trio_reports_all_three_seats_as_one_lineage(self):
        """The whole point of the preset is one contract; the whole risk is one lineage."""
        duplicates = PRESETS['Single-Provider Trio'].duplicated_lineages()

        assert set(duplicates) == {'openai'}
        assert len(duplicates['openai']) == 3

    def test_the_other_presets_seat_one_judge_per_lineage(self):
        for name in (
            'Balanced Trio',
            'Strong Jury',
            'Cheap Aggregate',
            'Open-Weight / Portable',
            'EU Region',
        ):
            assert PRESETS[name].duplicated_lineages() == {}


class TestDroppedRegister:
    def test_a_dropped_preset_is_gone_and_its_reason_is_on_record(self):
        """Retirement is allowed; silent retirement is not."""
        assert 'Value Trio' in DROPPED_PRESETS
        for name, reason in DROPPED_PRESETS.items():
            assert name not in PRESETS, f'{name} is both dropped and shipped'
            assert len(reason) > 40, f'{name} needs a real reason, not a stub'


class TestLLMJurySeam:
    """What `preset=` fills in, and what it refuses to fill in over."""

    def test_a_preset_seats_its_panel(self):
        evaluator = llm_jury(name='helpfulness', criteria='is it helpful', preset='Balanced Trio')

        assert callable(evaluator['scorer'])

    def test_preset_and_judges_together_is_a_contradiction_not_an_override(self):
        with pytest.raises(ValueError, match='not both'):
            llm_jury(name='x', criteria='c', preset='Balanced Trio', judges=['openai/gpt-5.6-luna'])

    def test_preset_and_model_together_is_rejected(self):
        with pytest.raises(ValueError, match='not both'):
            llm_jury(name='x', criteria='c', preset='Balanced Trio', model='openai/gpt-5.6-luna')

    def test_a_rotation_cannot_run_a_panel_preset(self):
        """Cyclic scores each item with one judge, so there is no panel left to agree."""
        with pytest.raises(ValueError, match='cannot run with'):
            llm_jury(name='x', criteria='c', preset='Balanced Trio', assignment='cyclic')

    @pytest.mark.parametrize('name', sorted(PRESETS))
    def test_quorum_defaults_to_a_majority_of_the_seats(self, name):
        """A floor on how many judges must answer, not the aggregation threshold.

        Requiring every seat made one unreachable judge void an item the rest of
        the panel agreed on, which a live run hit on the first try. A majority
        is the smallest count that cannot be carried by a minority of the panel.
        """
        judges, aggregator, quorum = _applied(name)

        assert quorum == len(judges) // 2 + 1
        assert quorum > len(judges) / 2
        assert aggregator == 'majority'

    def test_a_trio_still_needs_two_judges(self):
        """Majority of three is two, so no preset can conclude on a lone judge."""
        judges, _, quorum = _applied('Balanced Trio')

        assert (len(judges), quorum) == (3, 2)

    def test_an_explicit_quorum_still_wins(self):
        from evaluatorq.llm_jury import _apply_preset

        _, _, quorum = _apply_preset(
            'Balanced Trio',
            judges=None,
            model=None,
            aggregator=None,
            min_successful_judges=3,
            verdict_kind='categorical',
            assignment='all',
        )

        assert quorum == 3

    def test_a_numeric_jury_takes_the_presets_numeric_rule(self):
        from evaluatorq.llm_jury import _apply_preset

        _, aggregator, _ = _apply_preset(
            'Balanced Trio',
            judges=None,
            model=None,
            aggregator=None,
            min_successful_judges=None,
            verdict_kind='numeric',
            assignment='all',
        )

        assert aggregator == 'mean_std'

    def test_no_preset_leaves_the_documented_defaults_alone(self):
        from evaluatorq.llm_jury import _apply_preset

        judges, aggregator, quorum = _apply_preset(
            None,
            judges=None,
            model=None,
            aggregator=None,
            min_successful_judges=None,
            verdict_kind='categorical',
            assignment='all',
        )

        assert (judges, aggregator, quorum) == (None, None, 1)


def _applied(preset: str) -> tuple[list[str], object, int]:
    from evaluatorq.llm_jury import _apply_preset

    judges, aggregator, quorum = _apply_preset(
        preset,
        judges=None,
        model=None,
        aggregator=None,
        min_successful_judges=None,
        verdict_kind='categorical',
        assignment='all',
    )
    assert judges is not None, f'{preset} seated no panel'
    return judges, aggregator, quorum


class TestDocsTable:
    """The published table is generated by hand, so it is checked by machine.

    A preset's whole promise is that the panel and the price you read are the
    panel and the price you get. A docs table nobody diffs is how that promise
    quietly stops being true.
    """

    @staticmethod
    def _rows() -> dict[str, tuple[tuple[str, ...], str, float, tuple[str, ...]]]:
        """Every column of the table, because a column nobody reads is a column that rots.

        This used to capture the judges and the cost and discard the rest, which
        is how three reseated reserves went unchecked.
        """
        import re
        from pathlib import Path

        page = Path(__file__).resolve().parents[2] / 'docs' / 'jury-presets.md'
        rows: dict[str, tuple[tuple[str, ...], str, float, tuple[str, ...]]] = {}
        for line in page.read_text(encoding='utf-8').splitlines():
            match = re.match(
                r'^\| \*\*(?P<name>[^*]+)\*\*[^|]*\|(?P<judges>[^|]+)\|(?P<aggregation>[^|]+)\|'
                r'\s*(?P<cost>[\d.]+)\s*\|(?P<reserve>[^|]*)\|',
                line,
            )
            if match:
                rows[match['name'].strip()] = (
                    tuple(re.findall(r'`([^`]+)`', match['judges'])),
                    match['aggregation'].strip(),
                    float(match['cost']),
                    tuple(re.findall(r'`([^`]+)`', match['reserve'])),
                )
        return rows

    def test_the_table_lists_every_preset_and_only_those(self):
        assert set(self._rows()) == set(PRESETS)

    @pytest.mark.parametrize('preset', PRESETS.values(), ids=lambda p: p.name)
    def test_the_table_names_the_judges_that_are_seated(self, preset):
        judges, _, _, _ = self._rows()[preset.name]

        assert judges == preset.judges

    @pytest.mark.parametrize('preset', PRESETS.values(), ids=lambda p: p.name)
    def test_the_table_publishes_the_cost_the_code_computes(self, preset, garden):
        _, _, cost, _ = self._rows()[preset.name]

        assert cost == preset.estimated_cost_per_1k == preset.cost_per_1k(garden)

    @pytest.mark.parametrize('preset', PRESETS.values(), ids=lambda p: p.name)
    def test_the_table_names_the_reserves_the_preset_holds(self, preset):
        _, _, _, reserve = self._rows()[preset.name]

        assert reserve == preset.reserve_judges

    @pytest.mark.parametrize('preset', PRESETS.values(), ids=lambda p: p.name)
    def test_the_table_states_the_aggregation_a_preset_applies(self, preset):
        _, aggregation, _, _ = self._rows()[preset.name]
        _, applied, _ = _apply_preset(
            preset.name,
            judges=None,
            model=None,
            aggregator=None,
            min_successful_judges=None,
            verdict_kind='categorical',
            assignment='all',
        )

        assert aggregation == applied


class TestAsyncAndMissingSeats:
    @pytest.fixture
    def preset(self):
        return jury_presets.BALANCED_TRIO

    @pytest.mark.asyncio
    async def test_async_methods_match_the_sync_ones_on_the_same_garden(self, preset):
        garden = seated_garden()
        assert await preset.cost_per_1k_async(garden) == preset.cost_per_1k(garden)
        assert await preset.seated_efforts_async(garden) == preset.seated_efforts(garden)

    @pytest.mark.asyncio
    async def test_async_methods_await_the_live_garden(self, preset, monkeypatch):
        async def live(client=None):
            return seated_garden()

        monkeypatch.setattr(jury_presets, 'load_garden', live)
        assert await preset.cost_per_1k_async() == preset.cost_per_1k(seated_garden())

    def test_a_seat_with_no_card_raises_in_both_methods(self, preset):
        with pytest.raises(KeyError):
            preset.seated_efforts({})
        with pytest.raises(KeyError):
            preset.cost_per_1k({})
