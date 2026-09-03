"""Jury preset definitions and the `llm_jury(preset=...)` seam (RES-1171)."""

from typing import Any

import pytest

from evaluatorq import llm_jury
from evaluatorq.common.model_frontier import load_judge_pricing
from evaluatorq.jury_presets import (
    DEFAULT_PRESET,
    DROPPED_PRESETS,
    ESTIMATED_COMPLETION_TOKENS,
    ESTIMATED_PROMPT_TOKENS,
    PRESETS,
    Aggregation,
    JuryPreset,
    all_router_ids,
    get_preset,
    judge_family,
)


def _preset(**overrides: Any) -> dict[str, Any]:
    base = {
        'name': 'Test Panel',
        'judges': ('openai/gpt-5.6-luna', 'anthropic/claude-opus-5', 'google/gemini-3.6-flash'),
        'aggregation': Aggregation.MAJORITY,
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

    def test_three_of_five_aggregation_requires_five_judges(self):
        """The aggregation mode encodes a panel size, so it cannot outlive one."""
        with pytest.raises(ValueError, match='3-of-5 majority needs 5 judges'):
            JuryPreset(**_preset(aggregation=Aggregation.MAJORITY_3_OF_5))

    def test_odd_panel_with_a_distinct_reserve_is_accepted(self):
        preset = JuryPreset(**_preset())

        assert len(preset.judges) % 2 == 1
        assert not set(preset.reserve_judges) & set(preset.judges)


class TestShippedPresets:
    """Properties every published preset must hold, not just the ones hand-checked."""

    @pytest.mark.parametrize('preset', PRESETS.values(), ids=lambda p: p.name)
    def test_panel_has_no_duplicate_judges(self, preset):
        assert len(set(preset.judges)) == len(preset.judges)

    def test_default_preset_is_published(self):
        assert DEFAULT_PRESET.name in PRESETS

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


class TestPublishedCost:
    """The $/1k in the table is recomputed from the captured rates, never trusted."""

    @pytest.mark.parametrize('preset', PRESETS.values(), ids=lambda p: p.name)
    def test_preset_cost_matches_its_published_figure(self, preset):
        assert preset.cost_per_1k() == preset.estimated_cost_per_1k

    @pytest.mark.parametrize('router_id', sorted(all_router_ids()))
    def test_every_preset_router_id_is_priced(self, router_id):
        """A preset must never ship a judge that silently costs zero."""
        rates = load_judge_pricing().get(router_id)

        assert rates is not None
        assert rates['input_rate'] > 0
        assert rates['output_rate'] > 0

    def test_a_repriced_judge_moves_the_published_figure(self, monkeypatch):
        """The point of recomputing: a price change cannot leave the table standing."""
        pricing = dict(load_judge_pricing())
        preset = PRESETS['Balanced Trio']
        doubled = dict(pricing[preset.judges[0]])
        doubled['input_rate'] *= 2
        monkeypatch.setattr(
            'evaluatorq.jury_presets.load_judge_pricing',
            lambda: pricing | {preset.judges[0]: doubled},
        )

        assert preset.cost_per_1k() != preset.estimated_cost_per_1k

    def test_an_unpriced_judge_raises_rather_than_costing_nothing(self, monkeypatch):
        preset = PRESETS['Balanced Trio']
        monkeypatch.setattr('evaluatorq.jury_presets.load_judge_pricing', dict)

        with pytest.raises(KeyError, match='no captured pricing'):
            preset.cost_per_1k()

    def test_the_estimate_is_stated_in_tokens_the_docs_publish(self):
        """The figures mean nothing without the item they are per."""
        assert (ESTIMATED_PROMPT_TOKENS, ESTIMATED_COMPLETION_TOKENS) == (1500, 150)


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

    def test_quorum_defaults_to_a_majority_of_the_panel(self):
        """The count the panel's own aggregation needs, not every seat.

        Requiring all five made one unreachable judge void an item that the
        other four agreed on, which a live run hit on the first try.
        """
        judges, aggregator, quorum = _applied('Cheap Aggregate')

        assert len(judges) == 5
        assert quorum == 3
        assert aggregator == 'majority'

    def test_a_trio_still_needs_two_judges(self):
        """Majority of three is two, so no preset can conclude on a lone judge."""
        judges, _, quorum = _applied('Balanced Trio')

        assert (len(judges), quorum) == (3, 2)

    def test_an_explicit_quorum_still_wins(self):
        from evaluatorq.llm_jury import _apply_preset

        _, _, quorum = _apply_preset(
            'Cheap Aggregate',
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
    def _rows() -> dict[str, tuple[tuple[str, ...], float]]:
        import re
        from pathlib import Path

        page = Path(__file__).resolve().parents[2] / 'docs' / 'jury-presets.md'
        rows: dict[str, tuple[tuple[str, ...], float]] = {}
        for line in page.read_text().splitlines():
            match = re.match(
                r'^\| \*\*(?P<name>[^*]+)\*\*[^|]*\|(?P<judges>[^|]+)\|[^|]+\|\s*(?P<cost>[\d.]+)\s*\|', line
            )
            if match:
                judges = tuple(re.findall(r'`([^`]+)`', match['judges']))
                rows[match['name'].strip()] = (judges, float(match['cost']))
        return rows

    def test_the_table_lists_every_preset_and_only_those(self):
        assert set(self._rows()) == set(PRESETS)

    @pytest.mark.parametrize('preset', PRESETS.values(), ids=lambda p: p.name)
    def test_the_table_names_the_judges_that_are_seated(self, preset):
        judges, _ = self._rows()[preset.name]

        assert judges == preset.judges

    @pytest.mark.parametrize('preset', PRESETS.values(), ids=lambda p: p.name)
    def test_the_table_publishes_the_cost_the_code_computes(self, preset):
        _, cost = self._rows()[preset.name]

        assert cost == preset.cost_per_1k()
