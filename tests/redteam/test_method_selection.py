"""Tests for strategy-name / delivery-method selection in red_team().

Covers the predicate behavior in `_filter_by_method` directly, the filter
threaded end-to-end through the planner (including the filter-before-cap
ordering and the category-fallback path), plus runner-level warn/hard-fail
behavior for unknown names, unmatched delivery methods, and empty intersections.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from evaluatorq.redteam.adaptive.strategy_registry import (
    VULNERABILITY_STRATEGY_REGISTRY,
    _filter_by_method,
)
from evaluatorq.redteam.contracts import (
    AgentContext,
    AttackStrategy,
    AttackTechnique,
    DeliveryMethod,
    MemoryStoreInfo,
    Severity,
    ToolInfo,
    TurnType,
    Vulnerability,
)


def _capable_agent_context() -> AgentContext:
    """Agent context with tools + memory so capability filtering keeps >=2 strategies."""
    return AgentContext(
        key='test-agent',
        tools=[ToolInfo(name='search')],
        memory_stores=[MemoryStoreInfo(id='ms_001', key='history')],
    )


async def _plan(vulns, **kwargs):
    """Invoke the planner with the shared no-LLM defaults used across these tests."""
    from evaluatorq.redteam.adaptive.strategy_planner import plan_strategies_for_vulnerabilities

    base = dict(
        agent_context=_capable_agent_context(),
        vulnerabilities=vulns,
        llm_client=None,
        attack_model='test-model',
        max_turns=5,
        max_per_category=None,
        generate_additional_strategies=False,
        generated_strategy_count=0,
    )
    base.update(kwargs)
    return await plan_strategies_for_vulnerabilities(**base)


def _make_strategy(
    name: str,
    *,
    delivery_methods: list[DeliveryMethod],
    technique: AttackTechnique = AttackTechnique.DIRECT_INJECTION,
    turn_type: TurnType = TurnType.SINGLE,
    category: str = "ASI01",
) -> AttackStrategy:
    return AttackStrategy(
        category=category,
        name=name,
        description="test",
        attack_technique=technique,
        delivery_methods=delivery_methods,
        turn_type=turn_type,
        severity=Severity.MEDIUM,
        objective_template="x",
    )


@pytest.fixture
def sample_strategies() -> list[AttackStrategy]:
    return [
        _make_strategy("alpha", delivery_methods=[DeliveryMethod.CRESCENDO]),
        _make_strategy("beta", delivery_methods=[DeliveryMethod.BASE64, DeliveryMethod.LEETSPEAK]),
        _make_strategy("gamma", delivery_methods=[DeliveryMethod.MULTILINGUAL]),
        _make_strategy("alpha", delivery_methods=[DeliveryMethod.BASE64], category="LLM01"),
    ]


# ---------------------------------------------------------------------------
# 1. Predicate behavior
# ---------------------------------------------------------------------------


class TestFilterByMethod:
    """Pure predicate behavior — no runner, no planner."""

    def test_no_filters_returns_all(self, sample_strategies: list[AttackStrategy]) -> None:
        kept = _filter_by_method(sample_strategies, names=None, delivery_methods=None)
        assert kept == sample_strategies

    def test_filter_by_strategy_name_narrows_set(self, sample_strategies: list[AttackStrategy]) -> None:
        kept = _filter_by_method(sample_strategies, names={"beta"}, delivery_methods=None)
        assert [s.name for s in kept] == ["beta"]

    def test_filter_by_delivery_method_narrows_set(self, sample_strategies: list[AttackStrategy]) -> None:
        kept = _filter_by_method(
            sample_strategies, names=None, delivery_methods={DeliveryMethod.CRESCENDO}
        )
        assert [s.name for s in kept] == ["alpha"]

    def test_filter_and_semantics_when_both_set(self, sample_strategies: list[AttackStrategy]) -> None:
        # alpha appears twice (different categories, different delivery methods).
        # name=alpha + delivery=base64 should keep only the LLM01 variant.
        kept = _filter_by_method(
            sample_strategies, names={"alpha"}, delivery_methods={DeliveryMethod.BASE64}
        )
        assert len(kept) == 1
        assert kept[0].category == "LLM01"

    def test_strategy_name_matches_across_categories(self, sample_strategies: list[AttackStrategy]) -> None:
        # Duplicate names are intentional — both alpha strategies survive.
        kept = _filter_by_method(sample_strategies, names={"alpha"}, delivery_methods=None)
        assert {s.category for s in kept} == {"ASI01", "LLM01"}

    def test_delivery_method_overlap_semantics(self, sample_strategies: list[AttackStrategy]) -> None:
        # beta has [BASE64, LEETSPEAK] — selecting LEETSPEAK keeps it.
        kept = _filter_by_method(
            sample_strategies, names=None, delivery_methods={DeliveryMethod.LEETSPEAK}
        )
        assert [s.name for s in kept] == ["beta"]

    def test_empty_intersection_returns_empty(self, sample_strategies: list[AttackStrategy]) -> None:
        kept = _filter_by_method(
            sample_strategies,
            names={"nonexistent"},
            delivery_methods={DeliveryMethod.CRESCENDO},
        )
        assert kept == []

    def test_empty_name_set_filters_everything_out(self, sample_strategies: list[AttackStrategy]) -> None:
        # Explicit empty set means "match nothing", distinct from None.
        kept = _filter_by_method(sample_strategies, names=set(), delivery_methods=None)
        assert kept == []


# ---------------------------------------------------------------------------
# 2. Registry-level invariants
# ---------------------------------------------------------------------------


class TestRegistryNameUniqueness:
    """Regression guard: strategy names are unique across the registry today.

    The filter predicate does not rely on uniqueness — it handles duplicates
    correctly — but if uniqueness is ever broken, the docstring on
    ``red_team(strategies=...)`` should be revisited to clarify multi-match
    semantics to users.
    """

    def test_no_strategy_name_appears_in_multiple_vulnerabilities(self) -> None:
        name_to_vulns: dict[str, set[str]] = {}
        for vuln, strategies in VULNERABILITY_STRATEGY_REGISTRY.items():
            for strategy in strategies:
                name_to_vulns.setdefault(strategy.name, set()).add(vuln.value)

        duplicated = {name: vulns for name, vulns in name_to_vulns.items() if len(vulns) > 1}
        assert not duplicated, (
            f"Strategy names are no longer unique across the registry: {duplicated}. "
            "Update the docstring on red_team(strategies=...) to document the multi-match semantics."
        )

    def test_filter_against_real_registry_finds_each_name(self) -> None:
        # Sanity: a known registered name yields at least one match through
        # the real predicate path.
        all_strategies = [s for strategies in VULNERABILITY_STRATEGY_REGISTRY.values() for s in strategies]
        sample_name = all_strategies[0].name
        kept = _filter_by_method(all_strategies, names={sample_name}, delivery_methods=None)
        assert len(kept) >= 1
        assert all(s.name == sample_name for s in kept)


def _capture_warnings(fn) -> list[str]:
    """Run fn() with a temporary loguru sink and return WARNING-level messages."""
    from loguru import logger

    msgs: list[str] = []
    sink_id = logger.add(msgs.append, level="WARNING")
    try:
        fn()
    finally:
        logger.remove(sink_id)
    return msgs


def _static_dps(*methods: str) -> list:
    """Static (dataset-shaped) datapoints with a singular `delivery_method` field."""
    from evaluatorq import DataPoint

    return [DataPoint(inputs={"delivery_method": m, "category": "ASI01"}) for m in methods]


class TestCheckFilterResults:
    """Runner-level warn + hard-fail behavior shared by single/multi-target paths."""

    def _dps(self, *names: str, methods: list[str] | None = None) -> list:
        from evaluatorq import DataPoint

        return [
            DataPoint(inputs={"strategy": {"name": n, "delivery_methods": methods or []}})
            for n in names
        ]

    def test_no_filter_emits_nothing(self) -> None:
        from evaluatorq.redteam.runner import _check_filter_results

        msgs = _capture_warnings(lambda: _check_filter_results(self._dps("alpha"), None, None))
        assert msgs == []

    def test_unmatched_name_warns(self) -> None:
        from evaluatorq.redteam.runner import _check_filter_results

        msgs = _capture_warnings(
            lambda: _check_filter_results(self._dps("alpha"), {"alpha", "ghost"}, None)
        )
        assert any("Unmatched strategy name(s)" in m and "ghost" in m for m in msgs)
        # 'alpha' matched, so it must not be reported as unmatched.
        assert not any("alpha" in m for m in msgs)

    def test_all_names_matched_no_warning(self) -> None:
        from evaluatorq.redteam.runner import _check_filter_results

        msgs = _capture_warnings(
            lambda: _check_filter_results(self._dps("alpha", "beta"), {"alpha"}, None)
        )
        assert msgs == []

    def test_unmatched_delivery_method_warns(self) -> None:
        # #2: a requested delivery method that matches no datapoint must warn,
        # even when another method matched and the run is non-empty.
        from evaluatorq.redteam.runner import _check_filter_results

        dps = self._dps("alpha", methods=["crescendo"])
        msgs = _capture_warnings(
            lambda: _check_filter_results(
                dps, None, {DeliveryMethod.CRESCENDO, DeliveryMethod.BASE64}
            )
        )
        warning = next((m for m in msgs if "Unmatched delivery method(s)" in m), None)
        assert warning is not None
        assert "base64" in warning  # the unmatched method is named
        # 'crescendo' matched, so it appears in the Present hint, not as unmatched.
        assert "Present: ['crescendo']" in warning

    def test_delivery_method_only_no_warning_when_matched(self) -> None:
        from evaluatorq.redteam.runner import _check_filter_results

        dps = self._dps("alpha", methods=["crescendo"])
        msgs = _capture_warnings(
            lambda: _check_filter_results(dps, None, {DeliveryMethod.CRESCENDO})
        )
        assert msgs == []

    def test_empty_intersection_raises(self) -> None:
        # #1: an empty filtered run is a hard failure, not a warning.
        from evaluatorq.redteam.exceptions import RedTeamError
        from evaluatorq.redteam.runner import _check_filter_results

        with pytest.raises(RedTeamError, match="zero datapoints"):
            _check_filter_results([], {"alpha"}, {DeliveryMethod.CRESCENDO})

    def test_no_raise_without_user_filter(self) -> None:
        # Empty datapoints + no filter must not raise (unrelated empty runs).
        from evaluatorq.redteam.runner import _check_filter_results

        _check_filter_results([], None, None)  # must not raise

    def test_static_names_apply_false_suppresses_name_warning(self) -> None:
        # Static path: strategy names never match (no strategy name on rows), so
        # names_apply=False suppresses the per-name warning.
        from evaluatorq.redteam.runner import _check_filter_results

        msgs = _capture_warnings(
            lambda: _check_filter_results(
                _static_dps("crescendo"), {"alpha"}, {DeliveryMethod.CRESCENDO}, names_apply=False
            )
        )
        assert not any("Unmatched strategy name(s)" in m for m in msgs)

    def test_static_delivery_method_matched_via_singular_field(self) -> None:
        from evaluatorq.redteam.runner import _check_filter_results

        msgs = _capture_warnings(
            lambda: _check_filter_results(
                _static_dps("crescendo"), None, {DeliveryMethod.CRESCENDO}, names_apply=False
            )
        )
        assert msgs == []

    def test_mixed_dynamic_and_static_datapoints(self) -> None:
        # The docstring claims the check covers a combined dynamic+static set
        # (hybrid mode). A dynamic strategy provides the name match; a static row
        # provides the delivery-method match via its singular field.
        from evaluatorq.redteam.runner import _check_filter_results

        mixed = self._dps("alpha", methods=["crescendo"]) + _static_dps("base64")
        msgs = _capture_warnings(
            lambda: _check_filter_results(
                mixed, {"alpha"}, {DeliveryMethod.CRESCENDO, DeliveryMethod.BASE64}
            )
        )
        # Both the name and both methods matched somewhere in the combined set.
        assert msgs == []


# ---------------------------------------------------------------------------
# 4. Filter threaded end-to-end through the planner (no LLM)
# ---------------------------------------------------------------------------


VULN = Vulnerability.GOAL_HIJACKING


class TestPlannerFilterIntegration:
    """The filter must actually run inside the planner, not just in isolation.

    These exercise the real registry through the no-LLM planner path so a
    dropped ``strategy_names=``/``delivery_methods=`` kwarg at any planner call
    site fails a test.
    """

    @pytest.mark.asyncio
    async def test_filter_by_name_narrows_planner_output(self) -> None:
        base, _, _ = await _plan([VULN])
        names = [s.name for s in base[VULN]]
        assert len(names) >= 2, 'need >=2 applicable strategies for a meaningful filter'
        target = names[-1]

        filtered, _, _ = await _plan([VULN], strategy_names={target})
        assert [s.name for s in filtered[VULN]] == [target]

    @pytest.mark.asyncio
    async def test_filter_by_delivery_method_narrows_planner_output(self) -> None:
        base, _, _ = await _plan([VULN])
        # Find a delivery method that some — but not all — strategies use, so the
        # filter provably EXCLUDES something (not a vacuous all-pass assertion).
        method = next(
            (m for s in base[VULN] for m in s.delivery_methods
             if not all(m in t.delivery_methods for t in base[VULN])),
            None,
        )
        assert method is not None, 'need a non-universal delivery method for a meaningful filter'

        filtered, _, _ = await _plan([VULN], delivery_methods={method})
        assert filtered[VULN]  # at least the source strategy survives
        assert all(method in s.delivery_methods for s in filtered[VULN])  # survivors all match
        assert len(filtered[VULN]) < len(base[VULN])  # filter excluded at least one

    @pytest.mark.asyncio
    async def test_name_filter_survives_max_per_category_cap(self) -> None:
        # Load-bearing guarantee: filter runs BEFORE the cap, so an explicitly
        # selected strategy that sorts AFTER the cap index still survives.
        # If the filter ran after the cap, only names[0] would remain and the
        # filter on {late} would yield nothing.
        base, _, _ = await _plan([VULN])
        names = [s.name for s in base[VULN]]
        assert len(names) >= 2, 'need >=2 strategies so the cap can drop the late one'
        late = names[-1]

        filtered, _, _ = await _plan([VULN], max_per_category=1, strategy_names={late})
        assert [s.name for s in filtered[VULN]] == [late]

    @pytest.mark.asyncio
    async def test_category_fallback_path_applies_filter(self) -> None:
        # Unresolved category → fallback branch calls select_applicable_strategies
        # then _filter_by_method. Patch the lookup so we control the candidate set.
        from evaluatorq.redteam.adaptive import strategy_planner

        keep = _make_strategy('keep_me', delivery_methods=[DeliveryMethod.CRESCENDO])
        drop = _make_strategy('drop_me', delivery_methods=[DeliveryMethod.BASE64])

        with patch.object(strategy_planner, 'select_applicable_strategies', return_value=[keep, drop]):
            result, _, _ = await strategy_planner.plan_strategies_for_categories(
                agent_context=AgentContext(key='test-agent'),
                categories=['UNRESOLVABLE_CATEGORY'],
                llm_client=None,
                attack_model='test-model',
                max_turns=5,
                max_per_category=None,
                generate_additional_strategies=False,
                generated_strategy_count=0,
                strategy_names={'keep_me'},
            )

        assert [s.name for s in result['UNRESOLVABLE_CATEGORY']] == ['keep_me']


# ---------------------------------------------------------------------------
# 5. Static delivery-method filter — applied inside the dataset loader
# ---------------------------------------------------------------------------


_FIXTURE = Path(__file__).parent / 'fixtures' / 'static_e2e_dataset.json'


class TestBridgeDeliveryFilter:
    """`load_owasp_agentic_dataset`/`_apply_filters` filter static rows by delivery method.

    Static-mode `--delivery-method` flows here (not through the strategy registry),
    so this is the integration point for the static-path filter requirement.
    """

    def _dp(self, method: str, category: str = 'ASI01'):
        from evaluatorq import DataPoint

        return DataPoint(inputs={'delivery_method': method, 'category': category})

    def test_apply_filters_narrows_by_delivery_method(self) -> None:
        from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import _apply_filters

        dps = [self._dp('base64'), self._dp('crescendo'), self._dp('leetspeak')]
        kept = _apply_filters(dps, delivery_methods=['crescendo'])
        assert [dp.inputs['delivery_method'] for dp in kept] == ['crescendo']

    def test_delivery_filter_runs_before_num_samples_cap(self) -> None:
        # Regression: the cap must apply to the FILTERED set, not slice first.
        # crescendo row is last; cap=2 would slice it off if filtering ran after.
        from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import _apply_filters

        dps = [self._dp('base64'), self._dp('base64'), self._dp('crescendo')]
        kept = _apply_filters(dps, num_samples=2, delivery_methods=['crescendo'])
        assert [dp.inputs['delivery_method'] for dp in kept] == ['crescendo']

    def test_redteam_input_coerces_delivery_method(self) -> None:
        # Pattern A: known value -> DeliveryMethod enum (usable as-is); unknown -> raw str.
        from evaluatorq.redteam.contracts import RedTeamInput, Severity, TurnType, VulnerabilityDomain

        base = dict(
            id='t1', category='ASI01', severity=Severity.MEDIUM,
            vulnerability_domain=next(iter(VulnerabilityDomain)), turn_type=TurnType.SINGLE, source='test',
        )
        known = RedTeamInput(delivery_method='direct-request', **base)
        assert known.delivery_method is DeliveryMethod.DIRECT_REQUEST

        unknown = RedTeamInput(delivery_method='my-custom-method', **base)
        assert unknown.delivery_method == 'my-custom-method'
        assert not isinstance(unknown.delivery_method, DeliveryMethod)

    def test_delivery_match_is_exact(self) -> None:
        # delivery_method must equal the enum value exactly — no case/punctuation
        # coercion. A row spelled differently simply does not match.
        from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import _apply_filters

        dps = [self._dp('Base64'), self._dp('Direct Request'), self._dp('base64')]
        kept = _apply_filters(dps, delivery_methods=['base64'])
        assert [dp.inputs['delivery_method'] for dp in kept] == ['base64']

    def test_open_set_custom_method_is_filterable(self) -> None:
        # A non-enum (custom) delivery method present in a dataset can still be
        # selected — the open-set promise, exercised through the loader filter.
        from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import _apply_filters

        dps = [self._dp('my-custom-method'), self._dp('crescendo')]
        kept = _apply_filters(dps, delivery_methods=['my-custom-method'])
        assert [dp.inputs['delivery_method'] for dp in kept] == ['my-custom-method']

    def test_empty_delivery_value_never_matches(self) -> None:
        from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import _apply_filters

        dps = [self._dp(''), self._dp('crescendo')]
        kept = _apply_filters(dps, delivery_methods=['crescendo'])
        assert [dp.inputs['delivery_method'] for dp in kept] == ['crescendo']

    def test_load_from_file_applies_delivery_filter(self) -> None:
        # End-to-end through the public loader against the real e2e fixture
        # (all 3 rows are 'direct-request').
        from evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge import load_owasp_agentic_dataset

        assert _FIXTURE.exists(), 'e2e fixture missing'
        kept = load_owasp_agentic_dataset(dataset=_FIXTURE, delivery_methods=['direct-request'])
        assert len(kept) == 3
        assert load_owasp_agentic_dataset(dataset=_FIXTURE, delivery_methods=['base64']) == []
        # No filter → all rows load.
        assert len(load_owasp_agentic_dataset(dataset=_FIXTURE)) == 3
