"""Unit tests for the phase helpers extracted out of `_run_dynamic_or_hybrid`.

These paths were previously buried inside a 1079-line orchestration and could only
be reached by driving a whole red-team run. Now that each phase is a module-level
function, the branches that produce user-facing warning text and the round-robin
datapoint cap can be asserted directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from evaluatorq.contracts import AgentTarget, Message
from evaluatorq.redteam.adaptive.capability_classifier import AgentCapabilities
from evaluatorq.redteam.backends.base import BareTargetBackend
from evaluatorq.redteam.contracts import AgentContext, AgentResponse, Pipeline, RedTeamReport, TargetKind
from evaluatorq.redteam.reports.converters import compute_report_summary
from evaluatorq.redteam.runner import (
    PreparedTarget,
    _cap_strategy_breakdown,
    _collect_filter_warnings,
    _ctx_end_meta,
    _finalize_merged_report,
    _group_results_by_target,
)


class _StubTarget(AgentTarget):
    """Minimal AgentTarget so a real PreparedTarget can be built without a live backend."""

    async def respond(self, messages: list[Message]) -> AgentResponse:
        return AgentResponse(text='ok')

    def new(self) -> _StubTarget:
        return _StubTarget()


def _noop_job(*_args: Any, **_kwargs: Any) -> None:
    """Stand-in for the @job-decorated callables; these phase helpers never invoke them."""
    return None


def _prepared(safe_target: str, filtering_metadata: dict[str, Any] | None = None) -> PreparedTarget:
    """Build a real PreparedTarget; only safe_target and filtering_metadata are read here."""
    return PreparedTarget(
        target=safe_target,
        target_kind=TargetKind.AGENT,
        target_value=safe_target,
        safe_target=safe_target,
        agent_context=AgentContext(key=safe_target),
        dynamic_datapoints=[],
        static_datapoints=[],
        all_datapoints=[],
        job=_noop_job,
        dynamic_job=_noop_job,
        backend=BareTargetBackend(_StubTarget()),
        resolved_llm_client=None,
        filtering_metadata=filtering_metadata or {},
        memory_entity_ids=[],
    )


def _empty_report() -> RedTeamReport:
    return RedTeamReport(
        created_at=datetime.now(tz=timezone.utc),
        description=None,
        pipeline=Pipeline.DYNAMIC,
        framework=None,
        categories_tested=[],
        tested_agents=[],
        total_results=0,
        results=[],
        summary=compute_report_summary([]),
    )


class TestCapStrategyBreakdown:
    """The round-robin allocation that caps the confirm prompt's datapoint estimate."""

    def test_allocates_round_robin_up_to_the_budget(self):
        breakdown: dict[str, Any] = {'a': {'selected': 5}, 'b': {'selected': 3}}
        est = _cap_strategy_breakdown(strategy_breakdown=breakdown, max_dynamic_datapoints=4)

        assert est == 4
        assert breakdown['a']['capped'] + breakdown['b']['capped'] == 4
        # Round-robin hands out one at a time, so the two categories differ by at most one.
        assert abs(breakdown['a']['capped'] - breakdown['b']['capped']) <= 1

    def test_stops_when_every_category_is_exhausted(self):
        breakdown: dict[str, Any] = {'a': {'selected': 1}, 'b': {'selected': 1}}
        est = _cap_strategy_breakdown(strategy_breakdown=breakdown, max_dynamic_datapoints=10)

        # The budget exceeds the supply: allocation stops rather than looping forever.
        assert breakdown['a']['capped'] == 1
        assert breakdown['b']['capped'] == 1
        assert est == 10

    def test_empty_breakdown_terminates(self):
        breakdown: dict[str, Any] = {}
        assert _cap_strategy_breakdown(strategy_breakdown=breakdown, max_dynamic_datapoints=3) == 3
        assert breakdown == {}


class TestCollectFilterWarnings:
    """Each branch produces a distinct user-facing string."""

    def test_unresolved_category(self):
        warnings = _collect_filter_warnings(prepared_targets=[_prepared('t', {'_unresolved_categories': ['Foo']})])

        assert len(warnings) == 1
        assert "Category 'Foo'" in warnings[0]
        assert 'category could not be resolved' in warnings[0]

    def test_zero_selected_with_generation_error_names_the_error(self):
        warnings = _collect_filter_warnings(
            prepared_targets=[_prepared('t', {'ASI01': {'total_selected': 0, 'generation_error': 'boom'}})]
        )

        assert warnings == ["Category 'ASI01': zero strategies selected (generation error: boom)"]

    def test_zero_selected_without_generation_error(self):
        warnings = _collect_filter_warnings(prepared_targets=[_prepared('t', {'ASI01': {'total_selected': 0}})])

        assert len(warnings) == 1
        assert 'no applicable strategies found for this agent' in warnings[0]

    def test_generation_error_with_hardcoded_fallback_is_still_reported(self):
        warnings = _collect_filter_warnings(
            prepared_targets=[_prepared('t', {'ASI01': {'total_selected': 3, 'generation_error': 'HTTP 429'}})]
        )

        assert warnings == [
            "Category 'ASI01': strategy generation failed (HTTP 429); ran 3 hardcoded strategies only"
        ]

    def test_nonzero_selection_and_missing_metadata_warn_about_nothing(self):
        warnings = _collect_filter_warnings(
            prepared_targets=[
                _prepared('t1', {'ASI01': {'total_selected': 5}}),
                _prepared('t2', {}),
            ]
        )

        assert warnings == []

    def test_underscore_prefixed_keys_are_not_treated_as_categories(self):
        warnings = _collect_filter_warnings(prepared_targets=[_prepared('t', {'_internal': {'total_selected': 0}})])

        assert warnings == []


class TestFinalizeMergedReport:
    """The two zero-result warnings are mutually exclusive; both must fire on their own branch."""

    def test_zero_datapoints_branch(self):
        merged = _empty_report()
        _finalize_merged_report(
            merged=merged,
            run_id='run-1',
            pipeline_duration=1.5,
            prepared_targets=[],
            mode=Pipeline.DYNAMIC,
            all_datapoints=[],
        )

        assert merged.run_id == 'run-1'
        assert merged.duration_seconds == 1.5
        assert len(merged.pipeline_warnings) == 1
        assert 'Zero datapoints generated' in merged.pipeline_warnings[0]

    def test_zero_attacks_branch(self):
        merged = _empty_report()
        _finalize_merged_report(
            merged=merged,
            run_id='run-2',
            pipeline_duration=0.0,
            prepared_targets=[],
            mode=Pipeline.DYNAMIC,
            all_datapoints=[object()],
        )

        assert len(merged.pipeline_warnings) == 1
        assert 'Zero attacks executed' in merged.pipeline_warnings[0]

    def test_mutates_in_place_and_returns_nothing(self):
        merged = _empty_report()
        result = _finalize_merged_report(
            merged=merged,
            run_id='run-3',
            pipeline_duration=2.0,
            prepared_targets=[],
            mode=Pipeline.DYNAMIC,
            all_datapoints=[],
        )

        assert result is None
        assert merged.run_id == 'run-3'


class _StubJobResult:
    def __init__(self, job_name: str | None) -> None:
        self.job_name = job_name


class _StubResult:
    def __init__(self, job_results: list[_StubJobResult]) -> None:
        self.job_results = job_results


class TestGroupResultsByTarget:
    """Job results are matched to targets by an exact job_name lookup, never a substring."""

    def test_splits_a_shared_result_into_one_copy_per_target(self):
        targets = [_prepared('agent-a'), _prepared('agent-b')]
        result = _StubResult([
            _StubJobResult('redteam:dynamic:agent-a'),
            _StubJobResult('redteam:dynamic:agent-b'),
        ])

        grouped = _group_results_by_target(results=[result], prepared_targets=targets)

        assert set(grouped) == {'agent-a', 'agent-b'}
        assert len(grouped['agent-a']) == 1
        assert [jr.job_name for jr in grouped['agent-a'][0].job_results] == ['redteam:dynamic:agent-a']
        assert [jr.job_name for jr in grouped['agent-b'][0].job_results] == ['redteam:dynamic:agent-b']

    def test_unmatched_job_name_is_excluded(self):
        targets = [_prepared('agent-a')]
        result = _StubResult([_StubJobResult('redteam:dynamic:someone-else')])

        grouped = _group_results_by_target(results=[result], prepared_targets=targets)

        assert grouped == {}

    def test_result_without_job_results_is_assigned_to_every_target(self):
        targets = [_prepared('agent-a'), _prepared('agent-b')]
        result = _StubResult([])

        grouped = _group_results_by_target(results=[result], prepared_targets=targets)

        assert grouped['agent-a'] == [result]
        assert grouped['agent-b'] == [result]

    def test_static_and_hybrid_job_prefixes_both_match(self):
        targets = [_prepared('agent-a')]
        static = _StubResult([_StubJobResult('redteam:static:agent-a')])
        hybrid = _StubResult([_StubJobResult('redteam:hybrid:agent-a')])

        grouped = _group_results_by_target(results=[static, hybrid], prepared_targets=targets)

        assert len(grouped['agent-a']) == 2


class TestCtxEndMeta:
    """The CONTEXT_RETRIEVAL hook payload carries the classifier outcome per target."""

    def test_reports_the_error_string_for_a_failed_target(self):
        meta = _ctx_end_meta(
            target_label='agent-a',
            ctx=AgentContext(key='agent-a'),
            caps=AgentCapabilities(classification_failed=True),
            classification_errors={'agent-a': 'RuntimeError: boom'},
            classification_available=True,
        )

        assert meta['target'] == 'agent-a'
        assert meta['classification_error'] == 'RuntimeError: boom'
        assert meta['classification_available'] is True

    def test_reports_none_when_the_target_has_no_error(self):
        meta = _ctx_end_meta(
            target_label='agent-b',
            ctx=AgentContext(key='agent-b'),
            caps=AgentCapabilities(),
            classification_errors={'agent-a': 'RuntimeError: boom'},
            classification_available=False,
        )

        assert meta['classification_error'] is None
        assert meta['classification_available'] is False

    def test_counts_default_to_zero_for_an_empty_context(self):
        meta = _ctx_end_meta(
            target_label='agent-c',
            ctx=AgentContext(key='agent-c'),
            caps=AgentCapabilities(),
            classification_errors={},
            classification_available=True,
        )

        assert meta['num_tools'] == 0
        assert meta['num_memory_stores'] == 0
        assert meta['num_knowledge_bases'] == 0
