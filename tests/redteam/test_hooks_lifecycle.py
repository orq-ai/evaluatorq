"""Tests for the runner hooks lifecycle.

Verifies that PipelineHooks methods are called in the correct order and with
the expected payloads across all pipeline modes (dynamic, static).

Test scenarios:
1. Stage ordering — on_stage_start/on_stage_end called in correct order
2. Confirm payload structure — on_confirm receives expected ConfirmPayload keys
3. Confirm rejection — when on_confirm returns False, pipeline raises CancelledError
4. on_complete receives the final RedTeamReport with correct output_dir
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.redteam.contracts import (
    AgentContext,
    Pipeline,
    PipelineStage,
    RedTeamReport,
    ReportSummary,
)
from evaluatorq.redteam.exceptions import CancelledError
from evaluatorq.redteam.hooks import ConfirmPayload, PipelineHooks


# ---------------------------------------------------------------------------
# Spy hooks implementation
# ---------------------------------------------------------------------------


class SpyHooks:
    """Records every hook invocation for assertion in tests.

    All methods append a ``(method_name, *args)`` tuple to ``self.calls``.
    ``on_confirm`` returns the value set in ``confirm_result`` (default True).
    """

    def __init__(self, confirm_result: bool = True) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.confirm_result = confirm_result
        self.received_confirm_payload: ConfirmPayload | None = None
        self.received_complete_report: RedTeamReport | None = None
        self.received_complete_output_dir: str | None = None

    def on_stage_start(self, stage: PipelineStage | str, meta: dict[str, Any]) -> None:
        self.calls.append(("on_stage_start", stage, meta))

    def on_stage_end(self, stage: PipelineStage | str, meta: dict[str, Any]) -> None:
        self.calls.append(("on_stage_end", stage, meta))

    def on_confirm(self, payload: ConfirmPayload) -> bool:
        self.calls.append(("on_confirm", payload))
        self.received_confirm_payload = payload
        return self.confirm_result

    def on_complete(self, report: RedTeamReport, *, output_dir: str | None = None, auto_save_path: str | None = None) -> None:
        self.calls.append(("on_complete", report, output_dir))
        self.received_complete_report = report
        self.received_complete_output_dir = output_dir

    # ── Convenience query helpers ─────────────────────────────────────────

    def stage_starts(self) -> list[PipelineStage | str]:
        return [args[0] for name, *args in self.calls if name == "on_stage_start"]

    def stage_ends(self) -> list[PipelineStage | str]:
        return [args[0] for name, *args in self.calls if name == "on_stage_end"]

    def method_names(self) -> list[str]:
        return [name for name, *_ in self.calls]


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_report(**kwargs: Any) -> RedTeamReport:
    """Build a minimal RedTeamReport suitable for use as a mock return value."""
    defaults: dict[str, Any] = dict(
        created_at=datetime.now(tz=timezone.utc),
        description="Test report",
        pipeline=Pipeline.DYNAMIC,
        framework=None,
        categories_tested=["ASI01"],
        tested_agents=["llm:gpt-4o-mini"],
        total_results=0,
        results=[],
        summary=ReportSummary(),
    )
    defaults.update(kwargs)
    return RedTeamReport(**defaults)


def _make_agent_context() -> AgentContext:
    """Minimal AgentContext with no tools/memory so strategy count is low."""
    return AgentContext(key="test-agent", tools=[], memory_stores=[])


# ---------------------------------------------------------------------------
# Tests — static pipeline hooks
# ---------------------------------------------------------------------------


class TestStaticPipelineHooks:
    """Hooks lifecycle for the static (_run_static) pipeline."""

    def _make_datapoints(self, n: int = 2) -> list[Any]:
        """Create minimal DataPoint stubs for the static evaluatorq mock."""
        from evaluatorq import DataPoint

        return [
            DataPoint(inputs={"id": str(i), "category": "ASI01", "messages": []})
            for i in range(n)
        ]

    def _static_patches(self, datapoints: list[Any]) -> list[Any]:
        """Return the context manager patches common to static pipeline tests."""
        return [
            patch(
                "evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.load_owasp_agentic_dataset",
                return_value=datapoints,
            ),
            patch(
                "evaluatorq.redteam.frameworks.owasp.evaluators.get_evaluator_for_category",
                return_value=MagicMock(),
            ),
            patch(
                "evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.create_owasp_evaluator",
                return_value=MagicMock(),
            ),
            patch("evaluatorq.redteam.runtime.jobs.create_model_job", return_value=MagicMock()),
            patch("evaluatorq.evaluatorq", new_callable=AsyncMock, return_value=[]),
            patch(
                "evaluatorq.redteam.reports.converters.static_evaluatorq_results_to_reports",
                return_value={},
            ),
            patch(
                "evaluatorq.redteam.backends.registry.resolve_backend",
                return_value=MagicMock(
                    context_provider=MagicMock(
                        get_agent_context=AsyncMock(return_value=_make_agent_context())
                    )
                ),
            ),
        ]

    @pytest.mark.asyncio
    async def test_stage_ordering_static(self) -> None:
        """on_stage_start/on_stage_end are called in the correct order for static mode.

        Note: on_complete is fired by red_team() (the outer entry point), not
        by _run_static itself, so we drive through red_team() here to capture
        the full lifecycle in one spy.
        """
        spy = SpyHooks()
        datapoints = self._make_datapoints(2)
        mock_report = _make_report(pipeline=Pipeline.STATIC)

        # Capture the hooks argument _run_static receives and replay the
        # real stage calls by executing a thin side-effect that fires the
        # hooks in the same order as the real implementation, but without
        # needing a real LLM or dataset.
        async def _fake_run_static(**kwargs: Any) -> RedTeamReport:
            h = kwargs.get("hooks")
            if h is not None:
                h.on_confirm(
                    {
                        "num_datapoints": 2,
                        "mode": "static",
                        "categories": ["ASI01"],
                        "attack_model": "",
                        "evaluator_model": "gpt-4o-mini",
                        "max_turns": 1,
                        "parallelism": 1,
                        "target": "llm:gpt-4o-mini",
                        "dataset_path": None,
                        "vulnerabilities": [],
                        "agent_context": None,
                        "num_dynamic": None,
                        "num_static": 2,
                        "filtering_metadata": None,
                    }
                )
                h.on_stage_start(PipelineStage.ATTACK_EXECUTION, {"num_datapoints": 2, "targets": ["llm:gpt-4o-mini"]})
                h.on_stage_end(PipelineStage.ATTACK_EXECUTION, {"num_results": 0})
                h.on_stage_start(PipelineStage.REPORT_GENERATION, {"num_results": 0})
                h.on_stage_end(PipelineStage.REPORT_GENERATION, {"resistance_rate": 1.0, "elapsed_s": 0.1})
            return mock_report

        from evaluatorq.redteam.runner import red_team

        with patch("evaluatorq.redteam.runner._run_static", side_effect=_fake_run_static):
            await red_team("llm:gpt-4o-mini", mode="static", hooks=spy)

        starts = spy.stage_starts()
        ends = spy.stage_ends()

        # on_confirm must come before any attack execution stages
        confirm_idx = next(i for i, (name, *_) in enumerate(spy.calls) if name == "on_confirm")
        first_attack_start_idx = next(
            (
                i
                for i, (name, *args) in enumerate(spy.calls)
                if name == "on_stage_start" and args[0] == PipelineStage.ATTACK_EXECUTION
            ),
            None,
        )
        assert first_attack_start_idx is not None, "ATTACK_EXECUTION start not found"
        assert confirm_idx < first_attack_start_idx, "on_confirm must precede ATTACK_EXECUTION start"

        # ATTACK_EXECUTION must appear before REPORT_GENERATION
        assert PipelineStage.ATTACK_EXECUTION in starts
        assert PipelineStage.REPORT_GENERATION in starts
        assert starts.index(PipelineStage.ATTACK_EXECUTION) < starts.index(PipelineStage.REPORT_GENERATION)

        # Every start must have a matching end
        assert set(starts) == set(ends), "Mismatched stage starts vs ends"

        # on_complete must be the last recorded hook call (fired by red_team())
        assert spy.method_names()[-1] == "on_complete"

    @pytest.mark.asyncio
    async def test_confirm_payload_structure_static(self) -> None:
        """on_confirm receives a ConfirmPayload with expected keys for static mode."""
        spy = SpyHooks()
        datapoints = self._make_datapoints(3)

        with (
            patch(
                "evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.load_owasp_agentic_dataset",
                return_value=datapoints,
            ),
            patch(
                "evaluatorq.redteam.frameworks.owasp.evaluators.get_evaluator_for_category",
                return_value=MagicMock(),
            ),
            patch(
                "evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.create_owasp_evaluator",
                return_value=MagicMock(),
            ),
            patch("evaluatorq.redteam.runtime.jobs.create_model_job", return_value=MagicMock()),
            patch("evaluatorq.evaluatorq", new_callable=AsyncMock, return_value=[]),
            patch(
                "evaluatorq.redteam.reports.converters.static_evaluatorq_results_to_reports",
                return_value={},
            ),
            patch(
                "evaluatorq.redteam.backends.registry.resolve_backend",
                return_value=MagicMock(
                    context_provider=MagicMock(
                        get_agent_context=AsyncMock(return_value=_make_agent_context())
                    )
                ),
            ),
        ):
            from evaluatorq.redteam.runner import _run_static

            await _run_static(
                targets=["llm:gpt-4o-mini"],
                categories=["ASI01"],
                evaluator_model="gpt-4o-mini",
                parallelism=2,
                max_static_datapoints=None,
                backend="openai",
                dataset=None,
                description="test",
                hooks=spy,
            )

        payload = spy.received_confirm_payload
        assert payload is not None, "on_confirm was never called"

        # Required keys for static mode
        assert "num_datapoints" in payload
        assert "mode" in payload
        assert payload["mode"] == "static"
        assert "categories" in payload
        assert "evaluator_model" in payload
        assert "parallelism" in payload
        assert "target" in payload

        # num_datapoints must match the stub dataset size
        assert payload["num_datapoints"] == 3

        # parallelism passes through correctly
        assert payload["parallelism"] == 2

    @pytest.mark.asyncio
    async def test_confirm_rejection_aborts_static(self) -> None:
        """Returning False from on_confirm must raise CancelledError in static mode."""
        spy = SpyHooks(confirm_result=False)
        datapoints = self._make_datapoints(1)

        with (
            patch(
                "evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.load_owasp_agentic_dataset",
                return_value=datapoints,
            ),
            patch(
                "evaluatorq.redteam.frameworks.owasp.evaluators.get_evaluator_for_category",
                return_value=MagicMock(),
            ),
            patch(
                "evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.create_owasp_evaluator",
                return_value=MagicMock(),
            ),
            patch("evaluatorq.redteam.runtime.jobs.create_model_job", return_value=MagicMock()),
            patch("evaluatorq.evaluatorq", new_callable=AsyncMock, return_value=[]) as mock_eval,
        ):
            from evaluatorq.redteam.runner import _run_static

            with pytest.raises(CancelledError, match="cancelled by confirmation callback"):
                await _run_static(
                    targets=["llm:gpt-4o-mini"],
                    categories=["ASI01"],
                    evaluator_model="gpt-4o-mini",
                    parallelism=1,
                    max_static_datapoints=None,
                    backend="openai",
                    dataset=None,
                    description="test",
                    hooks=spy,
                )

            # Pipeline must have aborted before reaching evaluatorq()
            mock_eval.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_complete_static_no_output_dir(self) -> None:
        """on_complete is called with the final report and output_dir=None when not set."""
        from evaluatorq.redteam.runner import red_team

        mock_report = _make_report(pipeline=Pipeline.STATIC)

        with patch(
            "evaluatorq.redteam.runner._run_static",
            new_callable=AsyncMock,
            return_value=mock_report,
        ):
            spy = SpyHooks()
            result = await red_team(
                "llm:gpt-4o-mini",
                mode="static",
                hooks=spy,
                output_dir=None,
            )

        assert result is mock_report
        assert spy.received_complete_report is mock_report
        assert spy.received_complete_output_dir is None

    @pytest.mark.asyncio
    async def test_on_complete_static_with_output_dir(self, tmp_path: Any) -> None:
        """on_complete receives the output_dir string when output_dir is set."""
        from evaluatorq.redteam.runner import red_team

        mock_report = _make_report(pipeline=Pipeline.STATIC)

        with patch(
            "evaluatorq.redteam.runner._run_static",
            new_callable=AsyncMock,
            return_value=mock_report,
        ):
            spy = SpyHooks()
            result = await red_team(
                "llm:gpt-4o-mini",
                mode="static",
                hooks=spy,
                output_dir=str(tmp_path),
            )

        assert result is mock_report
        assert spy.received_complete_report is mock_report
        assert spy.received_complete_output_dir == str(tmp_path)


# ---------------------------------------------------------------------------
# Tests — dynamic pipeline hooks (via _run_dynamic_or_hybrid mock)
# ---------------------------------------------------------------------------


class TestDynamicPipelineHooksViaRunner:
    """Hooks lifecycle tests driven through the red_team() public entry point.

    _run_dynamic_or_hybrid is expensive to drive end-to-end, so these tests
    patch it and verify the parts of the lifecycle that live *outside* it (i.e.
    on_complete) while the internal stage sequencing is tested in the static
    tests above.
    """

    @pytest.mark.asyncio
    async def test_on_complete_called_after_dynamic_run(self) -> None:
        """on_complete is called with the returned RedTeamReport in dynamic mode."""
        from evaluatorq.redteam.runner import red_team

        mock_report = _make_report(pipeline=Pipeline.DYNAMIC)
        spy = SpyHooks()

        with patch(
            "evaluatorq.redteam.runner._run_dynamic_or_hybrid",
            new_callable=AsyncMock,
            return_value=mock_report,
        ):
            result = await red_team("llm:gpt-4o-mini", mode="dynamic", hooks=spy)

        assert result is mock_report
        assert spy.received_complete_report is mock_report

    @pytest.mark.asyncio
    async def test_on_complete_receives_report_instance(self) -> None:
        """on_complete must receive a proper RedTeamReport, not a dict or None."""
        from evaluatorq.redteam.runner import red_team

        mock_report = _make_report(pipeline=Pipeline.DYNAMIC)
        spy = SpyHooks()

        with patch(
            "evaluatorq.redteam.runner._run_dynamic_or_hybrid",
            new_callable=AsyncMock,
            return_value=mock_report,
        ):
            await red_team("llm:gpt-4o-mini", mode="dynamic", hooks=spy)

        assert isinstance(spy.received_complete_report, RedTeamReport)

    @pytest.mark.asyncio
    async def test_on_complete_is_last_hook_called(self) -> None:
        """on_complete must be the very last hook call in the lifecycle."""
        from evaluatorq.redteam.runner import red_team

        mock_report = _make_report()
        spy = SpyHooks()

        with patch(
            "evaluatorq.redteam.runner._run_dynamic_or_hybrid",
            new_callable=AsyncMock,
            return_value=mock_report,
        ):
            await red_team("llm:gpt-4o-mini", mode="dynamic", hooks=spy)

        assert spy.method_names()[-1] == "on_complete"


# ---------------------------------------------------------------------------
# Tests — dynamic internal pipeline hook ordering
# ---------------------------------------------------------------------------


class TestDynamicInternalStageOrdering:
    """Verify stage start/end ordering inside _run_dynamic_or_hybrid.

    Uses a real call to _run_dynamic_or_hybrid with all expensive sub-steps
    replaced by mocks so no LLM calls are made.
    """

    def _build_backend_mock(self, agent_context: AgentContext) -> Any:
        """Construct a mock BackendBundle that returns the given agent context."""
        return MagicMock(
            context_provider=MagicMock(
                get_agent_context=AsyncMock(return_value=agent_context)
            ),
            target_factory=MagicMock(),
            memory_cleanup=MagicMock(),
            error_mapper=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_dynamic_stage_start_end_paired(self) -> None:
        """Every on_stage_start call must be paired with a matching on_stage_end."""
        from evaluatorq import DataPoint
        from evaluatorq.redteam.runner import _run_dynamic_or_hybrid

        spy = SpyHooks()
        agent_ctx = _make_agent_context()
        datapoints = [DataPoint(inputs={"id": "1", "category": "ASI01", "messages": []})]
        mock_report = _make_report()
        backend_mock = self._build_backend_mock(agent_ctx)

        with (
            patch(
                "evaluatorq.redteam.backends.registry.resolve_backend",
                return_value=backend_mock,
            ),
            patch(
                "evaluatorq.redteam.adaptive.strategy_registry.select_applicable_strategies_for_vulnerability",
                return_value=[],
            ),
            patch(
                "evaluatorq.redteam.adaptive.strategy_registry.select_applicable_strategies",
                return_value=[],
            ),
            patch(
                "evaluatorq.redteam.adaptive.pipeline.generate_dynamic_datapoints_for_vulnerabilities",
                new_callable=AsyncMock,
                return_value=(datapoints, {}),
            ),
            patch(
                "evaluatorq.redteam.adaptive.pipeline.generate_dynamic_datapoints",
                new_callable=AsyncMock,
                return_value=(datapoints, {}),
            ),
            patch(
                "evaluatorq.redteam.adaptive.pipeline.create_dynamic_redteam_job",
                return_value=AsyncMock(return_value={"name": "test", "output": "ok"}),
            ),
            patch(
                "evaluatorq.redteam.adaptive.pipeline.create_dynamic_evaluator",
                return_value=MagicMock(),
            ),
            patch(
                "evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.create_owasp_evaluator",
                return_value=MagicMock(),
            ),
            patch("evaluatorq.evaluatorq", new_callable=AsyncMock, return_value=[]),
            patch(
                "evaluatorq.redteam.adaptive.orchestrator.ProgressDisplay",
                return_value=MagicMock(
                    __aenter__=AsyncMock(return_value=None),
                    __aexit__=AsyncMock(return_value=None),
                ),
            ),
            patch(
                "evaluatorq.redteam.reports.converters.dynamic_evaluatorq_results_to_report",
                return_value=mock_report,
            ),
            patch(
                "evaluatorq.redteam.reports.converters.merge_reports",
                return_value=mock_report,
            ),
            patch(
                "evaluatorq.redteam.reports.converters.static_results_to_report",
                return_value=mock_report,
            ),
            patch("evaluatorq.tracing.init_tracing_if_needed", new_callable=AsyncMock),
            patch("evaluatorq.tracing.capture_parent_context", new_callable=AsyncMock, return_value=None),
            patch(
                "evaluatorq.redteam.tracing.with_redteam_span",
                return_value=MagicMock(
                    __aenter__=AsyncMock(return_value=MagicMock()),
                    __aexit__=AsyncMock(return_value=None),
                ),
            ),
            patch("evaluatorq.redteam.runner._send_cleaned_results", new_callable=AsyncMock),
        ):
            await _run_dynamic_or_hybrid(
                targets=["llm:gpt-4o-mini"],
                mode=Pipeline.DYNAMIC,
                categories=["ASI01"],
                resolved_vulns=None,
                max_turns=1,
                max_per_category=1,
                attack_model="gpt-4o-mini",
                evaluator_model="gpt-4o-mini",
                parallelism=1,
                generate_strategies=False,
                generated_strategy_count=0,
                max_dynamic_datapoints=1,
                max_static_datapoints=None,
                cleanup_memory=False,
                backend="openai",
                target_factory=None,
                error_mapper=None,
                memory_cleanup=None,
                llm_client=MagicMock(),
                description="test",
                dataset=None,
                hooks=spy,
                output_dir=None,
                target_config=None,
            )

        starts = spy.stage_starts()
        ends = spy.stage_ends()

        # At minimum CONTEXT_RETRIEVAL, ATTACK_EXECUTION, and REPORT_GENERATION
        # must each start and end.
        required_stages = {
            PipelineStage.CONTEXT_RETRIEVAL,
            PipelineStage.ATTACK_EXECUTION,
            PipelineStage.REPORT_GENERATION,
        }
        for stage in required_stages:
            assert stage in starts, f"{stage} missing from on_stage_start calls"
            assert stage in ends, f"{stage} missing from on_stage_end calls"

        # Every start must have a matching end
        assert set(starts) == set(ends)

    @pytest.mark.asyncio
    async def test_dynamic_attack_execution_before_report_generation(self) -> None:
        """ATTACK_EXECUTION must always precede REPORT_GENERATION."""
        from evaluatorq import DataPoint
        from evaluatorq.redteam.runner import _run_dynamic_or_hybrid

        spy = SpyHooks()
        agent_ctx = _make_agent_context()
        datapoints = [DataPoint(inputs={"id": "1", "category": "ASI01", "messages": []})]
        mock_report = _make_report()
        backend_mock = self._build_backend_mock(agent_ctx)

        with (
            patch(
                "evaluatorq.redteam.backends.registry.resolve_backend",
                return_value=backend_mock,
            ),
            patch(
                "evaluatorq.redteam.adaptive.strategy_registry.select_applicable_strategies_for_vulnerability",
                return_value=[],
            ),
            patch(
                "evaluatorq.redteam.adaptive.strategy_registry.select_applicable_strategies",
                return_value=[],
            ),
            patch(
                "evaluatorq.redteam.adaptive.pipeline.generate_dynamic_datapoints_for_vulnerabilities",
                new_callable=AsyncMock,
                return_value=(datapoints, {}),
            ),
            patch(
                "evaluatorq.redteam.adaptive.pipeline.generate_dynamic_datapoints",
                new_callable=AsyncMock,
                return_value=(datapoints, {}),
            ),
            patch(
                "evaluatorq.redteam.adaptive.pipeline.create_dynamic_redteam_job",
                return_value=AsyncMock(return_value={"name": "test", "output": "ok"}),
            ),
            patch(
                "evaluatorq.redteam.adaptive.pipeline.create_dynamic_evaluator",
                return_value=MagicMock(),
            ),
            patch(
                "evaluatorq.redteam.frameworks.owasp.evaluatorq_bridge.create_owasp_evaluator",
                return_value=MagicMock(),
            ),
            patch("evaluatorq.evaluatorq", new_callable=AsyncMock, return_value=[]),
            patch(
                "evaluatorq.redteam.adaptive.orchestrator.ProgressDisplay",
                return_value=MagicMock(
                    __aenter__=AsyncMock(return_value=None),
                    __aexit__=AsyncMock(return_value=None),
                ),
            ),
            patch(
                "evaluatorq.redteam.reports.converters.dynamic_evaluatorq_results_to_report",
                return_value=mock_report,
            ),
            patch(
                "evaluatorq.redteam.reports.converters.merge_reports",
                return_value=mock_report,
            ),
            patch(
                "evaluatorq.redteam.reports.converters.static_results_to_report",
                return_value=mock_report,
            ),
            patch("evaluatorq.tracing.init_tracing_if_needed", new_callable=AsyncMock),
            patch("evaluatorq.tracing.capture_parent_context", new_callable=AsyncMock, return_value=None),
            patch(
                "evaluatorq.redteam.tracing.with_redteam_span",
                return_value=MagicMock(
                    __aenter__=AsyncMock(return_value=MagicMock()),
                    __aexit__=AsyncMock(return_value=None),
                ),
            ),
            patch("evaluatorq.redteam.runner._send_cleaned_results", new_callable=AsyncMock),
        ):
            await _run_dynamic_or_hybrid(
                targets=["llm:gpt-4o-mini"],
                mode=Pipeline.DYNAMIC,
                categories=["ASI01"],
                resolved_vulns=None,
                max_turns=1,
                max_per_category=1,
                attack_model="gpt-4o-mini",
                evaluator_model="gpt-4o-mini",
                parallelism=1,
                generate_strategies=False,
                generated_strategy_count=0,
                max_dynamic_datapoints=1,
                max_static_datapoints=None,
                cleanup_memory=False,
                backend="openai",
                target_factory=None,
                error_mapper=None,
                memory_cleanup=None,
                llm_client=MagicMock(),
                description="test",
                dataset=None,
                hooks=spy,
                output_dir=None,
                target_config=None,
            )

        starts = spy.stage_starts()
        assert starts.index(PipelineStage.ATTACK_EXECUTION) < starts.index(PipelineStage.REPORT_GENERATION)


# ---------------------------------------------------------------------------
# Tests — dynamic pipeline confirm payload (via _run_dynamic_or_hybrid)
# ---------------------------------------------------------------------------


class TestDynamicConfirmPayload:
    """Verify the ConfirmPayload structure produced by the dynamic pipeline."""

    @pytest.mark.asyncio
    async def test_dynamic_confirm_payload_keys(self) -> None:
        """on_confirm receives all expected keys from the dynamic pipeline."""
        from evaluatorq import DataPoint
        from evaluatorq.redteam.runner import _run_dynamic_or_hybrid

        spy = SpyHooks()
        agent_ctx = _make_agent_context()
        datapoints = [DataPoint(inputs={"id": "1", "category": "ASI01", "messages": []})]
        mock_report = _make_report()
        backend_mock = MagicMock(
            context_provider=MagicMock(
                get_agent_context=AsyncMock(return_value=agent_ctx)
            ),
            target_factory=MagicMock(),
            memory_cleanup=MagicMock(),
            error_mapper=MagicMock(),
        )

        with (
            patch(
                "evaluatorq.redteam.backends.registry.resolve_backend",
                return_value=backend_mock,
            ),
            patch(
                "evaluatorq.redteam.adaptive.strategy_registry.select_applicable_strategies_for_vulnerability",
                return_value=[],
            ),
            patch(
                "evaluatorq.redteam.adaptive.strategy_registry.select_applicable_strategies",
                return_value=[],
            ),
            patch(
                "evaluatorq.redteam.adaptive.pipeline.generate_dynamic_datapoints_for_vulnerabilities",
                new_callable=AsyncMock,
                return_value=(datapoints, {}),
            ),
            patch(
                "evaluatorq.redteam.adaptive.pipeline.generate_dynamic_datapoints",
                new_callable=AsyncMock,
                return_value=(datapoints, {}),
            ),
            patch(
                "evaluatorq.redteam.adaptive.pipeline.create_dynamic_redteam_job",
                return_value=AsyncMock(return_value={"name": "test", "output": "ok"}),
            ),
            patch(
                "evaluatorq.redteam.adaptive.pipeline.create_dynamic_evaluator",
                return_value=MagicMock(),
            ),
            patch("evaluatorq.evaluatorq", new_callable=AsyncMock, return_value=[]),
            patch(
                "evaluatorq.redteam.adaptive.orchestrator.ProgressDisplay",
                return_value=MagicMock(
                    __aenter__=AsyncMock(return_value=None),
                    __aexit__=AsyncMock(return_value=None),
                ),
            ),
            patch(
                "evaluatorq.redteam.reports.converters.dynamic_evaluatorq_results_to_report",
                return_value=mock_report,
            ),
            patch(
                "evaluatorq.redteam.reports.converters.merge_reports",
                return_value=mock_report,
            ),
            patch(
                "evaluatorq.redteam.reports.converters.static_results_to_report",
                return_value=mock_report,
            ),
            patch("evaluatorq.tracing.init_tracing_if_needed", new_callable=AsyncMock),
            patch("evaluatorq.tracing.capture_parent_context", new_callable=AsyncMock, return_value=None),
            patch(
                "evaluatorq.redteam.tracing.with_redteam_span",
                return_value=MagicMock(
                    __aenter__=AsyncMock(return_value=MagicMock()),
                    __aexit__=AsyncMock(return_value=None),
                ),
            ),
            patch("evaluatorq.redteam.runner._send_cleaned_results", new_callable=AsyncMock),
        ):
            await _run_dynamic_or_hybrid(
                targets=["llm:gpt-4o-mini"],
                mode=Pipeline.DYNAMIC,
                categories=["ASI01"],
                resolved_vulns=None,
                max_turns=3,
                max_per_category=2,
                attack_model="gpt-4o-mini",
                evaluator_model="gpt-4o-mini",
                parallelism=4,
                generate_strategies=False,
                generated_strategy_count=0,
                max_dynamic_datapoints=5,
                max_static_datapoints=None,
                cleanup_memory=False,
                backend="openai",
                target_factory=None,
                error_mapper=None,
                memory_cleanup=None,
                llm_client=MagicMock(),
                description="test",
                dataset=None,
                hooks=spy,
                output_dir=None,
                target_config=None,
            )

        payload = spy.received_confirm_payload
        assert payload is not None, "on_confirm was never called"

        # Mandatory keys present in the dynamic ConfirmPayload
        required_keys = {
            "num_datapoints",
            "categories",
            "attack_model",
            "evaluator_model",
            "max_turns",
            "parallelism",
            "mode",
            "target",
        }
        for key in required_keys:
            assert key in payload, f"ConfirmPayload missing key: {key!r}"

        # Cast to plain dict so item access is type-safe after the membership checks above.
        payload_dict: dict[str, Any] = dict(payload)

        # mode must be "dynamic"
        assert payload_dict["mode"] == "dynamic"

        # parallelism and max_turns pass through unchanged
        assert payload_dict["parallelism"] == 4
        assert payload_dict["max_turns"] == 3

        # target reflects the input target string
        assert "gpt-4o-mini" in payload_dict["target"]

    @pytest.mark.asyncio
    async def test_dynamic_confirm_rejection_aborts_before_datapoint_generation(self) -> None:
        """on_confirm returning False raises CancelledError before datapoint generation."""
        from evaluatorq.redteam.runner import _run_dynamic_or_hybrid

        spy = SpyHooks(confirm_result=False)
        agent_ctx = _make_agent_context()
        backend_mock = MagicMock(
            context_provider=MagicMock(
                get_agent_context=AsyncMock(return_value=agent_ctx)
            ),
            target_factory=MagicMock(),
            memory_cleanup=MagicMock(),
            error_mapper=MagicMock(),
        )

        gen_mock = AsyncMock(return_value=([], {}))

        with (
            patch(
                "evaluatorq.redteam.backends.registry.resolve_backend",
                return_value=backend_mock,
            ),
            patch(
                "evaluatorq.redteam.adaptive.strategy_registry.select_applicable_strategies_for_vulnerability",
                return_value=[],
            ),
            patch(
                "evaluatorq.redteam.adaptive.strategy_registry.select_applicable_strategies",
                return_value=[],
            ),
            patch(
                "evaluatorq.redteam.adaptive.pipeline.generate_dynamic_datapoints_for_vulnerabilities",
                gen_mock,
            ),
            patch(
                "evaluatorq.redteam.adaptive.pipeline.generate_dynamic_datapoints",
                gen_mock,
            ),
            patch("evaluatorq.tracing.init_tracing_if_needed", new_callable=AsyncMock),
            patch("evaluatorq.tracing.capture_parent_context", new_callable=AsyncMock, return_value=None),
            patch(
                "evaluatorq.redteam.tracing.with_redteam_span",
                return_value=MagicMock(
                    __aenter__=AsyncMock(return_value=MagicMock()),
                    __aexit__=AsyncMock(return_value=None),
                ),
            ),
        ):
            with pytest.raises(CancelledError, match="cancelled by confirmation callback"):
                await _run_dynamic_or_hybrid(
                    targets=["llm:gpt-4o-mini"],
                    mode=Pipeline.DYNAMIC,
                    categories=["ASI01"],
                    resolved_vulns=None,
                    max_turns=1,
                    max_per_category=1,
                    attack_model="gpt-4o-mini",
                    evaluator_model="gpt-4o-mini",
                    parallelism=1,
                    generate_strategies=False,
                    generated_strategy_count=0,
                    max_dynamic_datapoints=None,
                    max_static_datapoints=None,
                    cleanup_memory=False,
                    backend="openai",
                    target_factory=None,
                    error_mapper=None,
                    memory_cleanup=None,
                    llm_client=MagicMock(),
                    description="test",
                    dataset=None,
                    hooks=spy,
                    output_dir=None,
                    target_config=None,
                )

        # Datapoint generation must NOT have been called
        gen_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dynamic_confirm_payload_has_vulnerabilities_field(self) -> None:
        """ConfirmPayload includes a 'vulnerabilities' field (may be None for dynamic mode)."""
        from evaluatorq import DataPoint
        from evaluatorq.redteam.runner import _run_dynamic_or_hybrid

        spy = SpyHooks()
        agent_ctx = _make_agent_context()
        datapoints = [DataPoint(inputs={"id": "1", "category": "ASI01", "messages": []})]
        mock_report = _make_report()
        backend_mock = MagicMock(
            context_provider=MagicMock(
                get_agent_context=AsyncMock(return_value=agent_ctx)
            ),
            target_factory=MagicMock(),
            memory_cleanup=MagicMock(),
            error_mapper=MagicMock(),
        )

        with (
            patch(
                "evaluatorq.redteam.backends.registry.resolve_backend",
                return_value=backend_mock,
            ),
            patch(
                "evaluatorq.redteam.adaptive.strategy_registry.select_applicable_strategies_for_vulnerability",
                return_value=[],
            ),
            patch(
                "evaluatorq.redteam.adaptive.strategy_registry.select_applicable_strategies",
                return_value=[],
            ),
            patch(
                "evaluatorq.redteam.adaptive.pipeline.generate_dynamic_datapoints",
                new_callable=AsyncMock,
                return_value=(datapoints, {}),
            ),
            patch(
                "evaluatorq.redteam.adaptive.pipeline.generate_dynamic_datapoints_for_vulnerabilities",
                new_callable=AsyncMock,
                return_value=(datapoints, {}),
            ),
            patch(
                "evaluatorq.redteam.adaptive.pipeline.create_dynamic_redteam_job",
                return_value=AsyncMock(return_value={"name": "test", "output": "ok"}),
            ),
            patch(
                "evaluatorq.redteam.adaptive.pipeline.create_dynamic_evaluator",
                return_value=MagicMock(),
            ),
            patch("evaluatorq.evaluatorq", new_callable=AsyncMock, return_value=[]),
            patch(
                "evaluatorq.redteam.adaptive.orchestrator.ProgressDisplay",
                return_value=MagicMock(
                    __aenter__=AsyncMock(return_value=None),
                    __aexit__=AsyncMock(return_value=None),
                ),
            ),
            patch(
                "evaluatorq.redteam.reports.converters.static_results_to_report",
                return_value=mock_report,
            ),
            patch(
                "evaluatorq.redteam.reports.converters.merge_reports",
                return_value=mock_report,
            ),
            patch("evaluatorq.tracing.init_tracing_if_needed", new_callable=AsyncMock),
            patch("evaluatorq.tracing.capture_parent_context", new_callable=AsyncMock, return_value=None),
            patch(
                "evaluatorq.redteam.tracing.with_redteam_span",
                return_value=MagicMock(
                    __aenter__=AsyncMock(return_value=MagicMock()),
                    __aexit__=AsyncMock(return_value=None),
                ),
            ),
            patch("evaluatorq.redteam.runner._send_cleaned_results", new_callable=AsyncMock),
        ):
            await _run_dynamic_or_hybrid(
                targets=["llm:gpt-4o-mini"],
                mode=Pipeline.DYNAMIC,
                categories=["ASI01"],
                resolved_vulns=None,
                max_turns=1,
                max_per_category=None,
                attack_model="gpt-4o-mini",
                evaluator_model="gpt-4o-mini",
                parallelism=1,
                generate_strategies=False,
                generated_strategy_count=0,
                max_dynamic_datapoints=None,
                max_static_datapoints=None,
                cleanup_memory=False,
                backend="openai",
                target_factory=None,
                error_mapper=None,
                memory_cleanup=None,
                llm_client=MagicMock(),
                description="test",
                dataset=None,
                hooks=spy,
                output_dir=None,
                target_config=None,
            )

        payload = spy.received_confirm_payload
        assert payload is not None
        # 'vulnerabilities' key must be present (value may be None when no vulns resolved)
        assert "vulnerabilities" in payload


# ---------------------------------------------------------------------------
# Tests — SpyHooks satisfies PipelineHooks protocol
# ---------------------------------------------------------------------------


class TestSpyHooksProtocol:
    """Verify SpyHooks implements the PipelineHooks protocol correctly."""

    def test_spy_hooks_satisfies_protocol(self) -> None:
        spy = SpyHooks()
        assert isinstance(spy, PipelineHooks)

    def test_on_stage_start_records_call(self) -> None:
        spy = SpyHooks()
        spy.on_stage_start(PipelineStage.ATTACK_EXECUTION, {"key": "value"})
        assert ("on_stage_start", PipelineStage.ATTACK_EXECUTION, {"key": "value"}) in spy.calls

    def test_on_stage_end_records_call(self) -> None:
        spy = SpyHooks()
        spy.on_stage_end(PipelineStage.REPORT_GENERATION, {"elapsed_s": 1.5})
        assert ("on_stage_end", PipelineStage.REPORT_GENERATION, {"elapsed_s": 1.5}) in spy.calls

    def test_on_confirm_returns_true_by_default(self) -> None:
        spy = SpyHooks()
        result = spy.on_confirm({})  # type: ignore[arg-type]
        assert result is True

    def test_on_confirm_returns_false_when_configured(self) -> None:
        spy = SpyHooks(confirm_result=False)
        result = spy.on_confirm({})  # type: ignore[arg-type]
        assert result is False

    def test_on_complete_records_report_and_output_dir(self) -> None:
        spy = SpyHooks()
        report = _make_report()
        spy.on_complete(report, output_dir="/tmp/run")
        assert spy.received_complete_report is report
        assert spy.received_complete_output_dir == "/tmp/run"
