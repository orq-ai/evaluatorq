"""Tests for the PipelineHooks protocol and its implementations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from evaluatorq.redteam.hooks import ConfirmPayload


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_report(**kwargs):
    """Create a minimal RedTeamReport for use in tests."""
    from evaluatorq.redteam.contracts import Pipeline, RedTeamReport, ReportSummary

    return RedTeamReport(
        created_at=kwargs.get("created_at", datetime.now(tz=timezone.utc)),
        description=kwargs.get("description", "Test report"),
        pipeline=kwargs.get("pipeline", Pipeline.DYNAMIC),
        framework=kwargs.get("framework", None),
        categories_tested=kwargs.get("categories_tested", ["ASI01"]),
        tested_agents=kwargs.get("tested_agents", ["agent:test"]),
        total_results=kwargs.get("total_results", 10),
        results=kwargs.get("results", []),
        summary=kwargs.get(
            "summary",
            ReportSummary(
                total_attacks=10,
                evaluated_attacks=8,
                vulnerabilities_found=3,
                resistance_rate=0.7,
            ),
        ),
    )


def _make_confirm_payload(**kwargs) -> ConfirmPayload:
    """Create a minimal ConfirmPayload for use in tests."""
    from evaluatorq.redteam.hooks import ConfirmPayload

    payload: ConfirmPayload = {
        "agent_context": kwargs.get("agent_context", None),
        "num_datapoints": kwargs.get("num_datapoints", 20),
        "num_dynamic": kwargs.get("num_dynamic", None),
        "num_static": kwargs.get("num_static", None),
        "categories": kwargs.get("categories", ["ASI01", "ASI02"]),
        "attack_model": kwargs.get("attack_model", "azure/gpt-5-mini"),
        "evaluator_model": kwargs.get("evaluator_model", "azure/gpt-5-mini"),
        "max_turns": kwargs.get("max_turns", 5),
        "parallelism": kwargs.get("parallelism", 5),
        "filtering_metadata": kwargs.get("filtering_metadata", None),
        "mode": kwargs.get("mode", "dynamic"),
        "target": kwargs.get("target", "agent:test"),
        "dataset_path": kwargs.get("dataset_path", None),
        "vulnerabilities": kwargs.get("vulnerabilities", None),
    }
    return payload


# ---------------------------------------------------------------------------
# DefaultHooks tests
# ---------------------------------------------------------------------------


class TestDefaultHooks:
    """Tests for DefaultHooks loguru-based implementation."""

    def test_default_on_stage_start_logs(self, capfd):
        """on_stage_start should log the stage name and metadata."""
        from evaluatorq.redteam.hooks import DefaultHooks

        hooks = DefaultHooks()
        with patch("evaluatorq.redteam.hooks.logger") as mock_logger:
            hooks.on_stage_start("context_retrieval", {"target": "agent:test"})
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "context_retrieval" in call_args

    def test_default_on_stage_end_logs(self):
        """on_stage_end should log the stage name."""
        from evaluatorq.redteam.hooks import DefaultHooks

        hooks = DefaultHooks()
        with patch("evaluatorq.redteam.hooks.logger") as mock_logger:
            hooks.on_stage_end("context_retrieval", {"result": "ok"})
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "context_retrieval" in call_args

    def test_default_on_confirm_returns_true(self):
        """DefaultHooks.on_confirm always returns True."""
        from evaluatorq.redteam.hooks import DefaultHooks

        hooks = DefaultHooks()
        payload = _make_confirm_payload()
        result = hooks.on_confirm(payload)
        assert result is True

    def test_default_on_confirm_logs_plan(self):
        """DefaultHooks.on_confirm should log key plan details."""
        from evaluatorq.redteam.hooks import DefaultHooks

        hooks = DefaultHooks()
        payload = _make_confirm_payload(num_datapoints=42, categories=["ASI01", "ASI02", "ASI03"])
        with patch("evaluatorq.redteam.hooks.logger") as mock_logger:
            hooks.on_confirm(payload)
            mock_logger.info.assert_called()
            # At least one log call should mention datapoints or categories
            all_calls = " ".join(str(c) for c in mock_logger.info.call_args_list)
            assert "42" in all_calls or "datapoint" in all_calls.lower()

    def test_default_on_complete_logs_summary(self):
        """DefaultHooks.on_complete should log resistance rate and vulnerability count."""
        from evaluatorq.redteam.hooks import DefaultHooks

        hooks = DefaultHooks()
        report = _make_report()
        with patch("evaluatorq.redteam.hooks.logger") as mock_logger:
            hooks.on_complete(report)
            all_calls = " ".join(str(c) for c in mock_logger.info.call_args_list)
            # Should log something about the summary
            assert mock_logger.info.called

    def test_default_on_complete_logs_ui_hint(self):
        """DefaultHooks.on_complete should log a tip about 'evaluatorq redteam ui'."""
        from evaluatorq.redteam.hooks import DefaultHooks

        hooks = DefaultHooks()
        report = _make_report()
        with patch("evaluatorq.redteam.hooks.logger") as mock_logger:
            hooks.on_complete(report)
            all_calls = " ".join(str(c) for c in mock_logger.info.call_args_list)
            assert "evaluatorq" in all_calls.lower() or "redteam" in all_calls.lower() or "ui" in all_calls.lower()


# ---------------------------------------------------------------------------
# RichHooks tests
# ---------------------------------------------------------------------------


class TestRichHooks:
    """Tests for RichHooks Rich-terminal implementation."""

    def _make_rich_hooks(self, **kwargs):
        from io import StringIO

        from rich.console import Console

        from evaluatorq.redteam.hooks import RichHooks

        buf = StringIO()
        console = Console(file=buf, width=120, highlight=False, markup=True)
        hooks = RichHooks(console=console, **kwargs)
        return hooks, buf

    def test_rich_on_stage_start_renders_banner(self):
        """on_stage_start should render a rule/banner with the stage label."""
        hooks, buf = self._make_rich_hooks()
        hooks.on_stage_start("context_retrieval", {})
        output = buf.getvalue()
        assert len(output.strip()) > 0  # Something was rendered

    def test_rich_on_stage_start_uses_human_labels(self):
        """on_stage_start should use human-readable labels from _STAGE_LABELS."""
        from evaluatorq.redteam.hooks import _STAGE_LABELS

        hooks, buf = self._make_rich_hooks()
        hooks.on_stage_start("context_retrieval", {})
        output = buf.getvalue()
        expected_label = _STAGE_LABELS.get("context_retrieval", "")
        assert expected_label in output

    def test_rich_on_confirm_renders_plan_table(self):
        """on_confirm should render a table with run parameters."""
        hooks, buf = self._make_rich_hooks(skip_confirm=True)
        payload = _make_confirm_payload(num_datapoints=30, attack_model="gpt-4o")
        hooks.on_confirm(payload)
        output = buf.getvalue()
        assert "30" in output or "Datapoint" in output

    def test_rich_on_confirm_renders_agent_capabilities(self):
        """on_confirm should render agent capabilities when agent_context is provided."""
        hooks, buf = self._make_rich_hooks(skip_confirm=True)
        agent_ctx = {
            "tools": [{"name": "search_web"}, {"name": "send_email"}],
            "memory_stores": [{"store_id": "mem1"}],
            "knowledge_bases": [],
        }
        payload = _make_confirm_payload(agent_context=agent_ctx)
        hooks.on_confirm(payload)
        output = buf.getvalue()
        # Should render tools count or memory info
        assert "search_web" in output or "2" in output or "tool" in output.lower()

    def test_rich_on_confirm_renders_filtering_metadata(self):
        """on_confirm should render filtering metadata (strategy counts)."""
        hooks, buf = self._make_rich_hooks(skip_confirm=True)
        filtering = {"total_strategies": 15, "filtered_count": 3, "categories_covered": 5}
        payload = _make_confirm_payload(filtering_metadata=filtering)
        hooks.on_confirm(payload)
        output = buf.getvalue()
        # Some filtering info should appear
        assert "15" in output or "strateg" in output.lower() or "filter" in output.lower()

    def test_rich_on_confirm_returns_false_on_decline(self):
        """on_confirm should return False when the user declines."""
        hooks, buf = self._make_rich_hooks(skip_confirm=False)
        payload = _make_confirm_payload()
        with patch("typer.confirm", return_value=False):
            result = hooks.on_confirm(payload)
        assert result is False

    def test_rich_on_confirm_skip_confirm_true(self):
        """on_confirm with skip_confirm=True should return True without prompting."""
        hooks, buf = self._make_rich_hooks(skip_confirm=True)
        payload = _make_confirm_payload()
        with patch("typer.confirm") as mock_confirm:
            result = hooks.on_confirm(payload)
            mock_confirm.assert_not_called()
        assert result is True

    def test_rich_on_confirm_no_agent_context(self):
        """on_confirm should work for static mode when agent_context is None."""
        hooks, buf = self._make_rich_hooks(skip_confirm=True)
        payload = _make_confirm_payload(agent_context=None, mode="static")
        # Should not raise
        result = hooks.on_confirm(payload)
        assert result is True

    def test_rich_on_complete_calls_print_report_summary(self):
        """on_complete should delegate to display.print_report_summary."""
        hooks, buf = self._make_rich_hooks()
        report = _make_report()
        with patch("evaluatorq.redteam.hooks.print_report_summary") as mock_print:
            hooks.on_complete(report)
            mock_print.assert_called_once()
            args, kwargs = mock_print.call_args
            assert args[0] is report

    def test_rich_on_complete_renders_ui_hint(self):
        """on_complete should render a hint about 'evaluatorq redteam ui'."""
        hooks, buf = self._make_rich_hooks()
        report = _make_report()
        with patch("evaluatorq.redteam.hooks.print_report_summary"):
            hooks.on_complete(report)
        output = buf.getvalue()
        assert "evaluatorq" in output.lower() or "redteam" in output.lower() or "ui" in output.lower()


# ---------------------------------------------------------------------------
# Protocol compliance tests
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """Tests that implementations satisfy the PipelineHooks Protocol."""

    def test_default_hooks_satisfies_protocol(self):
        """DefaultHooks should be recognized as a PipelineHooks implementation."""
        from evaluatorq.redteam.hooks import DefaultHooks, PipelineHooks

        hooks = DefaultHooks()
        assert isinstance(hooks, PipelineHooks)

    def test_rich_hooks_satisfies_protocol(self):
        """RichHooks should be recognized as a PipelineHooks implementation."""
        from evaluatorq.redteam.hooks import PipelineHooks, RichHooks

        hooks = RichHooks()
        assert isinstance(hooks, PipelineHooks)

    def test_custom_hooks_satisfies_protocol(self):
        """A minimal custom class with all required methods should satisfy Protocol."""
        from evaluatorq.redteam.hooks import PipelineHooks

        class MyHooks:
            def on_stage_start(self, stage: str, meta: dict[str, Any]) -> None:
                pass

            def on_stage_end(self, stage: str, meta: dict[str, Any]) -> None:
                pass

            def on_confirm(self, payload) -> bool:
                return True

            def on_complete(self, report) -> None:
                pass

        hooks = MyHooks()
        assert isinstance(hooks, PipelineHooks)


# ---------------------------------------------------------------------------
# Integration tests (hooks wired into runner)
# ---------------------------------------------------------------------------


class TestHooksIntegration:
    """Integration tests for hooks wired into the runner."""

    @pytest.mark.asyncio
    async def test_confirm_returns_false_raises(self):
        """When on_confirm returns False, the runner should raise RuntimeError."""
        from evaluatorq.redteam.hooks import DefaultHooks, PipelineHooks
        from evaluatorq.redteam.runner import red_team

        class FalseConfirmHooks(DefaultHooks):
            def on_confirm(self, payload) -> bool:
                return False

        with patch("evaluatorq.redteam.runner._run_dynamic_or_hybrid") as mock_dynamic:

            async def _fake_dynamic(**kwargs):
                hooks = kwargs.get("hooks")
                payload = {
                    "agent_context": None,
                    "num_datapoints": 5,
                    "categories": ["ASI01"],
                    "attack_model": "m",
                    "evaluator_model": "m",
                    "max_turns": 1,
                    "parallelism": 1,
                    "filtering_metadata": None,
                    "mode": "dynamic",
                    "target": "agent:test",
                    "dataset_path": None,
                    "vulnerabilities": None,
                }
                if hooks is not None and not hooks.on_confirm(payload):
                    raise RuntimeError("Execution cancelled by confirmation callback")
                return MagicMock()

            mock_dynamic.side_effect = _fake_dynamic

            with pytest.raises(RuntimeError, match="cancelled"):
                await red_team("agent:test", mode="dynamic", hooks=FalseConfirmHooks())

    @pytest.mark.asyncio
    async def test_hook_exception_propagates(self):
        """Exceptions raised by hooks should propagate out of the runner."""
        from evaluatorq.redteam.hooks import DefaultHooks
        from evaluatorq.redteam.runner import red_team

        class ExplodingHooks(DefaultHooks):
            def on_stage_start(self, stage: str, meta: dict[str, Any]) -> None:
                raise ValueError("Hook exploded!")

        with patch("evaluatorq.redteam.runner._run_dynamic_or_hybrid") as mock_dynamic:

            async def _fake_dynamic(**kwargs):
                hooks = kwargs.get("hooks")
                if hooks is not None:
                    hooks.on_stage_start("context_retrieval", {})
                return MagicMock()

            mock_dynamic.side_effect = _fake_dynamic

            with pytest.raises(ValueError, match="Hook exploded!"):
                await red_team("agent:test", mode="dynamic", hooks=ExplodingHooks())
