"""Tests for the redteam CLI (RES-345).

Covers:
- -V short flag wiring for --vulnerability
- Correct pass-through of vulnerabilities to red_team()
- Help text content for --vulnerability
- No conflict between -V (vulnerability) and -v (verbose)
- Both --vulnerability and --category are forwarded when provided
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import Result as CliResult
from typer.testing import CliRunner

from evaluatorq.redteam.cli import app
from evaluatorq.redteam.contracts import (
    AgentInfo,
    AttackInfo,
    AttackTechnique,
    Framework,
    Pipeline,
    RedTeamReport,
    RedTeamResult,
    ReportSummary,
    RunError,
    Severity,
    TurnType,
)

runner = CliRunner()


def test_ui_warns_that_dashboard_is_deprecated(tmp_path: Path) -> None:
    report = tmp_path / 'report.json'
    report.write_text('{}', encoding='utf-8')

    with patch('evaluatorq.common.ui.launch.launch_streamlit'):
        result = runner.invoke(app, ['ui', str(report)])

    assert result.exit_code == 0, result.output
    assert 'deprecated' in result.stderr.lower()
    assert 'eq dashboard' in result.stderr


def test_runs_suggests_dashboard_directory(tmp_path: Path) -> None:
    report = tmp_path / 'run.json'
    report.write_text(
        '{"run_name":"demo","created_at":"2026-07-13T12:00:00",'
        '"pipeline":"dynamic","tested_agents":["agent:test"],'
        '"summary":{"total_attacks":1,"vulnerability_rate":0.0}}',
        encoding='utf-8',
    )

    result = runner.invoke(app, ['runs', str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert f'open: eq dashboard {tmp_path}' in result.output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_report() -> MagicMock:
    """Return a minimal mock RedTeamReport that satisfies the CLI's post-run logic.

    ``summary`` is a real ReportSummary, not a MagicMock: the CLI branches on the
    derived ``no_verdict`` property, and a MagicMock attribute is truthy, which would
    make every CLI test look like a zero-coverage run.
    """
    report = MagicMock()
    report.model_dump.return_value = {}
    report.summary = ReportSummary(total_attacks=5, evaluated_attacks=5, resistance_rate=0.8)
    return report


def _report_with_evaluation_errors(errors: list[RunError | None]) -> RedTeamReport:
    """A report whose results carry the given per-attack evaluation errors.

    Real models rather than stand-ins: the hint reads typed fields, and a duck-typed
    stub would keep passing if those fields were renamed out from under it.
    """
    results = [
        RedTeamResult(
            attack=AttackInfo(
                id=f'a-{i}',
                category='ASI01',
                framework=Framework.OWASP_ASI,
                attack_technique=AttackTechnique.DIRECT_INJECTION,
                delivery_methods=[],
                turn_type=TurnType.SINGLE,
                severity=Severity.HIGH,
                source='test',
            ),
            agent=AgentInfo(key='agent-a'),
            messages=[],
            vulnerable=None if err is not None else True,
            evaluation_error=err,
        )
        for i, err in enumerate(errors)
    ]
    return RedTeamReport(
        created_at=datetime.now(tz=timezone.utc),
        pipeline=Pipeline.DYNAMIC,
        categories_tested=['ASI01'],
        total_results=len(results),
        results=results,
        summary=ReportSummary(total_attacks=len(results)),
    )


def _run_with_mocked_red_team(args: list[str], report: MagicMock | None = None) -> tuple[CliResult, MagicMock]:
    """Invoke the CLI with red_team patched out.

    Returns the CliRunner result object.
    """
    if report is None:
        report = _make_mock_report()

    with patch("evaluatorq.redteam.red_team", new=AsyncMock(return_value=report)) as mock_rt:
        result = runner.invoke(app, args, catch_exceptions=False)
        return result, mock_rt


# ---------------------------------------------------------------------------
# 1. Short flag -V is registered and wires correctly
# ---------------------------------------------------------------------------


class TestVulnerabilityShortFlag:
    """-V short flag for --vulnerability."""

    def test_short_flag_V_accepted_single_value(self):
        """-V goal_hijacking is accepted and passes vulnerabilities to red_team()."""
        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "-V", "goal_hijacking", "--yes"]
        )
        assert result.exit_code == 0, result.output
        _kwargs = mock_rt.call_args.kwargs
        assert _kwargs["vulnerabilities"] == ["goal_hijacking"]

    def test_short_flag_V_accepted_multiple_values(self):
        """-V can be repeated to pass multiple vulnerability IDs."""
        result, mock_rt = _run_with_mocked_red_team(
            [
                "run",
                "--target", "agent:test-agent",
                "-V", "goal_hijacking",
                "-V", "prompt_injection",
                "--yes",
            ]
        )
        assert result.exit_code == 0, result.output
        _kwargs = mock_rt.call_args.kwargs
        assert _kwargs["vulnerabilities"] == ["goal_hijacking", "prompt_injection"]

    def test_long_flag_vulnerability_still_works(self):
        """--vulnerability long form continues to work alongside -V alias."""
        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "--vulnerability", "tool_misuse", "--yes"]
        )
        assert result.exit_code == 0, result.output
        _kwargs = mock_rt.call_args.kwargs
        assert _kwargs["vulnerabilities"] == ["tool_misuse"]

    def test_comma_separated_vulnerabilities(self):
        """-V accepts comma-separated IDs, split before validation."""
        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "-V", "goal_hijacking,prompt_injection", "--yes"]
        )
        assert result.exit_code == 0, result.output
        assert mock_rt.call_args.kwargs["vulnerabilities"] == ["goal_hijacking", "prompt_injection"]


# ---------------------------------------------------------------------------
# 2. No conflict between -V and -v
# ---------------------------------------------------------------------------


class TestFlagConflicts:
    """-V (vulnerability) and -v (verbose) must not conflict."""

    def test_V_and_v_can_be_used_together(self):
        """-V and -v can both be supplied in the same invocation."""
        result, mock_rt = _run_with_mocked_red_team(
            [
                "run",
                "--target", "agent:test-agent",
                "-V", "goal_hijacking",
                "-v",
                "--yes",
            ]
        )
        assert result.exit_code == 0, result.output
        _kwargs = mock_rt.call_args.kwargs
        assert _kwargs["vulnerabilities"] == ["goal_hijacking"]

    def test_lowercase_v_does_not_set_vulnerabilities(self):
        """-v only affects verbosity, not vulnerabilities."""
        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "-v", "--yes"]
        )
        assert result.exit_code == 0, result.output
        _kwargs = mock_rt.call_args.kwargs
        assert _kwargs["vulnerabilities"] is None


# ---------------------------------------------------------------------------
# 3. Pass-through to red_team()
# ---------------------------------------------------------------------------


class TestVulnerabilityPassThrough:
    """Verify the CLI correctly forwards --vulnerability to red_team()."""

    def test_no_vulnerability_passes_none(self):
        """When --vulnerability is omitted, red_team() receives vulnerabilities=None."""
        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "--yes"]
        )
        assert result.exit_code == 0, result.output
        _kwargs = mock_rt.call_args.kwargs
        assert _kwargs["vulnerabilities"] is None

    def test_owasp_category_code_forwarded_as_is(self):
        """OWASP category codes like ASI01 are forwarded verbatim to red_team()."""
        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "-V", "ASI01", "--yes"]
        )
        assert result.exit_code == 0, result.output
        _kwargs = mock_rt.call_args.kwargs
        assert _kwargs["vulnerabilities"] == ["ASI01"]

    def test_vulnerability_and_category_both_forwarded(self):
        """When both --vulnerability and --category are supplied, both are forwarded to red_team()."""
        result, mock_rt = _run_with_mocked_red_team(
            [
                "run",
                "--target", "agent:test-agent",
                "-V", "goal_hijacking",
                "--category", "ASI02",
                "--yes",
            ]
        )
        assert result.exit_code == 0, result.output
        _kwargs = mock_rt.call_args.kwargs
        assert _kwargs["vulnerabilities"] == ["goal_hijacking"]
        assert _kwargs["categories"] == ["ASI02"]


# ---------------------------------------------------------------------------
# 4. Help text content
# ---------------------------------------------------------------------------


class TestVulnerabilityHelpText:
    """The --vulnerability help text must describe IDs, examples, and precedence."""

    def _get_help_output(self) -> str:
        import re
        # TERM=dumb + wide COLUMNS: stop rich from routing help to its own
        # terminal console (empty captured output) or wrapping flag tokens.
        result = runner.invoke(app, ["run", "--help"], env={"TERM": "dumb", "COLUMNS": "200"})
        # Strip ANSI escape codes so assertions work regardless of terminal width
        return re.sub(r'\x1b\[[0-9;]*m', '', result.output)

    def test_help_shows_vulnerability_flag(self):
        """--vulnerability appears in the help output."""
        output = self._get_help_output()
        assert "--vulnerability" in output

    def test_help_shows_V_short_flag(self):
        """-V short flag appears in the help output."""
        output = self._get_help_output()
        assert "-V" in output

    def test_help_mentions_goal_hijacking_example(self):
        """Help text includes the 'goal_hijacking' example ID."""
        output = self._get_help_output()
        assert "goal_hijacking" in output

    def test_help_mentions_precedence_over_category(self):
        """Help text states that --vulnerability takes precedence over --category."""
        output = self._get_help_output()
        assert "precedence" in output.lower()

    def test_help_mentions_owasp_category_codes_accepted(self):
        """Help text mentions OWASP category codes are also accepted (e.g. ASI01)."""
        output = self._get_help_output()
        assert "ASI01" in output or "LLM01" in output


# ---------------------------------------------------------------------------
# 4. --strategy / --delivery-method flags
# ---------------------------------------------------------------------------


class TestStrategyFlag:
    """--strategy / -s short and long form, single + repeated."""

    def test_short_flag_s_accepted_single_value(self):
        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "-s", "direct_override", "--yes"]
        )
        assert result.exit_code == 0, result.output
        assert mock_rt.call_args.kwargs["strategies"] == ["direct_override"]

    def test_short_flag_s_repeats(self):
        result, mock_rt = _run_with_mocked_red_team(
            [
                "run",
                "--target", "agent:test-agent",
                "-s", "direct_override",
                "-s", "crescendo_injection",
                "--yes",
            ]
        )
        assert result.exit_code == 0, result.output
        assert mock_rt.call_args.kwargs["strategies"] == [
            "direct_override",
            "crescendo_injection",
        ]

    def test_long_flag_strategy(self):
        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "--strategy", "jailbreak_dan", "--yes"]
        )
        assert result.exit_code == 0, result.output
        assert mock_rt.call_args.kwargs["strategies"] == ["jailbreak_dan"]

    def test_comma_separated_strategies(self):
        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "-s", "direct_override,crescendo_injection", "--yes"]
        )
        assert result.exit_code == 0, result.output
        assert mock_rt.call_args.kwargs["strategies"] == ["direct_override", "crescendo_injection"]

    def test_comma_separated_strategy_with_invalid_token_rejected(self):
        # One bad token in a CSV list is rejected via the known_strategy_names
        # branch (distinct from the delivery-method enum validation).
        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "-s", "direct_override,definitely_not_a_strategy", "--yes"]
        )
        assert result.exit_code != 0
        mock_rt.assert_not_called()

    def test_strategy_defaults_to_none_when_omitted(self):
        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "--yes"]
        )
        assert result.exit_code == 0, result.output
        assert mock_rt.call_args.kwargs["strategies"] is None

    def test_unknown_strategy_name_rejected(self):
        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "--strategy", "definitely_not_a_strategy", "--yes"]
        )
        assert result.exit_code == 2  # typer.BadParameter
        mock_rt.assert_not_called()

    def test_generated_prefix_name_accepted(self):
        # Runtime-generated strategy names (generated_*) are not in the registry
        # but must still pass validation so a user can re-filter a prior run.
        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "--strategy", "generated_single_01_foo", "--yes"]
        )
        assert result.exit_code == 0, result.output
        assert mock_rt.call_args.kwargs["strategies"] == ["generated_single_01_foo"]


class TestDeliveryMethodFlag:
    """--delivery-method / -d short and long form, enum validation."""

    def test_short_flag_d_accepted_single_value(self):
        from evaluatorq.redteam.contracts import DeliveryMethod

        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "-d", "crescendo", "--yes"]
        )
        assert result.exit_code == 0, result.output
        assert mock_rt.call_args.kwargs["delivery_methods"] == [DeliveryMethod.CRESCENDO]

    def test_short_flag_d_repeats(self):
        from evaluatorq.redteam.contracts import DeliveryMethod

        result, mock_rt = _run_with_mocked_red_team(
            [
                "run",
                "--target", "agent:test-agent",
                "-d", "crescendo",
                "-d", "base64",
                "--yes",
            ]
        )
        assert result.exit_code == 0, result.output
        assert mock_rt.call_args.kwargs["delivery_methods"] == [
            DeliveryMethod.CRESCENDO,
            DeliveryMethod.BASE64,
        ]

    def test_long_flag_delivery_method(self):
        from evaluatorq.redteam.contracts import DeliveryMethod

        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "--delivery-method", "leetspeak", "--yes"]
        )
        assert result.exit_code == 0, result.output
        assert mock_rt.call_args.kwargs["delivery_methods"] == [DeliveryMethod.LEETSPEAK]

    def test_delivery_method_unknown_value_accepted_as_open_set(self):
        # Unknown delivery methods are an open set: accepted (with a warning) and
        # forwarded as a raw string so a dataset's custom method stays filterable.
        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "-d", "not-a-real-method", "--yes"]
        )
        assert result.exit_code == 0, result.output
        assert mock_rt.call_args.kwargs["delivery_methods"] == ["not-a-real-method"]
        assert "not known delivery methods" in result.output

    def test_registered_custom_delivery_method_no_warning(self):
        from evaluatorq.redteam.delivery_method_registry import register_delivery_method

        register_delivery_method('emoji-smuggling', category='obfuscation')
        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "-d", "emoji-smuggling", "--yes"]
        )
        assert result.exit_code == 0, result.output
        assert mock_rt.call_args.kwargs["delivery_methods"] == ["emoji-smuggling"]
        assert "not known delivery methods" not in result.output

    def test_delivery_method_defaults_to_none_when_omitted(self):
        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "--yes"]
        )
        assert result.exit_code == 0, result.output
        assert mock_rt.call_args.kwargs["delivery_methods"] is None

    def test_comma_separated_delivery_methods(self):
        from evaluatorq.redteam.contracts import DeliveryMethod

        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "-d", "crescendo,base64", "--yes"]
        )
        assert result.exit_code == 0, result.output
        assert mock_rt.call_args.kwargs["delivery_methods"] == [
            DeliveryMethod.CRESCENDO,
            DeliveryMethod.BASE64,
        ]

    def test_comma_separated_mixes_known_and_unknown(self):
        # Known token -> enum, unknown token -> raw string (open set), both forwarded.
        from evaluatorq.redteam.contracts import DeliveryMethod

        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "agent:test-agent", "-d", "crescendo,bogus", "--yes"]
        )
        assert result.exit_code == 0, result.output
        assert mock_rt.call_args.kwargs["delivery_methods"] == [DeliveryMethod.CRESCENDO, "bogus"]


class TestStrategyAndDeliveryMethodCombined:
    """--strategy and --delivery-method can be combined freely."""

    def test_both_flags_forwarded_together(self):
        from evaluatorq.redteam.contracts import DeliveryMethod

        result, mock_rt = _run_with_mocked_red_team(
            [
                "run",
                "--target", "agent:test-agent",
                "-s", "crescendo_injection",
                "-d", "crescendo",
                "--yes",
            ]
        )
        assert result.exit_code == 0, result.output
        kwargs = mock_rt.call_args.kwargs
        assert kwargs["strategies"] == ["crescendo_injection"]
        assert kwargs["delivery_methods"] == [DeliveryMethod.CRESCENDO]


class TestZeroEvaluationCoverageExits:
    """A run where nothing could be scored must not exit 0 (RES: guardrail-blocked run)."""

    def test_exits_1_when_no_attack_could_be_evaluated(self):
        report = _make_mock_report()
        report.summary = ReportSummary(total_attacks=5, evaluated_attacks=0, unevaluated_attacks=5)

        result, _ = _run_with_mocked_red_team(["run", "--target", "agent:test-agent", "--yes"], report=report)

        assert result.exit_code == 1, result.output
        assert "0/5 attacks could be evaluated" in result.output

    def test_exits_0_when_a_verdict_exists(self):
        report = _make_mock_report()
        report.summary = ReportSummary(total_attacks=5, evaluated_attacks=5, resistance_rate=0.8)

        result, _ = _run_with_mocked_red_team(["run", "--target", "agent:test-agent", "--yes"], report=report)

        assert result.exit_code == 0, result.output

    def test_exits_0_when_partial_coverage_and_no_floor_configured(self):
        """Partial coverage with no recorded floor is a verdict, not a hard stop.

        The floor is what decides this, and a report that never recorded one (a
        legacy artifact, or ``min_evaluation_coverage=None``) has opted out of the
        gate — see ``ReportSummary.coverage_below_minimum``. A run that *does* carry
        a floor fails here instead; ``TestCoverageBelowMinimumExits`` covers that.
        """
        report = _make_mock_report()
        report.summary = ReportSummary(total_attacks=5, evaluated_attacks=1, unevaluated_attacks=4, resistance_rate=1.0)
        assert report.summary.min_evaluation_coverage is None

        result, _ = _run_with_mocked_red_team(["run", "--target", "agent:test-agent", "--yes"], report=report)

        assert result.exit_code == 0, result.output

    def test_exits_0_when_no_attacks_were_run(self):
        """Zero attacks is not a failed evaluation — nothing was attempted."""
        report = _make_mock_report()
        report.summary = ReportSummary(total_attacks=0, evaluated_attacks=0)

        result, _ = _run_with_mocked_red_team(["run", "--target", "agent:test-agent", "--yes"], report=report)

        assert result.exit_code == 0, result.output


class TestCoverageBelowMinimumExits:
    """A run that does produce a verdict but on too small a sample must still fail
    CI — distinct from TestZeroEvaluationCoverageExits, which covers no verdict at all.
    """

    def test_exits_1_when_coverage_below_minimum(self):
        report = _make_mock_report()
        report.summary = ReportSummary(
            total_attacks=10,
            evaluated_attacks=5,
            unevaluated_attacks=5,
            evaluation_coverage=0.5,
            min_evaluation_coverage=0.8,
            resistance_rate=1.0,
        )

        result, _ = _run_with_mocked_red_team(["run", "--target", "agent:test-agent", "--yes"], report=report)

        assert result.exit_code == 1, result.output
        assert "5/10 attacks could be" in result.output
        assert "50%" in result.output
        assert "80%" in result.output

    def test_exits_0_when_coverage_meets_the_floor(self):
        report = _make_mock_report()
        report.summary = ReportSummary(
            total_attacks=10,
            evaluated_attacks=9,
            unevaluated_attacks=1,
            evaluation_coverage=0.9,
            min_evaluation_coverage=0.8,
            resistance_rate=1.0,
        )

        result, _ = _run_with_mocked_red_team(["run", "--target", "agent:test-agent", "--yes"], report=report)

        assert result.exit_code == 0, result.output


class TestEvaluationErrorHint:
    """_evaluation_error_hint names the dominant judge-failure cause, falling back
    to generic advice only when no result actually recorded one.
    """

    def test_names_dominant_code_and_count_with_sample_message(self):
        from evaluatorq.redteam.cli import _evaluation_error_hint
        from evaluatorq.redteam.contracts import RunError

        report = _report_with_evaluation_errors([
            RunError(
                message='blocked by content policy', error_type='api_status', stage='evaluation', code='api_status'
            ),
            RunError(message='blocked again', error_type='api_status', stage='evaluation', code='api_status'),
            None,
        ])
        hint = _evaluation_error_hint(report)
        assert 'api_status' in hint
        assert '2/2' in hint
        assert 'blocked by content policy' in hint

    def test_falls_back_to_generic_hint_when_no_errors_recorded(self):
        from evaluatorq.redteam.cli import _GENERIC_EVAL_HINT, _evaluation_error_hint

        report = _report_with_evaluation_errors([None])
        assert _evaluation_error_hint(report) == _GENERIC_EVAL_HINT
