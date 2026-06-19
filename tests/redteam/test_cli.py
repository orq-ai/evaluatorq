"""Tests for the redteam CLI (RES-345).

Covers:
- -V short flag wiring for --vulnerability
- Correct pass-through of vulnerabilities to red_team()
- Help text content for --vulnerability
- No conflict between -V (vulnerability) and -v (verbose)
- Both --vulnerability and --category are forwarded when provided
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import Result as CliResult
from typer.testing import CliRunner

from evaluatorq.redteam.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_report() -> MagicMock:
    """Return a minimal mock RedTeamReport that satisfies the CLI's post-run logic."""
    report = MagicMock()
    report.model_dump.return_value = {}
    return report


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

    def test_help_mentions_prompt_injection_example(self):
        """Help text includes the 'prompt_injection' example ID."""
        output = self._get_help_output()
        assert "prompt_injection" in output

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
        assert "not in DeliveryMethod" in result.output

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
