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

import pytest
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
            ["run", "--target", "openai:gpt-4o-mini", "-V", "goal_hijacking", "--yes"]
        )
        assert result.exit_code == 0, result.output
        _kwargs = mock_rt.call_args.kwargs
        assert _kwargs["vulnerabilities"] == ["goal_hijacking"]

    def test_short_flag_V_accepted_multiple_values(self):
        """-V can be repeated to pass multiple vulnerability IDs."""
        result, mock_rt = _run_with_mocked_red_team(
            [
                "run",
                "--target", "openai:gpt-4o-mini",
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
            ["run", "--target", "openai:gpt-4o-mini", "--vulnerability", "tool_misuse", "--yes"]
        )
        assert result.exit_code == 0, result.output
        _kwargs = mock_rt.call_args.kwargs
        assert _kwargs["vulnerabilities"] == ["tool_misuse"]


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
                "--target", "openai:gpt-4o-mini",
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
            ["run", "--target", "openai:gpt-4o-mini", "-v", "--yes"]
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
            ["run", "--target", "openai:gpt-4o-mini", "--yes"]
        )
        assert result.exit_code == 0, result.output
        _kwargs = mock_rt.call_args.kwargs
        assert _kwargs["vulnerabilities"] is None

    def test_owasp_category_code_forwarded_as_is(self):
        """OWASP category codes like ASI01 are forwarded verbatim to red_team()."""
        result, mock_rt = _run_with_mocked_red_team(
            ["run", "--target", "openai:gpt-4o-mini", "-V", "ASI01", "--yes"]
        )
        assert result.exit_code == 0, result.output
        _kwargs = mock_rt.call_args.kwargs
        assert _kwargs["vulnerabilities"] == ["ASI01"]

    def test_vulnerability_and_category_both_forwarded(self):
        """When both --vulnerability and --category are supplied, both are forwarded to red_team()."""
        result, mock_rt = _run_with_mocked_red_team(
            [
                "run",
                "--target", "openai:gpt-4o-mini",
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
        result = runner.invoke(app, ["run", "--help"])
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
