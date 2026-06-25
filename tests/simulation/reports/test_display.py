"""Tests for print_simulation_summary() Rich console renderer."""

from __future__ import annotations

import io

import pytest
from rich.console import Console


def _console() -> tuple[Console, io.StringIO]:
    """Create a fixed-width console writing to a string buffer."""
    buf = io.StringIO()
    console = Console(file=buf, width=120, highlight=False, markup=False, no_color=True)
    return console, buf


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------


def test_empty_results_prints_no_results_line():
    from evaluatorq.simulation.reports.display import print_simulation_summary

    console, buf = _console()
    print_simulation_summary([], console=console)
    out = buf.getvalue()
    assert "No results" in out
    assert "Per-Persona" not in out


def test_empty_results_prints_no_breakdown_tables():
    from evaluatorq.simulation.reports.display import print_simulation_summary

    console, buf = _console()
    print_simulation_summary([], console=console)
    out = buf.getvalue()
    assert "Per-Scenario" not in out
    assert "Judge Verdicts" not in out


# ---------------------------------------------------------------------------
# Non-empty results — summary section
# ---------------------------------------------------------------------------


def test_summary_shows_simulation_count(sim_result_factory):
    from evaluatorq.simulation.reports.display import print_simulation_summary

    results = [sim_result_factory(), sim_result_factory(goal_achieved=False)]
    console, buf = _console()
    print_simulation_summary(results, console=console)
    out = buf.getvalue()
    assert "2 simulations" in out


def test_summary_shows_success_rate(sim_result_factory):
    from evaluatorq.simulation.reports.display import print_simulation_summary

    results = [sim_result_factory(goal_achieved=True), sim_result_factory(goal_achieved=True)]
    console, buf = _console()
    print_simulation_summary(results, console=console)
    out = buf.getvalue()
    # 100% success rate should appear
    assert "100%" in out


def test_summary_shows_goals_achieved(sim_result_factory):
    from evaluatorq.simulation.reports.display import print_simulation_summary

    results = [
        sim_result_factory(goal_achieved=True),
        sim_result_factory(goal_achieved=False),
    ]
    console, buf = _console()
    print_simulation_summary(results, console=console)
    out = buf.getvalue()
    # 1 achieved, 50% success rate
    assert "50%" in out


# ---------------------------------------------------------------------------
# Non-empty results — breakdown sections
# ---------------------------------------------------------------------------


def test_persona_breakdown_shown(sim_result_factory):
    from evaluatorq.simulation.reports.display import print_simulation_summary

    results = [
        sim_result_factory(persona="Alice"),
        sim_result_factory(persona="Bob", goal_achieved=False),
    ]
    console, buf = _console()
    print_simulation_summary(results, console=console)
    out = buf.getvalue()
    assert "Per-Persona" in out
    assert "Alice" in out
    assert "Bob" in out


def test_scenario_breakdown_shown(sim_result_factory):
    from evaluatorq.simulation.reports.display import print_simulation_summary

    results = [
        sim_result_factory(scenario="ScenA"),
        sim_result_factory(scenario="ScenB", goal_achieved=False),
    ]
    console, buf = _console()
    print_simulation_summary(results, console=console)
    out = buf.getvalue()
    assert "Per-Scenario" in out
    assert "ScenA" in out
    assert "ScenB" in out


def test_judge_verdicts_shown(sim_result_factory):
    from evaluatorq.simulation.reports.display import print_simulation_summary

    results = [sim_result_factory(), sim_result_factory(goal_achieved=False)]
    console, buf = _console()
    print_simulation_summary(results, console=console)
    out = buf.getvalue()
    assert "Judge Verdicts" in out


# ---------------------------------------------------------------------------
# Error section
# ---------------------------------------------------------------------------


def test_error_section_shown_when_errors(sim_result_factory):
    from evaluatorq.simulation.reports.display import print_simulation_summary

    results = [
        sim_result_factory(goal_achieved=True),
        sim_result_factory(error="Connection timeout"),
    ]
    console, buf = _console()
    print_simulation_summary(results, console=console)
    out = buf.getvalue()
    assert "Errors" in out
    assert "Connection timeout" in out


def test_error_section_absent_when_no_errors(sim_result_factory):
    from evaluatorq.simulation.reports.display import print_simulation_summary

    results = [sim_result_factory(goal_achieved=True)]
    console, buf = _console()
    print_simulation_summary(results, console=console)
    out = buf.getvalue()
    # No error section header when all runs succeeded
    assert "Connection" not in out


# ---------------------------------------------------------------------------
# Default console (stderr=True)
# ---------------------------------------------------------------------------


def test_uses_stderr_console_by_default(sim_result_factory):
    """print_simulation_summary(results) must not raise when console=None."""
    from evaluatorq.simulation.reports.display import print_simulation_summary

    results = [sim_result_factory()]
    # Should not raise; output goes to stderr
    print_simulation_summary(results)
