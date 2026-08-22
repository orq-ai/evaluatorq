"""simulate()/generate_and_simulate() must be able to set the target's own
timeout and retry budget -- previously pinned at SimulationRunner's literal
defaults (target_agent_timeout_ms=240_000, max_target_retries=2) because
api.py never threaded them through (F3).
"""

from __future__ import annotations

# ruff: noqa: S101
from evaluatorq.simulation.api import _build_simulation_job_and_cache
from evaluatorq.simulation.types import Message


async def _stub_target(messages: list[Message]) -> str:
    return "hi"


def test_default_target_agent_knobs_match_runner_defaults() -> None:
    _, _, runner = _build_simulation_job_and_cache(
        job_name="sim",
        sim_dp_by_id={},
        target=_stub_target,
        target_agent=None,
        model="gpt-4o",
        max_turns=5,
        user_simulator=None,
        judge=None,
        generation_client=None,
        hooks=None,
    )
    assert runner._target_agent_timeout_ms == 240_000  # noqa: SLF001
    assert runner._max_target_retries == 2  # noqa: SLF001


def test_custom_target_agent_knobs_reach_the_runner() -> None:
    _, _, runner = _build_simulation_job_and_cache(
        job_name="sim",
        sim_dp_by_id={},
        target=_stub_target,
        target_agent=None,
        model="gpt-4o",
        max_turns=5,
        user_simulator=None,
        judge=None,
        generation_client=None,
        hooks=None,
        target_agent_timeout_ms=500_000,
        max_target_retries=7,
    )
    assert runner._target_agent_timeout_ms == 500_000  # noqa: SLF001
    assert runner._max_target_retries == 7  # noqa: SLF001


def test_default_max_tool_result_chars_matches_module_constant() -> None:
    from evaluatorq.simulation.runner.simulation import _MAX_TOOL_RESULT_CHARS

    _, _, runner = _build_simulation_job_and_cache(
        job_name="sim",
        sim_dp_by_id={},
        target=_stub_target,
        target_agent=None,
        model="gpt-4o",
        max_turns=5,
        user_simulator=None,
        judge=None,
        generation_client=None,
        hooks=None,
    )
    assert runner._max_tool_result_chars == _MAX_TOOL_RESULT_CHARS  # noqa: SLF001


def test_custom_max_tool_result_chars_reaches_the_runner() -> None:
    _, _, runner = _build_simulation_job_and_cache(
        job_name="sim",
        sim_dp_by_id={},
        target=_stub_target,
        target_agent=None,
        model="gpt-4o",
        max_turns=5,
        user_simulator=None,
        judge=None,
        generation_client=None,
        hooks=None,
        max_tool_result_chars=3000,
    )
    assert runner._max_tool_result_chars == 3000  # noqa: SLF001
