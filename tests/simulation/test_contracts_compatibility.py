from __future__ import annotations

import subprocess
import sys

from pathlib import Path


def _run_fresh_process(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args, '-c', script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )


def test_legacy_simulation_types_round_trips_data_and_warns() -> None:
    script = """
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    import evaluatorq.simulation as sim
    legacy = sim.types

deprecations = [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]
assert len(deprecations) == 1
assert 'evaluatorq.simulation.contracts' in str(deprecations[0].message)
assert deprecations[0].filename == '<string>'

from evaluatorq.contracts import Message, TokenUsage
result = legacy.SimulationResult(
    messages=[Message(role='user', content='hello')],
    terminated_by=legacy.TerminatedBy.judge,
    reason='completed',
    goal_achieved=True,
    goal_completion_score=1.0,
    rules_broken=[],
    turn_count=1,
    token_usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    turn_metrics=[],
)
restored = legacy.SimulationResult.model_validate_json(result.model_dump_json())
assert restored == result
assert restored.messages[0].content == 'hello'
assert restored.token_usage.total_tokens == 5
from evaluatorq.simulation.contracts import SimulationResult as CanonicalResult
assert legacy.SimulationResult is CanonicalResult
assert sim.types is legacy
"""
    result = _run_fresh_process(script)
    assert result.returncode == 0, result.stdout + result.stderr


def test_direct_legacy_simulation_types_import_warns() -> None:
    script = """
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    from evaluatorq.simulation.types import SimulationResult, TerminatedBy
    from evaluatorq.contracts import Message, TokenUsage
assert len(caught) == 1
assert issubclass(caught[0].category, DeprecationWarning)
assert caught[0].filename == '<string>'
result = SimulationResult(
    messages=[Message(role='user', content='hello')],
    terminated_by=TerminatedBy.judge,
    reason='completed',
    goal_achieved=True,
    goal_completion_score=1.0,
    rules_broken=[],
    turn_count=1,
    token_usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    turn_metrics=[],
)
assert SimulationResult.model_validate_json(result.model_dump_json()) == result
from evaluatorq.simulation.contracts import SimulationResult as CanonicalResult
assert SimulationResult is CanonicalResult
"""
    result = _run_fresh_process(script)
    assert result.returncode == 0, result.stdout + result.stderr


def test_canonical_simulation_contract_import_does_not_warn() -> None:
    script = """
import evaluatorq.simulation as sim
from evaluatorq.simulation.contracts import SimulationResult
assert sim.SimulationResult is SimulationResult
"""
    result = _run_fresh_process(script, '-W', 'error::DeprecationWarning')
    assert result.returncode == 0, result.stdout + result.stderr
