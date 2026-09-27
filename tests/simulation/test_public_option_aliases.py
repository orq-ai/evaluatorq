"""Clear simulation keyword names preserve existing calls and reject ambiguity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evaluatorq.simulation import api


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('entry_point', 'private_runner'),
    [('simulate', '_simulate_run'), ('generate_and_simulate', '_generate_and_simulate_run')],
)
@pytest.mark.parametrize(
    ('new_name', 'old_name', 'internal_name', 'value'),
    [
        ('experiment_description', 'evaluation_description', 'evaluation_description', 'support cases'),
        ('orq_folder_path', 'orq_results_path', 'orq_results_path', 'Support/September'),
        ('raise_on_execution_failure', 'exit_on_failure', 'exit_on_failure', False),
        ('report_path', 'report', 'report', Path('simulation.json')),
    ],
)
async def test_clear_and_existing_names_reach_the_same_run_option(
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
    private_runner: str,
    new_name: str,
    old_name: str,
    internal_name: str,
    value: object,
) -> None:
    captured: list[dict[str, Any]] = []

    async def fake_run(**kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs)
        return SimpleNamespace(results=[])

    monkeypatch.setattr(api, private_runner, fake_run)
    run = getattr(api, entry_point)
    for supplied in ({new_name: value}, {old_name: value}, {new_name: value, old_name: value}):
        assert await run(**supplied) == []
        assert captured[-1][internal_name] == value

    with pytest.raises(ValueError, match=f'{old_name} and {new_name} must match'):
        await run(**{old_name: value, new_name: 'different'})
    assert len(captured) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('entry_point', 'private_runner'),
    [('simulate', '_simulate_run'), ('generate_and_simulate', '_generate_and_simulate_run')],
)
async def test_execution_failure_gate_remains_enabled_by_default(
    monkeypatch: pytest.MonkeyPatch, entry_point: str, private_runner: str
) -> None:
    captured: dict[str, Any] = {}

    async def fake_run(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(results=[])

    monkeypatch.setattr(api, private_runner, fake_run)
    assert await getattr(api, entry_point)() == []
    assert captured['exit_on_failure'] is True
