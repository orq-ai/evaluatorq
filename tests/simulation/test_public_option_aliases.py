"""Clear simulation keyword names preserve existing calls and reject ambiguity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evaluatorq.contracts import LLMCallConfig
from evaluatorq.simulation import api


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('entry_point', 'private_runner'),
    [('simulate', '_simulate_run'), ('generate_and_simulate', '_generate_and_simulate_run')],
)
@pytest.mark.parametrize(
    ('new_name', 'old_name', 'value'),
    [
        ('experiment_description', 'evaluation_description', 'support cases'),
        ('orq_folder_path', 'orq_results_path', 'Support/September'),
        ('raise_on_execution_failure', 'exit_on_failure', False),
        ('report_path', 'report', Path('simulation.json')),
    ],
)
async def test_clear_and_existing_names_reach_the_same_run_option(
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
    private_runner: str,
    new_name: str,
    old_name: str,
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
        assert captured[-1][new_name] == value

    with pytest.raises(ValueError, match=f'{old_name} and {new_name} must match'):
        await run(**{old_name: value, new_name: 'different'})
    assert len(captured) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('entry_point', ['simulate', 'generate_and_simulate'])
async def test_execution_failure_gate_remains_enabled_by_default(
    monkeypatch: pytest.MonkeyPatch, entry_point: str
) -> None:
    configs: list[Any] = []

    async def fake_core(**kwargs: Any) -> SimpleNamespace:
        configs.append(kwargs['config'])
        return SimpleNamespace(results=[])

    async def fake_generate(**_kwargs: Any) -> tuple[list[Any], None, bool, None]:
        return [], None, False, None

    monkeypatch.setattr(api, '_simulate_core', fake_core)
    monkeypatch.setattr(api, '_generate_datapoints_inner', fake_generate)
    extra = {'datapoints': []} if entry_point == 'simulate' else {'agent_description': 'bot'}
    assert (
        await getattr(api, entry_point)(
            target=lambda messages: 'ok',
            llm_config=LLMCallConfig(model='test'),
            upload_results=False,
            executive_summary=False,
            recommendations=False,
            **extra,
        )
        == []
    )
    assert configs[0].raise_on_execution_failure is True
