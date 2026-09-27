"""Public run-name alias for simulation results and trace labels."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from evaluatorq.simulation.api import generate_and_simulate, simulate


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('run_name', 'evaluation_name'),
    [
        ('support-replay', ''),
        (None, 'support-replay'),
        ('support-replay', 'support-replay'),
    ],
)
async def test_simulate_accepts_run_name_without_changing_legacy_keyword(
    monkeypatch: pytest.MonkeyPatch, run_name: str | None, evaluation_name: str
) -> None:
    captured: dict[str, Any] = {}

    async def fake_run(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(results=[])

    monkeypatch.setattr('evaluatorq.simulation.api._simulate_run', fake_run)

    assert await simulate(run_name=run_name, evaluation_name=evaluation_name) == []
    assert captured['evaluation_name'] == 'support-replay'


@pytest.mark.asyncio
async def test_simulate_rejects_conflicting_run_names(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(**kwargs: Any) -> None:
        pytest.fail('conflicting names must fail before the run starts')

    monkeypatch.setattr('evaluatorq.simulation.api._simulate_run', fake_run)

    with pytest.raises(ValueError, match='run_name and evaluation_name'):
        await simulate(run_name='new', evaluation_name='old')


@pytest.mark.asyncio
async def test_generate_and_simulate_accepts_run_name(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_run(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(results=[])

    monkeypatch.setattr('evaluatorq.simulation.api._generate_and_simulate_run', fake_run)

    assert await generate_and_simulate(run_name='support-generated') == []
    assert captured['evaluation_name'] == 'support-generated'


@pytest.mark.asyncio
async def test_generate_and_simulate_rejects_conflicting_run_names(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(**kwargs: Any) -> None:
        pytest.fail('conflicting names must fail before the run starts')

    monkeypatch.setattr('evaluatorq.simulation.api._generate_and_simulate_run', fake_run)

    with pytest.raises(ValueError, match='run_name and evaluation_name'):
        await generate_and_simulate(run_name='new', evaluation_name='old')
