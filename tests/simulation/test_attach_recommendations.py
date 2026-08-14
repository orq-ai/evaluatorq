"""``_attach_recommendations`` degrades instead of taking the run down with it.

The docstring promises best-effort: missing credentials or a failing analysis call cost
the suggestions, never the results. All-success fakes would not prove that, so each
failure branch is exercised here.
"""

from __future__ import annotations

# ruff: noqa: S101
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.simulation import api
from evaluatorq.simulation.reports.recommendations import SimulationRecommendationConfig


def _run() -> Any:
    """A run stub carrying one result and no suggestions yet."""
    return SimpleNamespace(results=[MagicMock()], recommendations=None)


@pytest.mark.asyncio
async def test_missing_credentials_leaves_results_intact(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _no_creds() -> Any:
        raise RuntimeError('no LLM credentials configured')

    monkeypatch.setattr('evaluatorq.common.llm_client.resolve_llm_client', _no_creds)

    run = _run()
    with caplog.at_level(logging.WARNING):
        await api._attach_recommendations(run, SimulationRecommendationConfig(), 'test-model')  # noqa: SLF001

    assert run.recommendations is None
    assert len(run.results) == 1
    assert 'Failed to generate remediation suggestions' in caplog.text


@pytest.mark.asyncio
async def test_generation_failure_still_closes_an_owned_client(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = MagicMock()
    client.close = AsyncMock()
    monkeypatch.setattr(
        'evaluatorq.common.llm_client.resolve_llm_client',
        lambda *a, **k: SimpleNamespace(client=client, owned=True),
    )
    monkeypatch.setattr(
        'evaluatorq.simulation.reports.recommendations.generate_recommendations',
        AsyncMock(side_effect=RuntimeError('analysis call failed')),
    )

    run = _run()
    with caplog.at_level(logging.WARNING):
        await api._attach_recommendations(run, SimulationRecommendationConfig(), 'test-model')  # noqa: SLF001

    assert run.recommendations is None
    assert 'Failed to generate remediation suggestions' in caplog.text
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_suggestions_stays_none_rather_than_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty list would render as "recommendations section, zero rows" downstream."""
    monkeypatch.setattr(
        'evaluatorq.common.llm_client.resolve_llm_client',
        lambda *a, **k: SimpleNamespace(client=MagicMock(), owned=False),
    )
    monkeypatch.setattr(
        'evaluatorq.simulation.reports.recommendations.generate_recommendations',
        AsyncMock(return_value=[]),
    )

    run = _run()
    await api._attach_recommendations(run, SimulationRecommendationConfig(), 'test-model')  # noqa: SLF001

    assert run.recommendations is None


@pytest.mark.asyncio
async def test_unpersisted_run_warns_that_suggestions_are_dropped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """simulate() returns bare results, so with save/report both off the suggestions are
    paid for and discarded. The warning is not a skip — generation still happens."""
    generate = AsyncMock(return_value=[MagicMock()])
    monkeypatch.setattr(
        'evaluatorq.common.llm_client.resolve_llm_client',
        lambda *a, **k: SimpleNamespace(client=MagicMock(), owned=False),
    )
    monkeypatch.setattr('evaluatorq.simulation.reports.recommendations.generate_recommendations', generate)

    run = _run()
    with caplog.at_level(logging.WARNING):
        await api._attach_recommendations(run, SimulationRecommendationConfig(), 'test-model', persisted=False)  # noqa: SLF001

    assert 'generated and discarded' in caplog.text
    generate.assert_awaited_once()
    assert run.recommendations is not None


@pytest.mark.asyncio
async def test_persisted_run_does_not_warn(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(
        'evaluatorq.common.llm_client.resolve_llm_client',
        lambda *a, **k: SimpleNamespace(client=MagicMock(), owned=False),
    )
    monkeypatch.setattr(
        'evaluatorq.simulation.reports.recommendations.generate_recommendations',
        AsyncMock(return_value=[MagicMock()]),
    )

    with caplog.at_level(logging.WARNING):
        await api._attach_recommendations(_run(), SimulationRecommendationConfig(), 'test-model', persisted=True)  # noqa: SLF001

    assert 'generated and discarded' not in caplog.text
