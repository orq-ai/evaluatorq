"""Tests for simulation/reports/apply_suggestions.py.

Covers suggestion flattening/dedup, preview mode (no platform write), opt-in
write-back to a new agent version, and the no-op paths (no suggestions, empty
LLM output). The LLM merge call is stubbed so nothing hits the network.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.simulation.reports.apply import (
    ApplySuggestionsResult,
    _collect_suggestions,
    _RevisedInstructions,
    apply_suggestions,
)
from evaluatorq.simulation.types import SimulationRecommendation

_MODULE = 'evaluatorq.simulation.reports.apply'


def _rec(suggestions: list[str]) -> SimulationRecommendation:
    return SimulationRecommendation(
        result_index=0,
        persona='Alice',
        scenario='Billing',
        triggers=['rule_broken: leaked internal data'],
        suggestions=suggestions,
    )


def _orq_client(*, instructions: str = 'OLD', new_version: str = '1.1.0') -> MagicMock:
    client = MagicMock()
    client.agents.retrieve.return_value = MagicMock(instructions=instructions)
    client.agents.update.return_value = MagicMock(version=new_version)
    return client


def _stub_merge(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """Make the LLM merge return ``text`` as the revised instructions."""
    monkeypatch.setattr(
        f'{_MODULE}.generate_structured',
        AsyncMock(return_value=(_RevisedInstructions(instructions=text), '')),
    )


def test_collect_suggestions_dedups_and_caps():
    recs = [_rec(['  a ', 'b']), _rec(['a', 'c', '']), _rec(['d'])]
    assert _collect_suggestions(recs, max_suggestions=3) == ['a', 'b', 'c']


@pytest.mark.asyncio
async def test_preview_does_not_write(monkeypatch: pytest.MonkeyPatch):
    _stub_merge(monkeypatch, 'NEW INSTRUCTIONS')
    client = _orq_client(instructions='OLD INSTRUCTIONS')

    result = await apply_suggestions(
        [_rec(['Add a rule forbidding internal data'])],
        agent_key='support-bot',
        orq_client=client,
        llm_client=MagicMock(),
        model='test-model',
    )

    assert isinstance(result, ApplySuggestionsResult)
    assert result.applied is False
    assert result.new_version is None
    assert result.original_instructions == 'OLD INSTRUCTIONS'
    assert result.new_instructions == 'NEW INSTRUCTIONS'
    client.agents.retrieve.assert_called_once_with(agent_key='support-bot')
    client.agents.update.assert_not_called()


@pytest.mark.asyncio
async def test_apply_writes_new_version(monkeypatch: pytest.MonkeyPatch):
    _stub_merge(monkeypatch, 'NEW INSTRUCTIONS')
    client = _orq_client(new_version='2.0.0')

    result = await apply_suggestions(
        [_rec(['Add a rule', 'Add another rule'])],
        agent_key='support-bot',
        orq_client=client,
        llm_client=MagicMock(),
        model='test-model',
        apply=True,
    )

    assert result.applied is True
    assert result.new_version == '2.0.0'
    client.agents.update.assert_called_once()
    kwargs = client.agents.update.call_args.kwargs
    assert kwargs['agent_key'] == 'support-bot'
    assert kwargs['instructions'] == 'NEW INSTRUCTIONS'
    assert kwargs['version_increment'] == 'minor'
    assert '2' in kwargs['version_description']  # mentions the suggestion count


@pytest.mark.asyncio
async def test_no_suggestions_is_a_noop(monkeypatch: pytest.MonkeyPatch):
    _stub_merge(monkeypatch, 'NEW')
    client = _orq_client()

    result = await apply_suggestions(
        [_rec([]), _rec(['   '])],
        agent_key='support-bot',
        orq_client=client,
        llm_client=MagicMock(),
        model='test-model',
        apply=True,
    )

    assert result.suggestions == []
    assert result.applied is False
    client.agents.retrieve.assert_not_called()
    client.agents.update.assert_not_called()


@pytest.mark.asyncio
async def test_empty_llm_output_keeps_original(monkeypatch: pytest.MonkeyPatch):
    _stub_merge(monkeypatch, '   ')
    client = _orq_client(instructions='OLD')

    result = await apply_suggestions(
        [_rec(['Add a rule'])],
        agent_key='support-bot',
        orq_client=client,
        llm_client=MagicMock(),
        model='test-model',
        apply=True,
    )

    assert result.applied is False
    assert result.new_instructions == 'OLD'
    client.agents.update.assert_not_called()
