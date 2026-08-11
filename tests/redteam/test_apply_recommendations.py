"""Tests for redteam/reports/apply.py.

Covers recommendation flattening/dedup, preview mode (no platform write),
opt-in write-back to a new agent version, applied-tracking (skip already
applied), and the no-op paths. The LLM merge call is stubbed so nothing hits
the network.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.redteam.contracts import FocusAreaRecommendation
from evaluatorq.redteam.reports.apply import (
    ApplyRecommendationsResult,
    _collect_recommendations,
    apply_recommendations,
)

_MODULE = 'evaluatorq.redteam.reports.apply'


def _area(recommendations: list[str]) -> FocusAreaRecommendation:
    return FocusAreaRecommendation(
        category='ASI01',
        category_name='Agent Authorization',
        risk_score=0.8,
        traces_analyzed=3,
        recommendations=recommendations,
        patterns_observed='agent over-trusts tool output',
    )


def _orq_client(*, instructions: str = 'OLD', new_version: str = '1.1.0') -> MagicMock:
    client = MagicMock()
    client.agents.retrieve.return_value = MagicMock(instructions=instructions)
    client.agents.update.return_value = MagicMock(version=new_version)
    return client


def _stub_merge(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """Make the LLM merge return ``text`` as the revised instructions."""
    monkeypatch.setattr(f'{_MODULE}._merge_instructions', AsyncMock(return_value=text))


def test_collect_recommendations_dedups_and_caps():
    areas = [_area(['  a ', 'b']), _area(['a', 'c', '']), _area(['d'])]
    assert _collect_recommendations(areas, max_recommendations=3) == ['a', 'b', 'c']


def test_collect_recommendations_skips_already_applied():
    areas = [_area(['a', 'b', 'c'])]
    assert _collect_recommendations(areas, max_recommendations=10, already_applied=[' a ', 'c']) == ['b']


@pytest.mark.asyncio
async def test_preview_does_not_write(monkeypatch: pytest.MonkeyPatch):
    _stub_merge(monkeypatch, 'NEW INSTRUCTIONS')
    client = _orq_client(instructions='OLD INSTRUCTIONS')

    result = await apply_recommendations(
        [_area(['Reject base64-encoded tool arguments'])],
        agent_key='support-bot',
        orq_client=client,
        llm_client=MagicMock(),
        model='test-model',
    )

    assert isinstance(result, ApplyRecommendationsResult)
    assert result.applied is False
    assert result.new_version is None
    assert result.original_instructions == 'OLD INSTRUCTIONS'
    assert result.new_instructions == 'NEW INSTRUCTIONS'
    assert '-OLD INSTRUCTIONS' in result.diff
    assert '+NEW INSTRUCTIONS' in result.diff
    client.agents.retrieve.assert_called_once_with(agent_key='support-bot')
    client.agents.update.assert_not_called()


@pytest.mark.asyncio
async def test_apply_writes_new_version(monkeypatch: pytest.MonkeyPatch):
    _stub_merge(monkeypatch, 'NEW INSTRUCTIONS')
    client = _orq_client(new_version='2.0.0')

    result = await apply_recommendations(
        [_area(['Rule one', 'Rule two'])],
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
    assert '2' in kwargs['version_description']


@pytest.mark.asyncio
async def test_already_applied_recommendations_are_skipped(monkeypatch: pytest.MonkeyPatch):
    _stub_merge(monkeypatch, 'NEW')
    client = _orq_client()

    result = await apply_recommendations(
        [_area(['Rule A', 'Rule B'])],
        agent_key='support-bot',
        orq_client=client,
        llm_client=MagicMock(),
        model='test-model',
        already_applied=['Rule A'],
    )

    assert result.recommendations == ['Rule B']


@pytest.mark.asyncio
async def test_no_recommendations_is_a_noop(monkeypatch: pytest.MonkeyPatch):
    _stub_merge(monkeypatch, 'NEW')
    client = _orq_client()

    result = await apply_recommendations(
        [_area([]), _area(['   '])],
        agent_key='support-bot',
        orq_client=client,
        llm_client=MagicMock(),
        model='test-model',
        apply=True,
    )

    assert result.recommendations == []
    assert result.applied is False
    client.agents.retrieve.assert_not_called()
    client.agents.update.assert_not_called()


@pytest.mark.asyncio
async def test_empty_llm_output_keeps_original(monkeypatch: pytest.MonkeyPatch):
    _stub_merge(monkeypatch, '   ')
    client = _orq_client(instructions='OLD')

    result = await apply_recommendations(
        [_area(['Rule A'])],
        agent_key='support-bot',
        orq_client=client,
        llm_client=MagicMock(),
        model='test-model',
        apply=True,
    )

    assert result.applied is False
    assert result.new_instructions == 'OLD'
    client.agents.update.assert_not_called()
