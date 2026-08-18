"""Tests for the shared apply engine (common/apply.py).

Focus on the paths the surface-level tests stub over: the real merge call
routed through ``execute_chat_completion`` on the ``cfg is None`` (simulation)
path, the malformed-rewrite fallback that now reports a failed merge instead of
silently keeping the original, and the shared ``write_instructions`` write path.
The chat completion is stubbed so nothing hits the network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.common import apply as apply_mod
from evaluatorq.common.apply import apply_recommendations, write_instructions


def _completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _stub_chat(monkeypatch: pytest.MonkeyPatch, *contents: str) -> AsyncMock:
    """Stub execute_chat_completion to return the given contents in order."""
    mock = AsyncMock(side_effect=[(_completion(c), None) for c in contents])
    monkeypatch.setattr(apply_mod, 'execute_chat_completion', mock)
    return mock


def _orq_client(*, instructions: str = 'OLD RULES', new_version: str = '1.1.0') -> MagicMock:
    client = MagicMock()
    client.agents.retrieve.return_value = MagicMock(instructions=instructions)
    client.agents.update.return_value = MagicMock(version=new_version)
    return client


def test_parse_edits_accepts_fenced_json():
    content = '```json\n{"edits": [{"find": "OLD", "replace": "NEW"}]}\n```'

    assert apply_mod._parse_edits(content) == [{'find': 'OLD', 'replace': 'NEW'}]


@pytest.mark.asyncio
async def test_rewrite_fallback_accepts_fenced_json(monkeypatch: pytest.MonkeyPatch):
    _stub_chat(
        monkeypatch,
        '```json\n{"edits": []}\n```',
        '```json\n{"instructions": "NEW RULES"}\n```',
    )

    revised = await apply_mod._merge_instructions(
        MagicMock(),
        'test-model',
        'OLD RULES',
        ['Tighten the refund rule'],
    )

    assert revised == 'NEW RULES'


@pytest.mark.asyncio
async def test_cfg_none_path_merges_via_shared_wrapper(monkeypatch: pytest.MonkeyPatch):
    """The simulation path (cfg is None) exercises the real merge, not a stub."""
    chat = _stub_chat(monkeypatch, json.dumps({'edits': [{'find': 'OLD RULES', 'replace': 'NEW RULES'}]}))
    client = _orq_client(instructions='OLD RULES')

    result = await apply_recommendations(
        ['Tighten the refund rule'],
        agent_key='support-bot',
        orq_client=client,
        llm_client=MagicMock(),
        model='test-model',
    )

    assert result.new_instructions == 'NEW RULES'
    assert result.merge_failed is False
    # Went through the shared wrapper with a request timeout (not a raw create()).
    assert chat.await_count == 1
    kwargs = chat.await_args.kwargs if chat.await_args else {}
    assert kwargs['timeout_s'] == apply_mod._MERGE_TIMEOUT_S
    assert kwargs['span'] is None


@pytest.mark.asyncio
async def test_malformed_rewrite_json_reports_failed_merge(monkeypatch: pytest.MonkeyPatch):
    """Off-contract edits then non-JSON rewrite: no write, merge_failed=True."""
    _stub_chat(monkeypatch, json.dumps({'not_edits': 1}), 'this is not json')
    client = _orq_client(instructions='OLD RULES')

    result = await apply_recommendations(
        ['Add a guard'],
        agent_key='support-bot',
        orq_client=client,
        llm_client=MagicMock(),
        model='test-model',
        apply=True,
    )

    assert result.merge_failed is True
    assert result.applied is False
    assert result.new_instructions == 'OLD RULES'
    client.agents.update.assert_not_called()


@pytest.mark.asyncio
async def test_write_instructions_writes_new_minor_version(monkeypatch: pytest.MonkeyPatch):
    client = _orq_client(new_version='2.1.0')

    version = await write_instructions(client, 'support-bot', 'REVISED', version_description='note')

    assert version == '2.1.0'
    kwargs = client.agents.update.call_args.kwargs
    assert kwargs['agent_key'] == 'support-bot'
    assert kwargs['instructions'] == 'REVISED'
    assert kwargs['version_increment'] == 'minor'
    assert kwargs['version_description'] == 'note'


@pytest.mark.asyncio
async def test_write_instructions_none_version(monkeypatch: pytest.MonkeyPatch):
    client = MagicMock()
    client.agents.update.return_value = MagicMock(version=None)
    assert await write_instructions(client, 'k', 'X', version_description='n') is None
