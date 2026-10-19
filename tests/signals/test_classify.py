"""Opt-in tool-role classification tests with a fake judge."""

from __future__ import annotations

from typing import Any

import pytest

from evaluatorq.common.judge import EvaluatorResponsePayload, JudgeError, JudgeOutcome
from evaluatorq.signals import classify_tool_roles
from evaluatorq.signals.config import SignalsConfig

from .conftest import agent, call, traj, user


class _Client:
    """Marker fake passed through to the patched judge."""


@pytest.mark.asyncio
async def test_skips_tools_that_already_have_explicit_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[dict[str, Any], str]] = []

    async def judge(**kwargs: Any) -> JudgeOutcome:
        question = kwargs['classify']
        requests.append((question.state, kwargs['model']))
        role = 'bash' if question.state['tool_name'] == 'custom_shell' else 'webview'
        return JudgeOutcome(payload=EvaluatorResponsePayload(value=role, explanation='classified'))

    monkeypatch.setattr('evaluatorq.signals.classify.run_judge', judge)
    client: Any = _Client()
    original = SignalsConfig(tool_roles={'custom_shell': 'skill', 'WebThing': 'other'})
    trajectory = traj([
        user(),
        agent(
            calls=[call('custom_shell', {'cmd': 'first'}), call('custom_shell', {'cmd': 'second'}), call('WebThing')]
        ),
    ])

    result = await classify_tool_roles([trajectory], original, client=client)

    assert result is original
    assert result.tool_roles == {'custom_shell': 'skill', 'WebThing': 'other'}
    assert requests == []


@pytest.mark.asyncio
async def test_classifies_unknown_tool_without_sending_its_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[dict[str, Any], str]] = []

    async def judge(**kwargs: Any) -> JudgeOutcome:
        question = kwargs['classify']
        seen.append((question.state, kwargs['model']))
        return JudgeOutcome(payload=EvaluatorResponsePayload(value='bash', explanation='classified'))

    monkeypatch.setattr('evaluatorq.signals.classify.run_judge', judge)
    config = SignalsConfig(tool_roles={'known': 'other'})
    trajectory = traj([user(), agent(calls=[call('custom', {'cmd': 'first'}), call('custom', {'cmd': 'later'})])])

    client: Any = _Client()
    result = await classify_tool_roles([trajectory], config, client=client)

    assert result.tool_roles == {'known': 'other', 'custom': 'bash'}
    assert seen == [({'tool_name': 'custom'}, config.classifier.model)]
    assert config.tool_roles == {'known': 'other'}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'outcome',
    [
        JudgeOutcome(error_kind=JudgeError.PARSE, error_message='bad answer'),
        JudgeOutcome(payload=EvaluatorResponsePayload(value=None, explanation='abstained', abstain=True)),
        JudgeOutcome(payload=EvaluatorResponsePayload(value='unknown', explanation='invalid')),
    ],
)
async def test_failed_abstained_or_invalid_classification_falls_back_to_other(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    outcome: JudgeOutcome,
) -> None:
    async def judge(**kwargs: Any) -> JudgeOutcome:
        return outcome

    monkeypatch.setattr('evaluatorq.signals.classify.run_judge', judge)

    client: Any = _Client()
    result = await classify_tool_roles(
        [traj([user(), agent(calls=[call('unclassified', {'x': 1})])])],
        client=client,
    )

    assert result.tool_roles['unclassified'] == 'other'
    assert 'unclassified' in caplog.text
