"""Opt-in door classification tests with a fake judge."""

from __future__ import annotations

from typing import Any

import pytest

from evaluatorq.common.judge import EvaluatorResponsePayload, JudgeError, JudgeOutcome
from evaluatorq.signals import classify_tool_doors
from evaluatorq.signals.config import SignalsConfig

from .conftest import agent, call, traj, user


class _Client:
    """Marker fake passed through to the patched judge."""


_CLIENT: Any = _Client()


def _fake_judge(monkeypatch: pytest.MonkeyPatch, doors: dict[str, Any]) -> list[dict[str, Any]]:
    """Answer by command (shell) or tool name; record every question's state."""
    seen: list[dict[str, Any]] = []

    async def judge(**kwargs: Any) -> JudgeOutcome:
        state = kwargs['classify'].state
        seen.append(state)
        answer = doors[state.get('command', state['tool_name'])]
        if isinstance(answer, JudgeOutcome):
            return answer
        return JudgeOutcome(payload=EvaluatorResponsePayload(value=answer, explanation='classified'))

    monkeypatch.setattr('evaluatorq.signals.doors.run_judge', judge)
    return seen


@pytest.mark.asyncio
async def test_shell_calls_are_classified_once_per_distinct_command(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_judge(monkeypatch, {'git status': 'benign', 'git push --force': 'one_way'})
    trajectory = traj([
        user(),
        agent(calls=[call('Bash', {'command': 'git status'}, 'a'), call('Bash', {'command': 'git push --force'}, 'b')]),
        agent(calls=[call('Bash', {'command': 'git push --force'}, 'c')]),
    ])

    results = await classify_tool_doors(trajectory, client=_CLIENT)

    assert seen == [
        {'tool_name': 'Bash', 'command': 'git status'},
        {'tool_name': 'Bash', 'command': 'git push --force'},
    ]
    one_way = results['one_way_call_count']
    assert one_way.value == 2
    assert [(e.step_id, e.call_id, e.reason, e.subgroup) for e in one_way.evidence] == [
        (2, 'b', 'Bash: git push --force', 'one_way'),
        (3, 'c', 'Bash: git push --force', 'one_way'),
    ]
    assert results['two_way_call_count'].value == 0


@pytest.mark.asyncio
async def test_non_shell_tools_are_classified_once_by_name_and_description(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_judge(monkeypatch, {'Edit': 'two_way', 'mcp__slack__send_message': 'one_way'})
    trajectory = traj(
        [
            user(),
            agent(calls=[call('Edit', {'file_path': 'a.py'}, 'a'), call('Edit', {'file_path': 'b.py'}, 'b')]),
            agent(calls=[call('mcp__slack__send_message', {'text': 'hi'}, 'c')]),
        ],
        tool_definitions=[
            {'type': 'function', 'function': {'name': 'mcp__slack__send_message', 'description': 'Post to Slack.'}},
            {'name': 'NeverCalled', 'description': 'Not used in this run.'},
        ],
    )

    results = await classify_tool_doors(trajectory, client=_CLIENT)

    assert seen == [
        {'tool_name': 'Edit'},
        {'tool_name': 'mcp__slack__send_message', 'tool_description': 'Post to Slack.'},
    ]
    assert results['two_way_call_count'].value == 2
    assert [e.reason for e in results['two_way_call_count'].evidence] == ['Edit', 'Edit']
    assert results['one_way_call_count'].value == 1


@pytest.mark.asyncio
async def test_custom_shell_role_and_long_commands_are_clipped(monkeypatch: pytest.MonkeyPatch) -> None:
    long_command = 'echo start\n' + 'x' * 10_000 + '\ngit push --force'
    seen: list[dict[str, Any]] = []

    async def judge(**kwargs: Any) -> JudgeOutcome:
        seen.append(kwargs['classify'].state)
        return JudgeOutcome(payload=EvaluatorResponsePayload(value='one_way', explanation='classified'))

    monkeypatch.setattr('evaluatorq.signals.doors.run_judge', judge)
    config = SignalsConfig(tool_roles={'run': 'bash'})
    trajectory = traj([user(), agent(calls=[call('run', {'cmd': long_command}, 'a')])])

    results = await classify_tool_doors(trajectory, config, client=_CLIENT)

    sent = seen[0]['command']
    assert len(sent) < 4_100
    assert sent.startswith('echo start')
    assert sent.endswith('git push --force')
    assert results['one_way_call_count'].value == 1


@pytest.mark.asyncio
async def test_run_without_tool_calls_makes_no_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_judge(monkeypatch, {})

    results = await classify_tool_doors(traj([user(), agent()]), client=_CLIENT)

    assert seen == []
    assert results['one_way_call_count'].value == 0
    assert results['two_way_call_count'].value == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'outcome',
    [
        JudgeOutcome(error_kind=JudgeError.PARSE, error_message='bad answer'),
        JudgeOutcome(payload=EvaluatorResponsePayload(value=None, explanation='abstained', abstain=True)),
        JudgeOutcome(payload=EvaluatorResponsePayload(value='maybe', explanation='invalid')),
    ],
)
async def test_failed_classification_is_no_basis_not_benign(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    outcome: JudgeOutcome,
) -> None:
    _fake_judge(monkeypatch, {'rm -rf build': outcome, 'Read': 'benign'})
    trajectory = traj([
        user(),
        agent(calls=[call('Bash', {'command': 'rm -rf build'}, 'a'), call('Read', {'file_path': 'x'}, 'b')]),
    ])

    results = await classify_tool_doors(trajectory, client=_CLIENT)

    for result in results.values():
        assert result.value is None
        assert result.no_basis == 'door_classification: 1 of 2 tool calls could not be classified'
    assert 'rm -rf build' in caplog.text


@pytest.mark.asyncio
async def test_unavailable_client_leaves_every_call_unclassified(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def no_client(**_: Any) -> Any:
        raise RuntimeError('no credentials')

    monkeypatch.setattr('evaluatorq.signals.doors.resolve_llm_client', no_client)
    trajectory = traj([user(), agent(calls=[call('Read', {'file_path': 'x'}, 'a')])])

    results = await classify_tool_doors(trajectory)

    assert results['one_way_call_count'].no_basis is not None
    assert 'no credentials' in caplog.text
