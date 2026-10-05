"""Opt-in door classification tests with a fake judge."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import APIStatusError, BadRequestError, InternalServerError, RateLimitError

from evaluatorq.common import model_catalogue
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
async def test_failed_classification_is_unknown_and_counts_are_kept(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    outcome: JudgeOutcome,
) -> None:
    _fake_judge(monkeypatch, {'rm -rf build': outcome, 'git push': 'one_way'})
    trajectory = traj([
        user(),
        agent(calls=[call('Bash', {'command': 'rm -rf build'}, 'a'), call('Bash', {'command': 'git push'}, 'b')]),
    ])

    results = await classify_tool_doors(trajectory, client=_CLIENT)

    assert {name: r.value for name, r in results.items()} == {
        'one_way_call_count': 1,
        'two_way_call_count': 0,
        'unknown_call_count': 1,
    }
    assert [(e.call_id, e.reason, e.subgroup) for e in results['unknown_call_count'].evidence] == [
        ('a', 'Bash: rm -rf build', 'unknown'),
    ]
    assert 'rm -rf build' in caplog.text


@pytest.mark.asyncio
async def test_unknown_is_never_offered_to_jev(monkeypatch: pytest.MonkeyPatch) -> None:
    labels: list[set[str]] = []

    async def judge(**kwargs: Any) -> JudgeOutcome:
        labels.append(set(kwargs['classify'].criteria))
        return JudgeOutcome(payload=EvaluatorResponsePayload(value='benign', explanation='classified'))

    monkeypatch.setattr('evaluatorq.signals.doors.run_judge', judge)

    await classify_tool_doors(traj([user(), agent(calls=[call('Read', {}, 'a')])]), client=_CLIENT)

    assert labels == [{'benign', 'two_way', 'one_way'}]


# --- Retry: through the real run_judge, with a fake router client and no backoff sleep. ---

_ORQ_URL = 'https://my.orq.ai/v3/router'


@pytest.fixture
def jev_router(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    """Route the classifier model to /classify and record backoff waits instead of sleeping."""
    model_catalogue.reset_catalogue_cache()
    entry = model_catalogue.ModelInfo(0.0, 0.0, 'typesafe', supports_responses=False, supports_classify=True)

    async def fake_load(client: Any = None) -> dict[str, Any]:  # noqa: ARG001
        return {'typesafe/jev-latest': entry, 'jev-latest': entry}

    waits: list[float] = []

    async def no_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(model_catalogue, '_load_catalogue', fake_load)
    monkeypatch.setattr('evaluatorq.common.retry.asyncio.sleep', no_sleep)
    monkeypatch.delenv('EVALUATORQ_CLASSIFIER_MODEL', raising=False)
    yield waits
    model_catalogue.reset_catalogue_cache()


def _router_client(*post_results: Any) -> Any:
    client = MagicMock()
    client.base_url = _ORQ_URL
    client.post = AsyncMock(side_effect=list(post_results))
    return client


def _status_error(cls: type[APIStatusError], status: int) -> APIStatusError:
    response = httpx.Response(status, request=httpx.Request('POST', f'{_ORQ_URL}/classify'))
    return cls('failed', response=response, body=None)


def _choice(label: str) -> dict[str, Any]:
    answer = {'type': 'choice', 'choice': label, 'probabilities': {label: 0.9}}
    return {'answers': {'verdict': answer}, 'usage': {'input_tokens': 10, 'output_tokens': 1}, 'model': 'jev-latest'}


def _one_shell_call() -> Any:
    return traj([user(), agent(calls=[call('Bash', {'command': 'git push --force'}, 'a')])])


@pytest.mark.asyncio
async def test_transient_errors_are_retried_then_succeed(jev_router: list[float]) -> None:
    client = _router_client(
        _status_error(RateLimitError, 429), _status_error(InternalServerError, 503), _choice('one_way')
    )

    results = await classify_tool_doors(_one_shell_call(), client=client)

    assert client.post.await_count == 3
    assert len(jev_router) == 2
    assert jev_router[1] > jev_router[0]
    assert results['one_way_call_count'].value == 1
    assert results['unknown_call_count'].value == 0


@pytest.mark.asyncio
async def test_exhausted_retries_become_unknown_with_counts_returned(jev_router: list[float]) -> None:
    client = _router_client(*(_status_error(InternalServerError, 500) for _ in range(4)))

    results = await classify_tool_doors(_one_shell_call(), client=client)

    assert client.post.await_count == 4
    assert len(jev_router) == 3
    assert {name: r.value for name, r in results.items()} == {
        'one_way_call_count': 0,
        'two_way_call_count': 0,
        'unknown_call_count': 1,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize('first', ['bad_request', 'off_label_reply'])
async def test_non_transient_error_is_not_retried(jev_router: list[float], first: str) -> None:
    failure = _status_error(BadRequestError, 400) if first == 'bad_request' else _choice('maybe')
    client = _router_client(failure, _choice('one_way'))

    results = await classify_tool_doors(_one_shell_call(), client=client)

    assert client.post.await_count == 1
    assert jev_router == []
    assert results['unknown_call_count'].value == 1


@pytest.mark.asyncio
async def test_unavailable_client_leaves_every_call_unclassified(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def no_client(**_: Any) -> Any:
        raise RuntimeError('no credentials')

    monkeypatch.setattr('evaluatorq.signals.doors.resolve_llm_client', no_client)
    trajectory = traj([user(), agent(calls=[call('Read', {'file_path': 'x'}, 'a')])])

    results = await classify_tool_doors(trajectory)

    assert results['unknown_call_count'].value == 1
    assert results['one_way_call_count'].value == 0
    assert 'no credentials' in caplog.text
