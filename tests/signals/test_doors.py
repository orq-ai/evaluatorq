"""Opt-in door classification tests with a fake judge."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import APIStatusError, BadRequestError, InternalServerError, RateLimitError

from evaluatorq.common import model_catalogue, redact
from evaluatorq.common.judge import EvaluatorResponsePayload, JudgeError, JudgeOutcome
from evaluatorq.signals import classify_tool_doors
from evaluatorq.signals.config import SignalsConfig

from tests.common.fake_orq import FakeOrq

from .conftest import agent, call, traj, user


class _Client:
    """Marker fake passed through to the patched judge."""


_CLIENT: Any = _Client()
_SECRET = 'sk-live-0123456789abcdef'


@pytest.fixture
def orq() -> FakeOrq:
    """A fake PII endpoint; every test that classifies a call passes it, because a call is redacted before Jev sees it."""
    return FakeOrq()


def _fake_judge(monkeypatch: pytest.MonkeyPatch, doors: dict[str, Any]) -> list[dict[str, Any]]:
    """Answer by command (shell) or tool name; record every question's state."""
    seen: list[dict[str, Any]] = []

    async def judge(**kwargs: Any) -> JudgeOutcome:
        state = kwargs['classify'].state
        seen.append(state)
        key = next(k for k in (state.get('command'), state.get('arguments'), state['tool_name']) if k in doors)
        answer = doors[key]
        if isinstance(answer, JudgeOutcome):
            return answer
        return JudgeOutcome(payload=EvaluatorResponsePayload(value=answer, explanation='classified'))

    monkeypatch.setattr('evaluatorq.signals.doors.run_judge', judge)
    return seen


@pytest.mark.asyncio
async def test_shell_calls_are_classified_once_per_distinct_command(monkeypatch: pytest.MonkeyPatch, orq: FakeOrq) -> None:
    seen = _fake_judge(monkeypatch, {'git status': 'benign', 'git push --force': 'one_way'})
    trajectory = traj([
        user(),
        agent(calls=[call('Bash', {'command': 'git status'}, 'a'), call('Bash', {'command': 'git push --force'}, 'b')]),
        agent(calls=[call('Bash', {'command': 'git push --force'}, 'c')]),
    ])

    results = await classify_tool_doors(trajectory, client=_CLIENT, orq=orq.client)

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
async def test_non_shell_tools_are_classified_per_distinct_arguments_with_the_description(
    monkeypatch: pytest.MonkeyPatch, orq: FakeOrq
) -> None:
    seen = _fake_judge(
        monkeypatch,
        {'{"file_path": "a.py"}': 'two_way', '{"file_path": "b.py"}': 'two_way', '{"text": "hi"}': 'one_way'},
    )
    trajectory = traj(
        [
            user(),
            agent(
                calls=[
                    call('Edit', {'file_path': 'a.py'}, 'a'),
                    call('Edit', {'file_path': 'b.py'}, 'b'),
                    call('Edit', {'file_path': 'a.py'}, 'repeat'),
                ]
            ),
            agent(calls=[call('mcp__slack__send_message', {'text': 'hi'}, 'c')]),
        ],
        tool_definitions=[
            {'type': 'function', 'function': {'name': 'mcp__slack__send_message', 'description': 'Post to Slack.'}},
            {'name': 'NeverCalled', 'description': 'Not used in this run.'},
        ],
    )

    results = await classify_tool_doors(trajectory, client=_CLIENT, orq=orq.client)

    assert seen == [
        {'tool_name': 'Edit', 'arguments': '{"file_path": "a.py"}'},
        {'tool_name': 'Edit', 'arguments': '{"file_path": "b.py"}'},
        {
            'tool_name': 'mcp__slack__send_message',
            'tool_description': 'Post to Slack.',
            'arguments': '{"text": "hi"}',
        },
    ]
    assert results['two_way_call_count'].value == 3
    assert [e.reason for e in results['two_way_call_count'].evidence] == [
        'Edit: {"file_path": "a.py"}',
        'Edit: {"file_path": "b.py"}',
        'Edit: {"file_path": "a.py"}',
    ]
    assert results['one_way_call_count'].value == 1


@pytest.mark.asyncio
async def test_custom_shell_role_and_long_commands_are_clipped(monkeypatch: pytest.MonkeyPatch, orq: FakeOrq) -> None:
    long_command = 'echo start\n' + 'x' * 10_000 + '\ngit push --force'
    seen: list[dict[str, Any]] = []

    async def judge(**kwargs: Any) -> JudgeOutcome:
        seen.append(kwargs['classify'].state)
        return JudgeOutcome(payload=EvaluatorResponsePayload(value='one_way', explanation='classified'))

    monkeypatch.setattr('evaluatorq.signals.doors.run_judge', judge)
    config = SignalsConfig(tool_roles={'run': 'bash'})
    trajectory = traj([user(), agent(calls=[call('run', {'cmd': long_command}, 'a')])])

    results = await classify_tool_doors(trajectory, config, client=_CLIENT, orq=orq.client)

    sent = seen[0]['command']
    assert len(sent) < 4_100
    assert sent.startswith('echo start')
    assert sent.endswith('git push --force')
    assert results['one_way_call_count'].value == 1


@pytest.mark.asyncio
async def test_run_without_tool_calls_makes_no_requests(monkeypatch: pytest.MonkeyPatch, orq: FakeOrq) -> None:
    seen = _fake_judge(monkeypatch, {})

    results = await classify_tool_doors(traj([user(), agent()]), client=_CLIENT, orq=orq.client)

    assert seen == []
    assert orq.requests == []
    assert results['one_way_call_count'].value == 0
    assert results['two_way_call_count'].value == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'outcome',
    [
        JudgeOutcome(error_kind=JudgeError.PARSE, error_message='bad answer'),
        JudgeOutcome(payload=EvaluatorResponsePayload(value=None, explanation='abstained', abstain=True)),
        JudgeOutcome(payload=EvaluatorResponsePayload(value='maybe', explanation='invalid')),
        JudgeOutcome(payload=EvaluatorResponsePayload(value='unknown', explanation='not a label Jev may choose')),
    ],
)
async def test_failed_classification_is_unknown_and_counts_are_kept(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    orq: FakeOrq,
    outcome: JudgeOutcome,
) -> None:
    _fake_judge(monkeypatch, {'rm -rf build': outcome, 'git push': 'one_way'})
    trajectory = traj([
        user(),
        agent(calls=[call('Bash', {'command': 'rm -rf build'}, 'a'), call('Bash', {'command': 'git push'}, 'b')]),
    ])

    results = await classify_tool_doors(trajectory, client=_CLIENT, orq=orq.client)

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
async def test_response_local_tool_definitions_supply_the_description(
    monkeypatch: pytest.MonkeyPatch, orq: FakeOrq
) -> None:
    seen = _fake_judge(monkeypatch, {'deploy': 'one_way'})
    trajectory = traj([
        user(),
        agent(
            calls=[call('deploy', {'env': 'prod'}, 'a')],
            extra={'evaluatorq.responses_tools': [{'name': 'deploy', 'description': 'Deploy to production.'}]},
        ),
    ])

    results = await classify_tool_doors(trajectory, client=_CLIENT, orq=orq.client)

    assert seen == [
        {'tool_name': 'deploy', 'tool_description': 'Deploy to production.', 'arguments': '{"env": "prod"}'}
    ]
    assert results['one_way_call_count'].value == 1


@pytest.mark.asyncio
async def test_first_description_seen_wins_across_responses(monkeypatch: pytest.MonkeyPatch, orq: FakeOrq) -> None:
    seen = _fake_judge(monkeypatch, {'deploy': 'one_way'})

    def step(description: str, call_id: str) -> Any:
        return agent(
            calls=[call('deploy', {}, call_id)],
            extra={'evaluatorq.responses_tools': [{'name': 'deploy', 'description': description}]},
        )

    await classify_tool_doors(traj([user(), step('First.', 'a'), step('Second.', 'b')]), client=_CLIENT, orq=orq.client)

    assert seen == [{'tool_name': 'deploy', 'tool_description': 'First.', 'arguments': '{}'}]


@pytest.mark.asyncio
async def test_unmapped_tool_activity_leaves_every_count_without_basis(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_judge(monkeypatch, {})
    trajectory = traj([
        user(),
        agent(extra={'evaluatorq.responses_output_items': [{'type': 'mcp_call', 'name': 'delete_repo'}]}),
    ])

    results = await classify_tool_doors(trajectory, client=_CLIENT)

    assert seen == []
    assert set(results) == {'one_way_call_count', 'two_way_call_count', 'unknown_call_count'}
    for item in results.values():
        assert item.value is None
        assert item.no_basis is not None
        assert 'delete_repo' in item.no_basis


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
async def test_transient_errors_are_retried_then_succeed(jev_router: list[float], orq: FakeOrq) -> None:
    client = _router_client(
        _status_error(RateLimitError, 429), _status_error(InternalServerError, 503), _choice('one_way')
    )

    results = await classify_tool_doors(_one_shell_call(), client=client, orq=orq.client)

    assert client.post.await_count == 3
    assert len(jev_router) == 2
    assert jev_router[1] > jev_router[0]
    assert results['one_way_call_count'].value == 1
    assert results['unknown_call_count'].value == 0


@pytest.mark.asyncio
async def test_exhausted_retries_become_unknown_with_counts_returned(jev_router: list[float], orq: FakeOrq) -> None:
    client = _router_client(*(_status_error(InternalServerError, 500) for _ in range(4)))

    results = await classify_tool_doors(_one_shell_call(), client=client, orq=orq.client)

    assert client.post.await_count == 4
    assert len(jev_router) == 3
    assert {name: r.value for name, r in results.items()} == {
        'one_way_call_count': 0,
        'two_way_call_count': 0,
        'unknown_call_count': 1,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize('first', ['bad_request', 'off_label_reply'])
async def test_non_transient_error_is_not_retried(jev_router: list[float], orq: FakeOrq, first: str) -> None:
    failure = _status_error(BadRequestError, 400) if first == 'bad_request' else _choice('maybe')
    client = _router_client(failure, _choice('one_way'))

    results = await classify_tool_doors(_one_shell_call(), client=client, orq=orq.client)

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
    assert [e.reason for e in results['unknown_call_count'].evidence] == ['Read']
    assert 'no credentials' in caplog.text


# --- Redaction: what reaches Jev, the logs and the evidence. ---


def _questions(monkeypatch: pytest.MonkeyPatch, answer: str = 'benign') -> list[dict[str, Any]]:
    """Answer every question with `answer` and return the state of each one asked."""
    seen: list[dict[str, Any]] = []

    async def judge(**kwargs: Any) -> JudgeOutcome:
        seen.append(kwargs['classify'].state)
        return JudgeOutcome(payload=EvaluatorResponsePayload(value=answer, explanation='classified'))

    monkeypatch.setattr('evaluatorq.signals.doors.run_judge', judge)
    return seen


@pytest.mark.asyncio
async def test_secrets_in_a_command_reach_neither_jev_nor_the_logs_nor_the_evidence(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, orq: FakeOrq
) -> None:
    seen: list[dict[str, Any]] = []

    async def failing_judge(**kwargs: Any) -> JudgeOutcome:
        seen.append(kwargs['classify'].state)
        return JudgeOutcome(error_kind=JudgeError.PARSE, error_message='bad answer')  # logged with the subject

    monkeypatch.setattr('evaluatorq.signals.doors.run_judge', failing_judge)
    command = f'curl -X DELETE -H "Authorization: Bearer {_SECRET}" https://api.example.com/repo && echo done'
    trajectory = traj([user(), agent(calls=[call('Bash', {'command': command}, 'a')])])

    results = await classify_tool_doors(trajectory, client=_CLIENT, orq=orq.client)

    assert orq.requests == [command]  # the PII endpoint is the one place the raw command goes
    assert seen == [{'tool_name': 'Bash', 'command': command.replace(_SECRET, '<API_KEY>')}]
    assert _SECRET not in caplog.text
    assert 'curl -X DELETE' in caplog.text  # the redacted command is still logged for diagnosis
    [evidence] = results['unknown_call_count'].evidence
    assert _SECRET not in evidence.reason
    assert '<API_KEY>' in evidence.reason


@pytest.mark.asyncio
async def test_secrets_in_tool_arguments_are_redacted_before_jev(
    monkeypatch: pytest.MonkeyPatch, orq: FakeOrq
) -> None:
    seen = _questions(monkeypatch)
    trajectory = traj([user(), agent(calls=[call('http_request', {'url': 'https://x.test', 'key': _SECRET}, 'a')])])

    await classify_tool_doors(trajectory, client=_CLIENT, orq=orq.client)

    assert seen == [{'tool_name': 'http_request', 'arguments': '{"key": "<API_KEY>", "url": "https://x.test"}'}]


@pytest.mark.asyncio
async def test_calls_of_one_tool_with_destructive_and_benign_arguments_get_separate_questions(
    monkeypatch: pytest.MonkeyPatch, orq: FakeOrq
) -> None:
    _fake_judge(monkeypatch, {'{"action": "list"}': 'benign', '{"action": "delete_all"}': 'one_way'})
    trajectory = traj([
        user(),
        agent(calls=[call('db', {'action': 'list'}, 'a'), call('db', {'action': 'delete_all'}, 'b')]),
    ])

    results = await classify_tool_doors(trajectory, client=_CLIENT, orq=orq.client)

    assert [e.call_id for e in results['one_way_call_count'].evidence] == ['b']
    assert results['unknown_call_count'].value == 0


@pytest.mark.asyncio
async def test_calls_that_differ_only_in_an_id_or_email_share_one_question(
    monkeypatch: pytest.MonkeyPatch, orq: FakeOrq
) -> None:
    seen = _questions(monkeypatch, 'one_way')
    ids = ['8f14e45f-ceea-4672-9d4c-3a1f6d0b5c11', '1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed']
    trajectory = traj([
        user(),
        agent(
            calls=[
                call('send_mail', {'to': 'ann@example.com', 'ref': ids[0]}, 'a'),
                call('send_mail', {'to': 'bob@example.org', 'ref': ids[1]}, 'b'),
                call('Bash', {'command': f'rm -rf /tmp/{ids[0]}'}, 'c'),
                call('Bash', {'command': f'rm -rf /tmp/{ids[1]}'}, 'd'),
            ]
        ),
    ])

    results = await classify_tool_doors(trajectory, client=_CLIENT, orq=orq.client)

    assert seen == [
        {'tool_name': 'send_mail', 'arguments': '{"ref": "<UUID>", "to": "<EMAIL_ADDRESS>"}'},
        {'tool_name': 'Bash', 'command': 'rm -rf /tmp/<UUID>'},
    ]
    assert results['one_way_call_count'].value == 4


@pytest.mark.asyncio
async def test_calls_that_could_not_be_redacted_are_unknown_and_never_sent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    seen = _questions(monkeypatch)
    orq = FakeOrq(error=RuntimeError(f'down {_SECRET}'))
    trajectory = traj([
        user(),
        agent(calls=[call('Bash', {'command': f'echo {_SECRET}'}, 'a'), call('Edit', {'path': 'x'}, 'b')]),
    ])

    results = await classify_tool_doors(trajectory, client=_CLIENT, orq=orq.client)

    assert seen == []
    assert {name: r.value for name, r in results.items()} == {
        'one_way_call_count': 0,
        'two_way_call_count': 0,
        'unknown_call_count': 2,
    }
    assert [e.reason for e in results['unknown_call_count'].evidence] == ['Bash', 'Edit']
    assert 'skipped 1 call(s) of Bash' in caplog.text
    assert 'skipped 1 call(s) of Edit' in caplog.text
    assert _SECRET not in caplog.text


@pytest.mark.asyncio
async def test_without_an_orq_client_every_call_is_unknown_and_nothing_is_sent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    seen = _questions(monkeypatch)

    results = await classify_tool_doors(_one_shell_call(), client=_CLIENT)  # the client does not route through Orq

    assert seen == []
    assert results['unknown_call_count'].value == 1
    assert 'PII redaction is unavailable' in caplog.text


@pytest.mark.asyncio
async def test_the_orq_client_is_built_from_the_llm_client_and_closed_only_when_derived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _questions(monkeypatch, 'one_way')
    built: list[tuple[str, str]] = []
    closed: list[object] = []
    derived = FakeOrq()

    def resolve(api_key: str, base_url: str) -> FakeOrq:
        built.append((api_key, base_url))
        return derived

    async def close(client: object) -> None:
        closed.append(client)

    monkeypatch.setattr(redact, 'resolve_orq_client', resolve)
    monkeypatch.setattr(redact, 'close_orq_client', close)
    llm = _router_client()
    llm.api_key = 'router-key'

    await classify_tool_doors(_one_shell_call(), client=llm)

    assert built == [('router-key', 'https://my.orq.ai')]
    assert closed == [derived]
    assert seen == [{'tool_name': 'Bash', 'command': 'git push --force'}]

    closed.clear()
    injected = FakeOrq()
    await classify_tool_doors(_one_shell_call(), client=llm, orq=injected.client)

    assert closed == []  # an injected Orq client stays with its owner
    assert injected.requests == ['git push --force']
