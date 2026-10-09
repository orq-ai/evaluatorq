# ruff: noqa: S101, RUF029, S106

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from evaluatorq.common import cli_oauth


class _Answer(BaseModel):
    answer: str


class _Process:
    def __init__(self, output: str, returncode: int = 0, stderr: str = '') -> None:
        self.stdout = output.encode()
        self.stderr = stderr.encode()
        self.returncode = returncode
        self.stdin = None
        self.received: bytes | None = None

    async def communicate(self, body: bytes | None) -> tuple[bytes, bytes]:
        self.received = body
        return self.stdout, self.stderr


@pytest.mark.asyncio
async def test_cli_parse_uses_strict_schema_and_returns_validated_model(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    async def fake_create(*_args: str, **kwargs: Any) -> _Process:
        process = _Process('{"choices":[{"message":{"content":"{\\"answer\\":\\"ready\\"}"}}]}')
        seen['process'] = process
        return process

    monkeypatch.setattr(cli_oauth.shutil, 'which', lambda _: '/usr/local/bin/orq')
    monkeypatch.setattr(cli_oauth.asyncio, 'create_subprocess_exec', fake_create)
    _, llm = cli_oauth.build_cli_oauth_clients('https://my.orq.ai')
    result = await llm.chat.completions.parse(model='openai/gpt-6-luna', messages=[], response_format=_Answer)

    payload = json.loads(seen['process'].received)
    assert payload['response_format']['json_schema']['schema']['additionalProperties'] is False
    assert result.choices[0].message.parsed == _Answer(answer='ready')


@pytest.mark.asyncio
async def test_cli_classify_post_uses_oauth_request_endpoint(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    async def fake_create(*args: str, **kwargs: Any) -> _Process:
        seen['args'] = args
        seen['env'] = kwargs['env']
        process = _Process('{"body":{"answers":{}},"ok":true,"status":200}')
        seen['process'] = process
        return process

    monkeypatch.setattr(cli_oauth.shutil, 'which', lambda _: '/usr/local/bin/orq')
    monkeypatch.setattr(cli_oauth.asyncio, 'create_subprocess_exec', fake_create)
    monkeypatch.setenv('ORQ_API_KEY', 'must-not-be-forwarded')
    _, llm = cli_oauth.build_cli_oauth_clients('https://my.orq.ai')
    result = await llm.post(
        '/classify',
        body={'model': 'typesafe/jev-latest', 'state': {}, 'questions': {}},
        options={'headers': {'Authorization': 'Bearer secret-token', 'X-Api-Key': 'secret-key', 'X-Trace': 'trace-1'}},
    )

    assert result.answers == {}
    assert seen['args'][-5:] == ('request', 'POST', '/v3/router/classify', '--force', '--stdin')
    assert not any('secret-token' in arg or 'secret-key' in arg or 'X-Trace' in arg for arg in seen['args'])
    assert json.loads(seen['process'].received)['model'] == 'typesafe/jev-latest'
    assert 'ORQ_API_KEY' not in seen['env']


@pytest.mark.asyncio
async def test_cli_model_catalogue_uses_oauth_and_unwraps_response(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    async def fake_create(*args: str, **kwargs: Any) -> _Process:
        seen['args'] = args
        seen['env'] = kwargs['env']
        return _Process('{"body":[{"model_id":"gpt-6-luna"}],"ok":true,"status":200}')

    monkeypatch.setattr(cli_oauth.shutil, 'which', lambda _: '/usr/local/bin/orq')
    monkeypatch.setattr(cli_oauth.asyncio, 'create_subprocess_exec', fake_create)
    monkeypatch.setenv('ORQ_API_KEY', 'ambient-key-must-not-be-used')
    _, llm = cli_oauth.build_cli_oauth_clients('https://my.orq.ai')

    result = await llm.get_model_catalogue()

    assert result == [{'model_id': 'gpt-6-luna'}]
    assert seen['args'][-4:] == ('request', 'GET', '/v2/models', '--force')
    assert 'ORQ_API_KEY' not in seen['env']


@pytest.mark.asyncio
async def test_cli_model_catalogue_preserves_auth_status(monkeypatch: Any) -> None:
    async def fake_create(*_: str, **__: Any) -> _Process:
        return _Process('{"body":{"error":"unauthorized"},"ok":false,"status":401}')

    monkeypatch.setattr(cli_oauth.shutil, 'which', lambda _: 'orq')
    monkeypatch.setattr(cli_oauth.asyncio, 'create_subprocess_exec', fake_create)
    _, llm = cli_oauth.build_cli_oauth_clients()

    with pytest.raises(cli_oauth.OrqCLIError) as error:
        await llm.get_model_catalogue()
    assert error.value.status_code == 401
    assert error.value.failure_kind == 'authentication'


@pytest.mark.parametrize(
    'stderr',
    [
        'Error: missing API key; configure a profile with auth setup or set ORQ_API_KEY',
        'Error: authentication required',
        "Error: OAuth refresh token expired or was revoked; run 'orq auth login'",
    ],
)
@pytest.mark.asyncio
async def test_cli_catalogue_classifies_missing_or_expired_authentication(monkeypatch: Any, stderr: str) -> None:
    async def fake_create(*_: str, **__: Any) -> _Process:
        return _Process('', returncode=1, stderr=stderr)

    monkeypatch.setattr(cli_oauth.shutil, 'which', lambda _: 'orq')
    monkeypatch.setattr(cli_oauth.asyncio, 'create_subprocess_exec', fake_create)
    _, llm = cli_oauth.build_cli_oauth_clients()

    with pytest.raises(cli_oauth.OrqCLIError) as error:
        await llm.get_model_catalogue()
    assert error.value.failure_kind == 'authentication'


@pytest.mark.asyncio
async def test_cli_catalogue_keeps_unreadable_response_as_unavailable(monkeypatch: Any) -> None:
    async def fake_create(*_: str, **__: Any) -> _Process:
        return _Process('', returncode=1, stderr='connection refused')

    monkeypatch.setattr(cli_oauth.shutil, 'which', lambda _: 'orq')
    monkeypatch.setattr(cli_oauth.asyncio, 'create_subprocess_exec', fake_create)
    _, llm = cli_oauth.build_cli_oauth_clients()

    with pytest.raises(cli_oauth.OrqCLIError) as error:
        await llm.get_model_catalogue()
    assert error.value.failure_kind == 'unavailable'


@pytest.mark.parametrize('source', ['session', 'session-file', 'device-flow', 'future-cli-source'])
def test_oauth_subject_accepts_authenticated_cli_source(monkeypatch: Any, source: str) -> None:
    monkeypatch.setattr(cli_oauth.shutil, 'which', lambda _: 'orq')
    monkeypatch.setattr(
        cli_oauth.subprocess,
        'run',
        lambda *_args, **_kwargs: cli_oauth.subprocess.CompletedProcess(
            args=['orq'],
            returncode=0,
            stdout=json.dumps({'authenticated': True, 'source': source, 'user_id': 'user-1'}),
            stderr='',
        ),
    )

    assert cli_oauth.oauth_subject('https://my.orq.ai') == {
        'user_id': 'user-1',
        'workspace': None,
        'project': None,
    }


def test_oauth_subject_rejects_unauthenticated_cli(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli_oauth.shutil, 'which', lambda _: 'orq')
    monkeypatch.setattr(
        cli_oauth.subprocess,
        'run',
        lambda *_args, **_kwargs: cli_oauth.subprocess.CompletedProcess(
            args=['orq'],
            returncode=0,
            stdout=json.dumps({'authenticated': False, 'source': 'device-flow', 'user_id': 'user-1'}),
            stderr='',
        ),
    )

    with pytest.raises(cli_oauth.OrqCLIError, match='sign-in needs attention'):
        cli_oauth.oauth_subject('https://my.orq.ai')


@pytest.mark.parametrize(
    ('stderr', 'failure_kind', 'message'),
    [
        ('OAuth refresh token expired or was revoked; run orq auth login', 'authentication', 'sign-in needs attention'),
        ('HTTP 503 service unavailable', 'unavailable', 'Try again shortly'),
    ],
)
def test_oauth_subject_classifies_nonzero_whoami_from_sanitized_stderr(
    monkeypatch: Any, stderr: str, failure_kind: Literal['authentication', 'unavailable'], message: str
) -> None:
    monkeypatch.setattr(cli_oauth.shutil, 'which', lambda _: 'orq')
    monkeypatch.setattr(
        cli_oauth.subprocess,
        'run',
        lambda *_args, **_kwargs: cli_oauth.subprocess.CompletedProcess(
            args=['orq'],
            returncode=1,
            stdout='',
            stderr=stderr,
        ),
    )

    with pytest.raises(cli_oauth.OrqCLIError, match=message) as error:
        cli_oauth.oauth_subject('https://my.orq.ai')
    assert error.value.failure_kind == failure_kind


@pytest.mark.parametrize('stdout', ['{}', '{"authenticated":true}', '{"authenticated":"yes","user_id":"u"}'])
def test_oauth_subject_keeps_unknown_identity_shape_unavailable(monkeypatch: Any, stdout: str) -> None:
    monkeypatch.setattr(cli_oauth.shutil, 'which', lambda _: 'orq')
    monkeypatch.setattr(
        cli_oauth.subprocess,
        'run',
        lambda *_args, **_kwargs: cli_oauth.subprocess.CompletedProcess(
            args=['orq'],
            returncode=0,
            stdout=stdout,
            stderr='',
        ),
    )

    with pytest.raises(cli_oauth.OrqCLIError) as error:
        cli_oauth.oauth_subject('https://my.orq.ai')
    assert error.value.failure_kind == 'unavailable'
    assert 'Try again shortly' in str(error.value)


@pytest.mark.parametrize(
    ('message', 'expected'),
    [
        ('The Orq CLI OAuth sign-in needs attention.', 'authentication'),
        ('not signed in; run orq auth login', 'authentication'),
        ('missing API key', 'authentication'),
        ('The orq CLI is not installed.', 'setup'),
        ('connection refused', 'unavailable'),
    ],
)
def test_legacy_orq_cli_error_infers_failure_kind(message: str, expected: str) -> None:
    assert cli_oauth.OrqCLIError(message).failure_kind == expected


@pytest.mark.parametrize('failure_kind', ['setup', 'unavailable'])
def test_explicit_orq_cli_error_failure_kind_wins(failure_kind: Literal['setup', 'unavailable']) -> None:
    error = cli_oauth.OrqCLIError('not signed in; run orq auth login', failure_kind=failure_kind)

    assert error.failure_kind == failure_kind


@pytest.mark.asyncio
async def test_cli_trace_query_uses_json_stdin_and_does_not_pass_environment_credentials(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    async def fake_create(*args: str, **kwargs: Any) -> _Process:
        seen['args'] = args
        seen['env'] = kwargs['env']
        process = _Process('{"search":{"data":[]}}')
        seen['process'] = process
        return process

    monkeypatch.setattr(cli_oauth.shutil, 'which', lambda _: '/usr/local/bin/orq')
    monkeypatch.setattr(cli_oauth.asyncio, 'create_subprocess_exec', fake_create)
    monkeypatch.setenv('ORQ_API_KEY', 'must-not-be-forwarded')
    monkeypatch.setenv('ORQ_PROFILE', 'must-not-be-forwarded-either')
    client, _ = cli_oauth.build_cli_oauth_clients('https://orq.example', 'acme', 'project-1')
    captured: list[tuple[str, dict[str, Any]]] = []
    client.sdk_configuration._hooks.register_after_success_hook(  # noqa: SLF001
        lambda context, response: captured.append((context.operation_id, response.json()))
    )

    class SDKHook:
        def after_success(self, context: Any, response: Any) -> None:
            captured.append((context.operation_id, response.json()))

    client.sdk_configuration._hooks.register_after_success_hook(SDKHook())  # noqa: SLF001

    result = await client.traces.query_async(
        from_=datetime(2026, 1, 2, tzinfo=timezone.utc),
        to=datetime(2026, 1, 3, tzinfo=timezone.utc),
        oql='operation = "chat"',
        limit=25,
        page_token='page-2',
        timeout_ms=30000,
    )

    assert result.search.data == []
    assert captured == [
        ('TracesQueryOql', {'search': {'data': []}}),
        ('TracesQueryOql', {'search': {'data': []}}),
    ]
    assert seen['args'][:12] == (
        '/usr/local/bin/orq',
        '--no-input',
        '--output-format',
        'json',
        '--profile',
        '',
        '--server',
        'https://orq.example',
        '--workspace',
        'acme',
        '--project',
        'project-1',
    )
    assert seen['args'][12:] == ('traces', 'query-oql', '--stdin')
    assert json.loads(seen['process'].received) == {
        'from': '2026-01-02T00:00:00+00:00',
        'to': '2026-01-03T00:00:00+00:00',
        'oql': 'operation = "chat"',
        'limit': 25,
        'page_token': 'page-2',
    }
    assert 'ORQ_API_KEY' not in seen['env']
    assert 'ORQ_PROFILE' not in seen['env']


@pytest.mark.asyncio
async def test_cli_adapters_route_facets_projects_chat_and_embeddings(monkeypatch: Any) -> None:
    calls: list[tuple[tuple[str, ...], bytes | None]] = []

    async def fake_create(*args: str, **kwargs: Any) -> _Process:
        if 'chat' in args:
            output = '{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":2}}'
        elif 'embeddings' in args:
            output = '{"data":[{"embedding":[0.1,0.2]}],"usage":{"prompt_tokens":2}}'
        else:
            output = '{"data":[],"usage":{"input_tokens":1}}'
        process = _Process(output)
        old_communicate = process.communicate

        async def capture(body: bytes | None) -> tuple[bytes, bytes]:
            calls.append((args, body))
            return await old_communicate(body)

        monkeypatch.setattr(process, 'communicate', capture)
        return process

    monkeypatch.setattr(cli_oauth.shutil, 'which', lambda _: 'orq')
    monkeypatch.setattr(cli_oauth.asyncio, 'create_subprocess_exec', fake_create)
    orq, llm = cli_oauth.build_cli_oauth_clients()

    await orq.traces.list_facet_values_async(
        field='project_id', from_=datetime(2026, 1, 1, tzinfo=timezone.utc), limit=12
    )
    await orq.projects.list_async(limit=100, starting_after='p1')
    chat = await llm.chat.completions.create(model='openai/gpt-5.6-luna', messages=[{'role': 'user', 'content': 'hi'}])
    embeddings = await llm.embeddings.create(model='openai/text-embedding-3-small', input=['one', 'two'])
    response = await llm.responses.create(model='openai/gpt-5.6-luna', input='hello')

    assert calls[0][0][-7:] == (
        'traces',
        'list-facet-values',
        'project_id',
        '--from',
        '2026-01-01T00:00:00+00:00',
        '--limit',
        '12',
    )
    assert calls[1][0][-6:] == ('projects', 'list', '--limit', '100', '--starting-after', 'p1')
    assert calls[2][0][-3:] == ('chat', 'create', '--stdin')
    assert json.loads(calls[2][1] or b'{}')['model'] == 'openai/gpt-5.6-luna'
    assert calls[3][0][-3:] == ('embeddings', 'create', '--stdin')
    assert json.loads(calls[3][1] or b'{}')['input'] == ['one', 'two']
    assert calls[4][0][-3:] == ('responses', 'create', '--stdin')
    assert chat.choices[0].message.content == 'ok'
    assert chat.usage.prompt_tokens == 2
    assert embeddings.data[0].embedding == [0.1, 0.2]
    assert response.usage.input_tokens == 1


@pytest.mark.asyncio
async def test_cli_errors_redact_credential_shaped_text(monkeypatch: Any) -> None:
    async def fake_create(*_: str, **__: Any) -> _Process:
        return _Process('', returncode=1, stderr='request failed Bearer super-secret-token')

    monkeypatch.setattr(cli_oauth.shutil, 'which', lambda _: 'orq')
    monkeypatch.setattr(cli_oauth.asyncio, 'create_subprocess_exec', fake_create)
    _, llm = cli_oauth.build_cli_oauth_clients()

    with pytest.raises(cli_oauth.OrqCLIError, match=r'Bearer \[redacted\]'):
        await llm.embeddings.create(model='openai/text-embedding-3-small', input='hello')
