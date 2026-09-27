# ruff: noqa: S101, RUF029, S106

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

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
    result = await llm.post('/classify', body={'model': 'typesafe/jev-latest', 'state': {}, 'questions': {}}, options={})

    assert result.answers == {}
    assert seen['args'][-5:] == ('request', 'POST', '/v3/router/classify', '--force', '--stdin')
    assert json.loads(seen['process'].received)['model'] == 'typesafe/jev-latest'
    assert 'ORQ_API_KEY' not in seen['env']


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

        process.communicate = capture
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
