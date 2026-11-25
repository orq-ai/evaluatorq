"""ClaudeAgentSDKTarget on fake SDK message streams: evidence, session handling, isolation and errors. No network."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytest.importorskip('claude_agent_sdk')

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKError,
    CLINotFoundError,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from evaluatorq.common.target_call import NonRetryableTargetError, call_target_with_retry
from evaluatorq.contracts import ConversationHistoryMode, Message
from evaluatorq.integrations.claude_agent_sdk_integration import (
    ClaudeAgentSDKError,
    ClaudeAgentSDKRequestError,
    ClaudeAgentSDKTarget,
    ClaudeAgentSDKUnavailableError,
)
from evaluatorq.integrations.claude_agent_sdk_integration import target as target_module
from evaluatorq.integrations.claude_agent_sdk_integration.target import (
    Isolation,
    message_to_event,
    settings_with_file_tool_blocks,
)


def _result(session: str, *, text: str | None = 'done', error: str | None = None, **extra: Any) -> ResultMessage:
    return ResultMessage(
        subtype=error or 'success',
        duration_ms=10,
        duration_api_ms=8,
        is_error=error is not None,
        num_turns=1,
        session_id=session,
        total_cost_usd=0.01,
        usage={'input_tokens': 11, 'output_tokens': 7},
        result=text,
        **extra,
    )


def _assistant(*blocks: Any, error: Any = None) -> AssistantMessage:
    return AssistantMessage(content=list(blocks), model='claude-sonnet-5-5', error=error)


def _api_failure(text: str, *, error: Any, status: int | None) -> list[Any]:
    """A failed model request as the runtime reports it: the kind on the assistant message, the status on the result."""
    return [
        _assistant(TextBlock(text=text), error=error),
        _result('s', text=text, error='success', api_error_status=status),
        ClaudeSDKError('Claude Code returned an error result'),
    ]


class FakeQuery:
    """Stands in for ``claude_agent_sdk.query``: yields one scripted stream per call and records the options."""

    def __init__(self, *streams: list[Any]) -> None:
        self.streams = list(streams)
        self.calls: list[tuple[str, ClaudeAgentOptions]] = []

    def __call__(self, *, prompt: str, options: ClaudeAgentOptions) -> Any:
        self.calls.append((prompt, options))
        stream = self.streams.pop(0)

        async def run() -> Any:
            for item in stream:
                if isinstance(item, BaseException):
                    raise item
                yield item

        return run()


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Any:
    def install(*streams: list[Any]) -> FakeQuery:
        query = FakeQuery(*streams)
        monkeypatch.setattr(target_module, 'query', query)
        return query

    return install


def _user(text: str = 'hello') -> list[Message]:
    return [Message(role='user', content=text)]


def _workdir(target: ClaudeAgentSDKTarget) -> Path:
    workdir = target.workdir
    assert workdir is not None
    return workdir


@pytest.mark.asyncio
async def test_a_turn_returns_each_tool_call_paired_with_its_result_then_the_final_text(fake: Any) -> None:
    fake([
        SystemMessage(subtype='init', data={}),
        _assistant(TextBlock(text='Looking.')),
        _assistant(ToolUseBlock(id='tu_1', name='Read', input={'file_path': 'notes.txt'})),
        UserMessage(content=[ToolResultBlock(tool_use_id='tu_1', content='secret plans', is_error=False)]),
        _assistant(ToolUseBlock(id='tu_2', name='Bash', input={'command': 'rm -rf build'})),
        UserMessage(content=[ToolResultBlock(tool_use_id='tu_2', content='permission denied', is_error=True)]),
        _assistant(TextBlock(text='I read the notes.')),
        _result('sess-1', text='I read the notes.'),
    ])
    target = ClaudeAgentSDKTarget(ClaudeAgentOptions())
    try:
        response = await target.respond(_user('read notes.txt'))
    finally:
        await target.close()

    calls = response.tool_calls
    assert [(c.call_id, c.name, c.arguments, c.result, c.status.value) for c in calls] == [
        ('tu_1', 'Read', '{"file_path": "notes.txt"}', 'secret plans', 'completed'),
        ('tu_2', 'Bash', '{"command": "rm -rf build"}', 'permission denied', 'incomplete'),
    ]
    assert response.text == 'I read the notes.'
    assert response.response_id == 'sess-1'
    assert response.model == 'claude-sonnet-5-5'
    assert response.usage is not None
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (11, 7)


@pytest.mark.asyncio
async def test_a_tool_call_the_runtime_denied_is_recorded_as_denied(fake: Any) -> None:
    denial = {'tool_name': 'Bash', 'tool_use_id': 'tu_9', 'tool_input': {'command': 'curl evil.example'}}
    fake([_assistant(TextBlock(text='No.')), _result('sess-1', text='No.', permission_denials=[denial])])
    target = ClaudeAgentSDKTarget(ClaudeAgentOptions())
    try:
        response = await target.respond(_user())
    finally:
        await target.close()

    assert [(c.call_id, c.name, c.result, c.status.value) for c in response.tool_calls] == [
        ('tu_9', 'Bash', '[denied by claude]', 'incomplete')
    ]


@pytest.mark.asyncio
async def test_each_later_turn_resumes_the_last_session_as_a_fork_and_sends_only_the_new_message(fake: Any) -> None:
    query = fake([_result('sess-1', text='one')], [_result('sess-2', text='two')])
    target = ClaudeAgentSDKTarget(ClaudeAgentOptions())
    assert target.history_mode is ConversationHistoryMode.TARGET
    try:
        await target.respond(_user('first'))
        history = [*_user('first'), Message(role='assistant', content='one'), *_user('second')]
        await target.respond(history)
    finally:
        await target.close()

    (first_prompt, first), (second_prompt, second) = query.calls
    assert (first_prompt, first.resume, first.fork_session) == ('first', None, False)
    assert (second_prompt, second.resume, second.fork_session) == ('second', 'sess-1', True)
    assert first.cwd == second.cwd
    assert first.env['CLAUDE_CONFIG_DIR'] == second.env['CLAUDE_CONFIG_DIR']


@pytest.mark.asyncio
async def test_a_failed_turn_leaves_the_session_where_it_was_so_a_retry_starts_from_the_same_point(fake: Any) -> None:
    query = fake(
        [_result('sess-1', text='one')],
        [_assistant(TextBlock(text='partial')), _result('sess-broken', text=None, error='error_max_turns')],
        [_result('sess-3', text='three')],
    )
    target = ClaudeAgentSDKTarget(ClaudeAgentOptions())
    try:
        await target.respond(_user('first'))
        with pytest.raises(ClaudeAgentSDKError) as info:
            await target.respond(_user('second'))
        assert target.session_id == 'sess-1'
        await target.respond(_user('second'))
    finally:
        await target.close()

    assert (info.value.code, info.value.message) == ('agent_sdk.error_max_turns', 'error_max_turns')
    assert [options.resume for _, options in query.calls] == [None, 'sess-1', 'sess-1']
    assert target.session_id is None  # closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('stream', 'kind', 'code', 'message'),
    [
        (
            [_result('s', text='budget spent', error='error_max_budget_usd'), ClaudeSDKError('run failed')],
            ClaudeAgentSDKError,
            'agent_sdk.error_max_budget_usd',
            'budget spent',
        ),
        ([ClaudeSDKError('pipe closed')], ClaudeAgentSDKError, 'agent_sdk.sdk_error', 'ClaudeSDKError: pipe closed'),
        # What a wrong gateway address looks like: the runtime reports subtype ``success`` with ``is_error`` set.
        (
            [_assistant(TextBlock(text='API Error: Invalid URL')), _result('s', text='API Error: Invalid URL', error='success')],
            ClaudeAgentSDKError,
            'agent_sdk.api_error',
            'API Error: Invalid URL',
        ),
        ([CLINotFoundError('Claude Code not found')], ClaudeAgentSDKUnavailableError, 'agent_sdk.cli_not_found', None),
        ([_assistant(TextBlock(text='   ')), _result('s', text='  ')], ClaudeAgentSDKError, 'agent_sdk.no_result', None),
        # The shapes below are what a live run through the Orq router returned for a wrong key and an unknown model.
        (
            _api_failure('Failed to authenticate. API Error: 401', error='authentication_failed', status=401),
            ClaudeAgentSDKRequestError,
            'agent_sdk.authentication_failed',
            'Failed to authenticate. API Error: 401',
        ),
        (
            _api_failure("API Error: 400 Model 'x' not found", error='unknown', status=400),
            ClaudeAgentSDKRequestError,
            'agent_sdk.invalid_request',
            "HTTP 400: API Error: 400 Model 'x' not found",
        ),
        (
            _api_failure('Credit balance is too low', error='billing_error', status=None),
            ClaudeAgentSDKRequestError,
            'agent_sdk.billing_error',
            None,
        ),
        (
            _api_failure('Bad request', error='invalid_request', status=None),
            ClaudeAgentSDKRequestError,
            'agent_sdk.invalid_request',
            None,
        ),
        (_api_failure('Rate limited', error='rate_limit', status=429), ClaudeAgentSDKError, 'agent_sdk.api_error', None),
        (_api_failure('Overloaded', error='server_error', status=529), ClaudeAgentSDKError, 'agent_sdk.api_error', None),
        (_api_failure('Timed out', error='unknown', status=408), ClaudeAgentSDKError, 'agent_sdk.api_error', None),
        (
            [ClaudeSDKError('Claude Code returned an error result: Sandbox required but unavailable: bwrap missing')],
            ClaudeAgentSDKUnavailableError,
            'agent_sdk.sandbox_unavailable',
            None,
        ),
    ],
)
async def test_each_way_a_turn_can_fail_has_its_own_code(
    fake: Any, stream: list[Any], kind: type[Exception], code: str, message: str | None
) -> None:
    fake(stream)
    target = ClaudeAgentSDKTarget(ClaudeAgentOptions())
    try:
        with pytest.raises(ClaudeAgentSDKError) as info:
            await target.respond(_user())
    finally:
        await target.close()

    assert type(info.value) is kind
    assert info.value.code == code
    if message is not None:
        assert info.value.message == message
    assert target.map_error(info.value) == (info.value.code, info.value.message)
    assert isinstance(info.value, NonRetryableTargetError) is (code in NON_RETRYABLE_CODES)


NON_RETRYABLE_CODES = {
    'agent_sdk.cli_not_found',
    'agent_sdk.sandbox_unavailable',
    'agent_sdk.authentication_failed',
    'agent_sdk.billing_error',
    'agent_sdk.invalid_request',
}


@pytest.mark.asyncio
async def test_a_rejected_key_is_tried_once_through_the_shared_retry_layer(fake: Any) -> None:
    query = fake(
        _api_failure('Failed to authenticate. API Error: 401', error='authentication_failed', status=401),
        [_result('never', text='unreached')],
    )
    target = ClaudeAgentSDKTarget(ClaudeAgentOptions())
    try:
        result = await call_target_with_retry(
            target, _user(), target_agent_timeout_ms=5_000, max_target_retries=3, map_error=target.map_error
        )
    finally:
        await target.close()

    assert len(query.calls) == 1
    assert result.error_payload()['error_code'] == 'agent_sdk.authentication_failed'


@pytest.mark.asyncio
async def test_a_rate_limit_is_retried(fake: Any) -> None:
    query = fake(_api_failure('Rate limited', error='rate_limit', status=429), [_result('s2', text='second try')])
    target = ClaudeAgentSDKTarget(ClaudeAgentOptions())
    try:
        result = await call_target_with_retry(
            target, _user(), target_agent_timeout_ms=5_000, max_target_retries=3, map_error=target.map_error
        )
    finally:
        await target.close()

    assert len(query.calls) == 2
    assert result.error is None
    assert result.response.text == 'second try'


@pytest.mark.asyncio
async def test_a_sandbox_message_is_an_ordinary_sdk_error_without_sandbox_isolation(fake: Any) -> None:
    fake([ClaudeSDKError('Sandbox required but unavailable')])
    target = ClaudeAgentSDKTarget(ClaudeAgentOptions(), isolation='none')
    try:
        with pytest.raises(ClaudeAgentSDKError) as info:
            await target.respond(_user())
    finally:
        await target.close()
    assert info.value.code == 'agent_sdk.sdk_error'


def test_map_error_has_no_opinion_on_other_exceptions() -> None:
    assert ClaudeAgentSDKTarget(ClaudeAgentOptions()).map_error(RuntimeError('boom')) is None


@pytest.mark.asyncio
async def test_a_missing_runtime_is_reported_once_through_the_shared_retry_layer(fake: Any) -> None:
    query = fake([CLINotFoundError('Claude Code not found')], [_result('never', text='unreached')])
    target = ClaudeAgentSDKTarget(ClaudeAgentOptions())
    try:
        result = await call_target_with_retry(
            target, _user(), target_agent_timeout_ms=5_000, max_target_retries=3, map_error=target.map_error
        )
    finally:
        await target.close()

    assert len(query.calls) == 1
    assert result.error is not None
    assert result.error_payload()['error_code'] == 'agent_sdk.cli_not_found'


@pytest.mark.asyncio
@pytest.mark.parametrize('messages', [[], [Message(role='assistant', content='hi')]])
async def test_respond_needs_a_user_message_last(messages: list[Message]) -> None:
    with pytest.raises(ValueError, match=r"requires messages\[-1\]\.role == 'user'"):
        await ClaudeAgentSDKTarget(ClaudeAgentOptions()).respond(messages)


@pytest.mark.asyncio
async def test_sandbox_isolation_moves_the_agent_off_the_host_configuration() -> None:
    options = ClaudeAgentOptions(
        cwd='/somewhere/real', env={'ANTHROPIC_BASE_URL': 'https://gateway.example', 'CLAUDE_CONFIG_DIR': '/home/me'}
    )
    options.sandbox = {'enabled': False, 'allowUnsandboxedCommands': True, 'excludedCommands': ['git']}
    target = ClaudeAgentSDKTarget(options)
    try:
        turn = target.turn_options()
        root = _workdir(target).parent
        assert turn.cwd == root / 'workdir'
        assert turn.cwd.is_dir()
        assert turn.env == {
            'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1',
            'CLAUDE_CODE_MAX_RETRIES': '0',
            'ENABLE_CLAUDEAI_MCP_SERVERS': 'false',
            'ANTHROPIC_BASE_URL': 'https://gateway.example',
            'CLAUDE_CONFIG_DIR': str(root / 'config'),
        }
        assert turn.setting_sources == []
        assert turn.sandbox == {
            'enabled': True,
            'allowUnsandboxedCommands': False,
            'failIfUnavailable': True,
            'excludedCommands': ['git'],
        }
        assert json.loads(turn.settings or '') == {
            'permissions': {'blockWritesOutsideWorkingDirectories': True, 'blockReadsOutsideWorkingDirectories': True}
        }
        # The caller's options are not changed.
        assert (options.cwd, options.setting_sources) == ('/somewhere/real', None)
        assert options.sandbox['enabled'] is False
    finally:
        await target.close()


@pytest.mark.asyncio
async def test_sandbox_isolation_keeps_setting_sources_and_env_the_caller_chose() -> None:
    options = ClaudeAgentOptions(setting_sources=['project'], env={'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '0'})
    target = ClaudeAgentSDKTarget(options)
    try:
        turn = target.turn_options()
        assert turn.setting_sources == ['project']
        assert turn.env['CLAUDE_CODE_DISABLE_AUTO_MEMORY'] == '0'
    finally:
        await target.close()


@pytest.mark.asyncio
async def test_no_isolation_only_moves_the_two_directories() -> None:
    options = ClaudeAgentOptions(env={'A': 'b'})
    target = ClaudeAgentSDKTarget(options, isolation='none')
    try:
        turn = target.turn_options()
        root = _workdir(target).parent
        assert turn.env == {'CLAUDE_CODE_MAX_RETRIES': '0', 'A': 'b', 'CLAUDE_CONFIG_DIR': str(root / 'config')}
        assert turn.cwd == root / 'workdir'
        assert turn.setting_sources is None
        assert turn.sandbox is None
        assert turn.settings is None
    finally:
        await target.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('isolation', ['sandbox', 'none'])
async def test_the_runtimes_own_retry_is_off_unless_the_caller_sets_it(isolation: Isolation) -> None:
    default = ClaudeAgentSDKTarget(ClaudeAgentOptions(), isolation=isolation)
    chosen = ClaudeAgentSDKTarget(ClaudeAgentOptions(env={'CLAUDE_CODE_MAX_RETRIES': '3'}), isolation=isolation)
    try:
        assert default.turn_options().env['CLAUDE_CODE_MAX_RETRIES'] == '0'
        assert chosen.turn_options().env['CLAUDE_CODE_MAX_RETRIES'] == '3'
    finally:
        await default.close()
        await chosen.close()


BLOCKS = {'blockWritesOutsideWorkingDirectories': True, 'blockReadsOutsideWorkingDirectories': True}


def test_file_tool_blocks_are_forced_on_and_the_callers_other_settings_are_kept(tmp_path: Path) -> None:
    assert json.loads(settings_with_file_tool_blocks(None)) == {'permissions': BLOCKS}

    own = {'model': 'x', 'permissions': {'allow': ['Read'], 'blockWritesOutsideWorkingDirectories': False}}
    expected = {'model': 'x', 'permissions': {'allow': ['Read'], **BLOCKS}}
    assert json.loads(settings_with_file_tool_blocks(json.dumps(own))) == expected

    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(own), encoding='utf-8')
    assert json.loads(settings_with_file_tool_blocks(str(path))) == expected


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        ('/no/such/settings.json', 'is not JSON and could not be read as a file'),
        ('{not json', 'is not valid JSON'),
        ('{"permissions": ["Read"]}', 'must be a JSON object whose "permissions" is an object'),
    ],
)
def test_settings_that_cannot_carry_the_blocks_are_refused_before_any_turn(settings: str, message: str) -> None:
    target = ClaudeAgentSDKTarget(ClaudeAgentOptions(settings=settings))
    with pytest.raises(ValueError, match=message):
        target.turn_options()
    # Without isolation the settings are passed through untouched, valid or not.
    assert ClaudeAgentSDKTarget(ClaudeAgentOptions(settings=settings), isolation='none').turn_options().settings == settings


def test_isolation_and_workdir_are_checked_at_construction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='Unknown isolation'):
        ClaudeAgentSDKTarget(ClaudeAgentOptions(), isolation=cast(Any, 'docker'))
    with pytest.raises(ValueError, match='is not a directory'):
        ClaudeAgentSDKTarget(ClaudeAgentOptions(), workdir=tmp_path / 'missing')


@pytest.mark.asyncio
async def test_each_clone_gets_its_own_copy_of_the_workdir_and_close_removes_it(tmp_path: Path) -> None:
    source = tmp_path / 'project'
    source.mkdir()
    (source / 'README.md').write_text('hello', encoding='utf-8')

    class Subclass(ClaudeAgentSDKTarget):
        pass

    first = Subclass(ClaudeAgentOptions(), workdir=source)
    second = first.new()
    assert type(second) is Subclass
    assert first.workdir is None  # nothing is created before the first turn

    first.turn_options()
    second.turn_options()
    assert first.workdir != second.workdir
    (_workdir(first) / 'README.md').write_text('changed by the agent', encoding='utf-8')
    assert (_workdir(second) / 'README.md').read_text(encoding='utf-8') == 'hello'
    assert (source / 'README.md').read_text(encoding='utf-8') == 'hello'

    root = _workdir(first).parent
    await first.close()
    await second.close()
    assert not root.exists()
    assert first.workdir is None


@pytest.mark.asyncio
async def test_a_failed_copy_of_the_workdir_is_removed_and_the_next_turn_copies_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retried turn on the same target must not reuse a partial copy: the runner retries the same instance."""
    source = tmp_path / 'project'
    source.mkdir()
    (source / 'app.py').write_text('print(1)', encoding='utf-8')
    roots: list[Path] = []
    real_copytree = target_module.shutil.copytree

    def failing_copytree(src: Any, dst: Any, **kwargs: Any) -> None:
        roots.append(Path(dst).parent)
        Path(dst).mkdir()  # a partial copy
        raise OSError('disk full')

    monkeypatch.setattr(target_module.shutil, 'copytree', failing_copytree)
    target = ClaudeAgentSDKTarget(ClaudeAgentOptions(), workdir=source)
    with pytest.raises(OSError, match='disk full'):
        target.turn_options()
    assert not roots[0].exists()
    assert target.workdir is None

    monkeypatch.setattr(target_module.shutil, 'copytree', real_copytree)
    options = target.turn_options()
    assert options.cwd is not None
    cwd = Path(options.cwd)
    assert cwd != roots[0] / 'workdir'
    assert (cwd / 'app.py').read_text(encoding='utf-8') == 'print(1)'
    assert Path(options.env['CLAUDE_CONFIG_DIR']).is_dir()
    await target.close()


def test_a_clone_keeps_the_isolation_and_keep_workdir_of_its_original() -> None:
    first = ClaudeAgentSDKTarget(ClaudeAgentOptions(), isolation='none', keep_workdir=True)
    clone = first.new()
    assert clone._isolation == 'none'
    assert clone._keep_workdir is True


@pytest.mark.asyncio
async def test_symbolic_links_in_the_workdir_are_not_copied(tmp_path: Path) -> None:
    """A link would still point at the host; following it would copy host files nobody put in the project."""
    secret = tmp_path / 'host-secret.txt'
    secret.write_text('host only', encoding='utf-8')
    outside_dir = tmp_path / 'host-dir'
    outside_dir.mkdir()
    (outside_dir / 'inner.txt').write_text('host only', encoding='utf-8')
    source = tmp_path / 'project'
    (source / 'pkg').mkdir(parents=True)
    (source / 'README.md').write_text('hello', encoding='utf-8')
    (source / 'pkg' / 'module.py').write_text('x = 1', encoding='utf-8')
    (source / 'link-to-file').symlink_to(secret)
    (source / 'pkg' / 'link-to-dir').symlink_to(outside_dir, target_is_directory=True)

    target = ClaudeAgentSDKTarget(ClaudeAgentOptions(), workdir=source)
    try:
        target.turn_options()
        copied = sorted(str(path.relative_to(_workdir(target))) for path in _workdir(target).rglob('*'))
    finally:
        await target.close()

    assert copied == ['README.md', 'pkg', str(Path('pkg') / 'module.py')]
    # The originals, links included, are untouched.
    assert (source / 'link-to-file').is_symlink()
    assert secret.read_text(encoding='utf-8') == 'host only'


@pytest.mark.asyncio
async def test_keep_workdir_leaves_the_directories_after_close() -> None:
    import shutil

    target = ClaudeAgentSDKTarget(ClaudeAgentOptions(), keep_workdir=True)
    target.turn_options()
    root = _workdir(target).parent
    await target.close()
    try:
        assert (root / 'workdir').is_dir()
    finally:
        shutil.rmtree(root, ignore_errors=True)


class FakeClient:
    """Stands in for ``ClaudeSDKClient``: announces ``init`` (or fails to start) and records the options it got."""

    instances: ClassVar[list[FakeClient]] = []

    def __init__(self, *, options: ClaudeAgentOptions) -> None:
        self.options = options
        self.disconnected = False
        self.prompts: list[str] = []
        FakeClient.instances.append(self)

    init: ClassVar[dict[str, Any] | BaseException] = {}

    async def connect(self) -> None:
        if isinstance(FakeClient.init, BaseException):
            raise FakeClient.init

    async def query(self, prompt: str) -> None:
        self.prompts.append(prompt)

    async def receive_messages(self) -> Any:
        yield SystemMessage(subtype='hook_started', data={})
        assert isinstance(FakeClient.init, dict)
        yield SystemMessage(subtype='init', data=FakeClient.init)
        raise AssertionError('the probe must stop reading at init')

    async def disconnect(self) -> None:
        self.disconnected = True


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> type[FakeClient]:
    FakeClient.instances = []
    FakeClient.init = {}
    monkeypatch.setattr(target_module, 'ClaudeSDKClient', FakeClient)
    return FakeClient


RUNTIME_TOOLS = ['Task', 'Bash', 'Edit', 'Read', 'WebSearch', 'Write', 'mcp__shop__lookup_order', 'mcp__ext__ext_tool']


@pytest.mark.asyncio
async def test_agent_context_lists_the_tools_the_runtime_announces_without_a_reachable_model(
    fake_client: type[FakeClient], tmp_path: Path
) -> None:
    """No ``tools`` and an MCP server: the options name nothing, but the runtime gives the agent built-ins and MCP tools."""
    fake_client.init = {
        'tools': RUNTIME_TOOLS,
        'mcp_servers': [{'name': 'shop', 'status': 'connected'}, {'name': 'ext', 'status': 'connected'}],
    }

    async def hook(*_: Any) -> dict[str, Any]:
        return {}

    options = ClaudeAgentOptions(
        system_prompt='You are a support agent.',
        model='claude-sonnet-5-5',
        allowed_tools=['Read'],
        env={
            'ANTHROPIC_BASE_URL': 'https://gateway.example',
            'ANTHROPIC_AUTH_TOKEN': 'kept',
            'CLAUDE_CODE_MAX_RETRIES': '5',
        },
        hooks=cast(Any, {'PreToolUse': [hook]}),
    )
    target = ClaudeAgentSDKTarget(options)
    context = await target.get_agent_context()

    assert (context.key, context.system_prompt, context.model) == (
        'claude_agent_sdk',
        'You are a support agent.',
        'claude-sonnet-5-5',
    )
    assert context.get_tool_names() == RUNTIME_TOOLS
    (client,) = fake_client.instances
    assert client.disconnected
    probe = client.options
    # No model can answer the probe, whichever provider the options pick; everything else is a real turn's options.
    assert {key: probe.env[key] for key in target_module.PROBE_ENDPOINT_ENV} == dict.fromkeys(
        target_module.PROBE_ENDPOINT_ENV, 'http://127.0.0.1:9'
    )
    assert probe.env['ANTHROPIC_AUTH_TOKEN'] == 'kept'
    # Retrying the dead address would only delay the probe, so the caller's retry count is overridden there.
    assert probe.env['CLAUDE_CODE_MAX_RETRIES'] == '0'
    assert (probe.hooks, probe.can_use_tool, probe.resume) == (None, None, None)
    assert cast(dict[str, Any], probe.sandbox)['failIfUnavailable'] is True
    assert probe.setting_sources == []
    # The probe's directories are its own and are gone; the target has not started its own conversation.
    assert probe.cwd is not None and not Path(probe.cwd).exists()
    assert target.workdir is None
    assert options.env['ANTHROPIC_BASE_URL'] == 'https://gateway.example'


@pytest.mark.asyncio
async def test_agent_context_warns_about_an_mcp_server_whose_tools_are_unknown(
    fake_client: type[FakeClient], capsys: pytest.CaptureFixture[str]
) -> None:
    fake_client.init = {'tools': ['Read'], 'mcp_servers': [{'name': 'remote', 'status': 'failed'}]}
    context = await ClaudeAgentSDKTarget(ClaudeAgentOptions()).get_agent_context()

    assert context.get_tool_names() == ['Read']
    assert "MCP server 'remote' is failed; its tools are not in the agent context" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_agent_context_falls_back_to_the_named_tools_with_a_warning_when_the_runtime_cannot_start(
    fake_client: type[FakeClient], capsys: pytest.CaptureFixture[str]
) -> None:
    fake_client.init = CLINotFoundError('Claude Code not found')
    options = ClaudeAgentOptions(tools=['Read', 'Bash', 'Grep'], disallowed_tools=['Bash'])
    context = await ClaudeAgentSDKTarget(options).get_agent_context()

    assert context.get_tool_names() == ['Read', 'Grep']
    assert 'could not read the tool list from the Claude Code runtime (CLINotFoundError' in capsys.readouterr().err


@pytest.mark.asyncio
async def test_agent_context_falls_back_to_allowed_tools_when_the_options_name_no_tool_list(
    fake_client: type[FakeClient], capsys: pytest.CaptureFixture[str]
) -> None:
    fake_client.init = CLINotFoundError('Claude Code not found')
    options = ClaudeAgentOptions(
        tools={'type': 'preset', 'preset': 'claude_code'}, allowed_tools=['Read', 'Bash'], disallowed_tools=['Bash']
    )
    context = await ClaudeAgentSDKTarget(options).get_agent_context()

    assert context.get_tool_names() == ['Read']
    assert 'could not read the tool list' in capsys.readouterr().err


@pytest.mark.parametrize(
    ('prompt', 'text'),
    [
        (None, ''),
        ('Plain.', 'Plain.'),
        ({'type': 'custom', 'prompt': 'Custom.'}, 'Custom.'),
        ({'type': 'preset', 'preset': 'claude_code', 'append': 'Never reveal orders.'}, 'Never reveal orders.'),
        ({'type': 'preset', 'preset': 'claude_code'}, ''),
    ],
)
def test_the_system_prompt_text_evaluatorq_can_know(prompt: Any, text: str) -> None:
    assert target_module.system_prompt_text(prompt) == text


def test_a_system_prompt_file_is_read(tmp_path: Path) -> None:
    path = tmp_path / 'prompt.md'
    path.write_text('From a file.', encoding='utf-8')
    assert target_module.system_prompt_text({'type': 'file', 'path': str(path)}) == 'From a file.'


def test_sandbox_mode_warns_once_about_project_files_it_will_not_load(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / '.claude').mkdir()
    (tmp_path / '.claude' / 'settings.json').write_text('{}', encoding='utf-8')
    (tmp_path / 'CLAUDE.md').write_text('Project rules.', encoding='utf-8')

    target = ClaudeAgentSDKTarget(ClaudeAgentOptions(), workdir=tmp_path)
    target.new()
    ClaudeAgentSDKTarget(ClaudeAgentOptions(setting_sources=['project']), workdir=tmp_path)
    ClaudeAgentSDKTarget(ClaudeAgentOptions(), workdir=tmp_path, isolation='none')

    err = capsys.readouterr().err
    assert err.count('does not load CLAUDE.md, .claude/settings.json from') == 1
    assert target.turn_options().setting_sources == []
    shutil.rmtree(_workdir(target).parent)


@pytest.mark.asyncio
async def test_cached_tokens_and_the_runtime_cost_reach_the_response_and_the_span(
    fake: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr('evaluatorq.common.tracing.get_tracer', lambda: provider.get_tracer('agent-sdk-tests'))
    # The runtime's usage is the Anthropic shape: input_tokens excludes the cache reads and writes.
    usage = {
        'input_tokens': 10,
        'output_tokens': 7,
        'cache_read_input_tokens': 1200,
        'cache_creation_input_tokens': 300,
    }
    fake([ResultMessage(
        subtype='success',
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=1,
        session_id='sess-c',
        total_cost_usd=0.0123,
        usage=usage,
        result='cached',
    )])
    target = ClaudeAgentSDKTarget(ClaudeAgentOptions(model='claude-haiku-5-5'))
    try:
        response = await target.respond(_user())
    finally:
        await target.close()
        provider.shutdown()

    assert response.usage is not None
    assert (response.usage.input_tokens, response.usage.cached_tokens, response.usage.cache_creation_tokens) == (
        1510,
        1200,
        300,
    )
    assert response.usage.output_tokens == 7
    assert response.usage.total_cost is None  # evaluatorq prices the tokens itself; the runtime's cost is on the span
    (span,) = exporter.get_finished_spans()
    attributes = span.attributes or {}
    assert attributes['evaluatorq.claude_agent_sdk.cost_usd'] == 0.0123
    assert attributes['gen_ai.response.id'] == 'sess-c'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'stream',
    [
        # The run raises after yielding the error result.
        _api_failure('Failed to authenticate. API Error: 401', error='authentication_failed', status=401),
        # The run ends on the error result without raising.
        [_assistant(TextBlock(text='Overloaded'), error='server_error'), _result('s', text='Overloaded', error='success')],
    ],
)
async def test_a_failed_turn_still_puts_its_usage_and_cost_on_the_span(
    fake: Any, monkeypatch: pytest.MonkeyPatch, stream: list[Any]
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr('evaluatorq.common.tracing.get_tracer', lambda: provider.get_tracer('agent-sdk-tests'))
    fake(stream)
    target = ClaudeAgentSDKTarget(ClaudeAgentOptions(model='claude-haiku-5-5'))
    try:
        with pytest.raises(ClaudeAgentSDKError):
            await target.respond(_user())
    finally:
        await target.close()
        provider.shutdown()

    (span,) = exporter.get_finished_spans()
    attributes = span.attributes or {}
    assert attributes['evaluatorq.claude_agent_sdk.cost_usd'] == 0.01
    assert (attributes['gen_ai.usage.input_tokens'], attributes['gen_ai.usage.output_tokens']) == (11, 7)


def test_messages_with_nothing_for_the_transcript_are_skipped() -> None:
    assert message_to_event(SystemMessage(subtype='init', data={})) is None
    assert message_to_event(UserMessage(content='plain user text')) is None
    assert message_to_event(UserMessage(content=[TextBlock(text='no tool result here')])) is None
    assert message_to_event(object()) is None
