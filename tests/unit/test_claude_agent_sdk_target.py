"""ClaudeAgentSDKTarget on fake SDK message streams: evidence, session handling, isolation and errors. No network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

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


def _assistant(*blocks: Any) -> AssistantMessage:
    return AssistantMessage(content=list(blocks), model='claude-sonnet-5-5')


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
    assert isinstance(info.value, NonRetryableTargetError) is (code == 'agent_sdk.cli_not_found')


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
        assert turn.sandbox == {'enabled': True, 'allowUnsandboxedCommands': False, 'excludedCommands': ['git']}
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


@pytest.mark.asyncio
async def test_agent_context_comes_from_the_options() -> None:
    options = ClaudeAgentOptions(system_prompt='You are a support agent.', model='claude-sonnet-5-5', tools=['Read', 'Bash'])
    context = await ClaudeAgentSDKTarget(options).get_agent_context()
    assert (context.key, context.system_prompt, context.model) == ('claude_agent_sdk', 'You are a support agent.', 'claude-sonnet-5-5')
    assert context.get_tool_names() == ['Read', 'Bash']

    listed = await ClaudeAgentSDKTarget(ClaudeAgentOptions(allowed_tools=['Grep'])).get_agent_context()
    assert (listed.system_prompt, listed.get_tool_names()) == ('', ['Grep'])


def test_messages_with_nothing_for_the_transcript_are_skipped() -> None:
    assert message_to_event(SystemMessage(subtype='init', data={})) is None
    assert message_to_event(UserMessage(content='plain user text')) is None
    assert message_to_event(UserMessage(content=[TextBlock(text='no tool result here')])) is None
    assert message_to_event(object()) is None
