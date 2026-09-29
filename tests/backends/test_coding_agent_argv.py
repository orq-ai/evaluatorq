"""The six frozen argv vectors (three agents times two launchers) and flag placement."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pydantic
import pytest

from evaluatorq.backends import coding_agent as coding_agent_module
from evaluatorq.backends.coding_agent import (
    AGENTS,
    CodingAgentTarget,
    CodingAgentUnavailableError,
    DockerOptions,
    OrqLaunchOptions,
    ParsedTurn,
    build_argv,
    refuse_unsafe_batch_args,
)
from evaluatorq.contracts import Message

PROMPT = 'reply to the user'


def _argv(**kw: Any) -> tuple[list[str], str | None]:
    defaults: dict[str, Any] = dict(
        launcher='direct',
        model=None,
        permission_mode=None,
        system_prompt=None,
        extra_args=None,
        orq=None,
        prompt=PROMPT,
    )
    defaults.update(kw)
    return build_argv(**defaults)


def test_claude_direct() -> None:
    argv, stdin = _argv(agent='claude')
    assert argv == ['claude', '-p', '--output-format', 'stream-json', '--verbose']
    assert stdin == PROMPT


def test_codex_direct() -> None:
    argv, stdin = _argv(agent='codex')
    assert argv == ['codex', 'exec', '--json', '--skip-git-repo-check', '-']
    assert stdin == PROMPT


def test_opencode_direct() -> None:
    argv, stdin = _argv(agent='opencode')
    assert argv == ['opencode', 'run', '--format', 'json']
    assert stdin == PROMPT


def test_claude_orq() -> None:
    argv, stdin = _argv(agent='claude', launcher='orq')
    assert argv == ['orq', 'launch', 'claude', '--', '-p', '--output-format', 'stream-json', '--verbose']
    assert stdin == PROMPT


def test_codex_orq_prompt_via_orq_and_stdin_closed() -> None:
    argv, stdin = _argv(agent='codex', launcher='orq')
    assert argv == ['orq', 'launch', 'codex', '-p', PROMPT, '--', '--json', '--skip-git-repo-check']
    assert stdin is None


def test_opencode_orq() -> None:
    argv, stdin = _argv(agent='opencode', launcher='orq')
    assert argv == ['orq', 'launch', 'opencode', '-p', PROMPT, '--', '--format', 'json']
    assert stdin is None


def test_model_flag_per_agent_direct() -> None:
    assert '--model' in _argv(agent='claude', model='claude-fable-5-1')[0]
    argv, _ = _argv(agent='codex', model='gpt-5.6-luna')
    assert argv[argv.index('-m') + 1] == 'gpt-5.6-luna'
    argv, _ = _argv(agent='opencode', model='openai/gpt-5.6-luna')
    assert argv[argv.index('--model') + 1] == 'openai/gpt-5.6-luna'


def test_model_under_orq_goes_to_orq_launch() -> None:
    argv, _ = _argv(agent='claude', launcher='orq', model='anthropic/claude-fable-5-1')
    dashdash = argv.index('--')
    assert argv[argv.index('--model') + 1] == 'anthropic/claude-fable-5-1'
    assert argv.index('--model') < dashdash


def test_orq_options_render_as_flags() -> None:
    argv, _ = _argv(
        agent='claude',
        launcher='orq',
        orq=OrqLaunchOptions(mcp=False, skills=False, base_url='https://my.orq.ai', fetch_models=False),
    )
    before = argv[: argv.index('--')]
    assert before == [
        'orq', 'launch', 'claude', '--no-mcp', '--no-skills', '--base-url', 'https://my.orq.ai', '--no-fetch-models'
    ]


def test_permission_mode_flags() -> None:
    argv, _ = _argv(agent='claude', permission_mode='plan')
    assert argv[argv.index('--permission-mode') + 1] == 'plan'
    argv, _ = _argv(agent='codex', permission_mode='read-only')
    assert argv[argv.index('--sandbox') + 1] == 'read-only'


def test_permission_mode_on_opencode_raises() -> None:
    with pytest.raises(ValueError, match='opencode'):
        _argv(agent='opencode', permission_mode='anything')


def test_system_prompt_claude_flag_and_extra_args_last() -> None:
    argv, _ = _argv(agent='claude', system_prompt='be terse', extra_args=['--disallowedTools', 'Bash'])
    assert argv[argv.index('--append-system-prompt') + 1] == 'be terse'
    assert argv[-2:] == ['--disallowedTools', 'Bash']


def test_system_prompt_not_a_flag_for_codex_or_opencode() -> None:
    for agent in ('codex', 'opencode'):
        argv, _ = _argv(agent=agent, system_prompt='be terse')
        assert 'be terse' not in argv


def test_orq_launch_options_is_frozen() -> None:
    opts = OrqLaunchOptions()
    with pytest.raises(pydantic.ValidationError):
        opts.mcp = False  # pyright: ignore[reportAttributeAccessIssue]


def test_orq_launch_options_keeps_positional_constructor_and_value_semantics() -> None:
    positional = OrqLaunchOptions(False, False, 'https://my.orq.ai', False)
    keyword = OrqLaunchOptions(mcp=False, skills=False, base_url='https://my.orq.ai', fetch_models=False)
    assert positional == keyword
    assert hash(positional) == hash(keyword)
    assert positional.to_flags() == ['--no-mcp', '--no-skills', '--base-url', 'https://my.orq.ai', '--no-fetch-models']


def test_container_clone_reapplies_implicit_agent_defaults() -> None:
    for original in (
        CodingAgentTarget(agent='claude', container=DockerOptions()),
        CodingAgentTarget(agent='opencode', container=DockerOptions()),
    ):
        clone = original.new()
        assert clone._permission_mode == original._permission_mode
        assert clone._extra_args == original._extra_args
        assert clone._container == original._container


def test_agents_registry_is_read_only() -> None:
    assert isinstance(AGENTS, types.MappingProxyType)
    with pytest.raises(TypeError):
        AGENTS['x'] = AGENTS['claude']  # pyright: ignore[reportIndexIssue]


def test_agent_spec_is_frozen() -> None:
    with pytest.raises(pydantic.ValidationError):
        AGENTS['claude'].binary = 'x'  # pyright: ignore[reportAttributeAccessIssue]


def test_parsed_turn_stays_mutable() -> None:
    turn = ParsedTurn()
    turn.text = 'hi'
    assert turn.text == 'hi' and turn.tool_calls == []


@pytest.mark.parametrize('arg', ['a & calc', 'say "hi"', '%PATH%', 'line\nbreak'])
def test_batch_launcher_refuses_cmd_metacharacters(arg: str) -> None:
    with pytest.raises(CodingAgentUnavailableError) as info:
        refuse_unsafe_batch_args(['C:\\npm\\claude.CMD', '--model', arg])
    assert info.value.code == 'cli.unsafe_shim'
    assert repr(arg) in info.value.message and 'native .exe' in info.value.message
    refuse_unsafe_batch_args(['C:\\npm\\claude.cmd', '-p', '--verbose'])
    refuse_unsafe_batch_args(['/usr/bin/claude', '--model', arg])


@pytest.mark.asyncio
async def test_batch_launcher_moves_system_prompt_to_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shim = tmp_path / 'claude.cmd'
    monkeypatch.setattr(coding_agent_module.shutil, 'which', lambda name, path=None: str(shim))
    seen: dict[str, Any] = {}

    async def fake_run(self: CodingAgentTarget, argv: list[str], stdin_text: str | None, **_: Any) -> tuple[int, str, str]:
        seen.update(argv=argv, stdin=stdin_text)
        return 0, '{"type":"result","result":"done","session_id":"s"}\n', ''

    monkeypatch.setattr(CodingAgentTarget, '_run', fake_run)
    system_prompt = 'Say "hi" & stop\nnow'
    target = CodingAgentTarget(agent='claude', system_prompt=system_prompt)
    response = await target.respond([Message(role='user', content='x')])
    assert response.text == 'done'
    assert seen['argv'][0] == str(shim)
    assert '--append-system-prompt' not in seen['argv']
    assert '{"role": "system", "content": "Say \\"hi\\"' in seen['stdin']
    await target.close()


def test_orq_batch_launcher_points_at_the_native_release() -> None:
    with pytest.raises(CodingAgentUnavailableError) as info:
        refuse_unsafe_batch_args(['C:/npm/orq.cmd', 'claude', '--model', 'a & b'])
    assert 'orq-win32-x64.exe' in info.value.message and 'native .exe' not in info.value.message
