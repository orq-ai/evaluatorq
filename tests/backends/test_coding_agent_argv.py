"""The six frozen argv vectors (three agents times two launchers) and flag placement."""

from __future__ import annotations

from typing import Any

import pytest

from evaluatorq.backends.coding_agent import OrqLaunchOptions, build_argv

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
