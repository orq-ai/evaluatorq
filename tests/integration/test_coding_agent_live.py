"""Live coding-agent runs. Skipped unless the binary is on PATH. Not run in CI."""

from __future__ import annotations

import shutil

import pytest

from evaluatorq.backends.coding_agent import CodingAgentTarget, OrqLaunchOptions
from evaluatorq.contracts import Message

pytestmark = pytest.mark.integration


def _needs(binary: str) -> None:
    if shutil.which(binary) is None:
        pytest.skip(f'{binary} not installed')


@pytest.mark.asyncio
async def test_claude_direct_stdin_prompt_text_turn() -> None:
    _needs('claude')
    target = CodingAgentTarget('claude', permission_mode='plan', timeout_ms=120_000)
    response = await target.respond([Message(role='user', content='Reply with the single word: pong')])
    assert 'pong' in response.text.lower()
    await target.close()


@pytest.mark.asyncio
async def test_codex_orq_sandbox_after_full_auto() -> None:
    """Spec question 1: does a later --sandbox win over orq's injected --full-auto?"""
    _needs('orq')
    _needs('codex')
    target = CodingAgentTarget('codex', launcher='orq', permission_mode='read-only', timeout_ms=180_000)
    response = await target.respond([
        Message(role='user', content='Create a file named probe.txt containing "x". Then say done or blocked.')
    ])
    assert target.workdir is not None
    assert not (target.workdir / 'probe.txt').exists(), 'read-only sandbox was overridden by --full-auto'
    await target.close()


@pytest.mark.asyncio
async def test_orq_skill_cleanup_leaves_our_links() -> None:
    """Spec question 3: orq launch removes only the skills it linked, not ours."""
    _needs('orq')
    _needs('claude')
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        skill = Path(d) / 'probe-skill'
        skill.mkdir()
        (skill / 'SKILL.md').write_text('# probe\n')
        target = CodingAgentTarget('claude', launcher='orq', skills=[skill], permission_mode='plan', timeout_ms=120_000)
        await target.respond([Message(role='user', content='Reply with: ok')])
        assert target.workdir is not None
        assert (target.workdir / '.claude' / 'skills' / 'probe-skill').is_symlink()
        await target.close()


@pytest.mark.asyncio
async def test_orq_mcp_tool_call_is_parsed() -> None:
    """Spec question 4: an MCP tool call under launcher='orq' shows up as a tool call."""
    _needs('orq')
    _needs('claude')
    target = CodingAgentTarget(
        'claude', launcher='orq', orq=OrqLaunchOptions(mcp=True), permission_mode='acceptEdits', timeout_ms=180_000
    )
    response = await target.respond([
        Message(role='user', content='Use the orq MCP server to list prompts in this workspace, then say done.')
    ])
    assert any('mcp' in c.name.lower() or 'orq' in c.name.lower() for c in response.tool_calls), response.tool_calls
    await target.close()


@pytest.mark.asyncio
async def test_opencode_reads_stdin_prompt() -> None:
    """Spec question 2: opencode event names are covered by fixtures; this confirms stdin delivery."""
    _needs('opencode')
    target = CodingAgentTarget('opencode', extra_args=['--auto'], timeout_ms=120_000)
    response = await target.respond([Message(role='user', content='Reply with the single word: pong')])
    assert 'pong' in response.text.lower()
    await target.close()
