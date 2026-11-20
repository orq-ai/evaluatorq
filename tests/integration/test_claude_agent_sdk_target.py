"""Live check of ClaudeAgentSDKTarget: real Claude Code runtime, model calls through the Orq router.

Needs ``ORQ_API_KEY``. Runs two short turns on a small model, one attempt to leave the working directory, and one
turn with a wrong key.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

pytest.importorskip('claude_agent_sdk')

from claude_agent_sdk import ClaudeAgentOptions

from evaluatorq.common.target_call import call_target_with_retry
from evaluatorq.contracts import Message
from evaluatorq.integrations.claude_agent_sdk_integration import ClaudeAgentSDKTarget

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get('ORQ_API_KEY'), reason='needs ORQ_API_KEY'),
]


def _options(*, token: str | None = None) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt='You are a file assistant. Always call the tool the user names. Answer in one sentence.',
        model='claude-haiku-5-5',
        tools=['Read', 'Write'],
        allowed_tools=['Read', 'Write'],
        permission_mode='acceptEdits',
        max_turns=4,
        env={
            'ANTHROPIC_BASE_URL': 'https://my.orq.ai/v3/anthropic',
            'ANTHROPIC_AUTH_TOKEN': token if token is not None else os.environ['ORQ_API_KEY'],
            'ANTHROPIC_API_KEY': '',
            'ANTHROPIC_DEFAULT_HAIKU_MODEL': 'claude-haiku-5-5',
        },
    )


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_a_real_agent_keeps_its_session_and_cannot_write_outside_its_working_directory(tmp_path: Path) -> None:
    """Assertions hold whatever the model chooses to do: none depends on it calling, or not calling, a tool."""
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'notes.txt').write_text('The file word is HERON-7.\n', encoding='utf-8')
    outside = tmp_path / 'outside.txt'

    target = ClaudeAgentSDKTarget(_options(), workdir=project)
    try:
        first = await target.respond([
            Message(
                role='user',
                content='Remember this word, it is in no file: PELICAN-42. Then call the Read tool on notes.txt and tell me the file word.',
            )
        ])
        first_session = target.session_id
        second = await target.respond([Message(role='user', content='Which word did I ask you to remember?')])
        second_session = target.session_id
        third = await target.respond([
            Message(role='user', content=f'Call the Write tool with file_path {outside} and content: escaped')
        ])
        workdir = target.workdir
    finally:
        await target.close()

    # Tool evidence: a Read the model made is recorded with its result. The file word reaches the answer only that way.
    reads = [call for call in first.tool_calls if call.name == 'Read']
    assert 'HERON-7' in first.text
    assert reads
    assert all(call.status.value == 'completed' for call in reads)
    assert any('HERON-7' in (call.result or '') for call in reads)
    # Session: the remembered word is in no file, so only the resumed conversation can supply it.
    assert 'PELICAN-42' in second.text
    assert first_session
    assert second_session
    assert second_session != first_session  # each turn forks the session it resumes
    # Isolation: nothing is written outside the working directory, and a Write the model attempted is recorded as refused.
    assert not outside.exists()
    assert all(call.status.value == 'incomplete' for call in third.tool_calls if call.name == 'Write')
    assert workdir is not None
    assert not workdir.exists()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_wrong_key_fails_fast_because_the_runtime_does_not_retry_on_its_own() -> None:
    """The options leave ``CLAUDE_CODE_MAX_RETRIES`` unset, so the target's default turns the runtime's retry off.

    With the runtime's own retry on, a wrong key took 174 s to fail before ``call_target_with_retry`` saw it.
    """
    options = _options(token='wrong-key')
    assert 'CLAUDE_CODE_MAX_RETRIES' not in options.env
    target = ClaudeAgentSDKTarget(options)
    started = time.monotonic()
    try:
        result = await call_target_with_retry(
            target,
            [Message(role='user', content='Say hello.')],
            target_agent_timeout_ms=90_000,
            max_target_retries=0,
            map_error=target.map_error,
        )
    finally:
        await target.close()
    elapsed = time.monotonic() - started
    print(f'wrong key failed after {elapsed:.1f} s with {result.error_payload().get("error_code")}')

    assert result.error is not None
    assert result.attempts == 1
    assert elapsed < 60, f'a wrong key took {elapsed:.0f} s to fail'
