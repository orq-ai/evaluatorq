"""Live check of ClaudeAgentSDKTarget: real Claude Code runtime, model calls through the Orq router.

Needs ``ORQ_API_KEY``. Runs two short turns on a small model and one attempt to leave the working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip('claude_agent_sdk')

from claude_agent_sdk import ClaudeAgentOptions

from evaluatorq.contracts import Message
from evaluatorq.integrations.claude_agent_sdk_integration import ClaudeAgentSDKTarget

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get('ORQ_API_KEY'), reason='needs ORQ_API_KEY'),
]


def _options() -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt='You are a file assistant. Always call the tool the user names. Answer in one sentence.',
        model='claude-haiku-5-5',
        tools=['Read', 'Write'],
        allowed_tools=['Read', 'Write'],
        permission_mode='acceptEdits',
        max_turns=4,
        env={
            'ANTHROPIC_BASE_URL': 'https://my.orq.ai/v3/anthropic',
            'ANTHROPIC_AUTH_TOKEN': os.environ['ORQ_API_KEY'],
            'ANTHROPIC_API_KEY': '',
            'ANTHROPIC_DEFAULT_HAIKU_MODEL': 'claude-haiku-5-5',
        },
    )


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_a_real_agent_keeps_its_session_and_cannot_write_outside_its_working_directory(tmp_path: Path) -> None:
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'notes.txt').write_text('The code word is PELICAN-42.\n', encoding='utf-8')
    outside = tmp_path / 'outside.txt'

    target = ClaudeAgentSDKTarget(_options(), workdir=project)
    try:
        first = await target.respond([Message(role='user', content='Call the Read tool on notes.txt and tell me the code word.')])
        first_session = target.session_id
        second = await target.respond([Message(role='user', content='Without any tool, repeat the code word.')])
        third = await target.respond([
            Message(role='user', content=f'Call the Write tool with file_path {outside} and content: escaped')
        ])
        workdir = target.workdir
    finally:
        await target.close()

    read = first.tool_calls[0]
    assert (read.name, read.status.value) == ('Read', 'completed')
    assert 'PELICAN-42' in (read.result or '')
    assert 'PELICAN-42' in first.text
    # The second turn has no file access of its own: it knows the word only through the resumed session.
    assert second.tool_calls == []
    assert 'PELICAN-42' in second.text
    assert first_session and target.session_id is None and second.response_id != first_session
    assert [(call.name, call.status.value) for call in third.tool_calls] == [('Write', 'incomplete')]
    assert not outside.exists()
    assert workdir is not None and not workdir.exists()
