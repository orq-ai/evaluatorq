"""Execution backends evaluatorq owns: targets that run a system under test locally.

`CodingAgentTarget` runs an installed coding-agent CLI (Claude Code, Codex CLI, OpenCode) as an
``AgentTarget``, directly or through ``orq launch``. Sandboxed and remote variants land here too.
"""

from evaluatorq.backends.coding_agent import (
    CodingAgentError,
    CodingAgentTarget,
    CodingAgentUnavailableError,
    OrqLaunchOptions,
)

__all__ = ['CodingAgentError', 'CodingAgentTarget', 'CodingAgentUnavailableError', 'OrqLaunchOptions']
