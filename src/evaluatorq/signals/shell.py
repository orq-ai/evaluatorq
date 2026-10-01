"""Shared shell command classification for structure and tool signals."""

from __future__ import annotations

import re
import shlex

_CLIS = frozenset({'git', 'orq', 'gh', 'uv', 'docker', 'kubectl', 'npm', 'pnpm', 'yarn', 'bun', 'cargo', 'go'})
_ASSIGNMENT = re.compile(r'[A-Za-z_][A-Za-z_0-9]*=.*')


def shell_command_family(command: object) -> str | None:
    """Read the first command family after leading assignments and directory changes."""
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=';&|')
        lexer.whitespace_split = True
        lexer.commenters = '#'
        tokens = list(lexer)
    except ValueError:
        return None
    while tokens:
        if _ASSIGNMENT.fullmatch(tokens[0]):
            tokens.pop(0)
        elif len(tokens) >= 3 and tokens[0] == 'cd' and tokens[2] == '&&':
            tokens = tokens[3:]
        else:
            break
    if not tokens or tokens[0] in {';', '&&', '||', '|'}:
        return None
    executable = tokens[0].rsplit('/', 1)[-1]
    if (
        executable in _CLIS
        and len(tokens) > 1
        and tokens[1] not in {';', '&&', '||', '|'}
        and not tokens[1].startswith('-')
    ):
        return f'{executable} {tokens[1]}'
    return executable
