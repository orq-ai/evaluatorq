"""Shared TTY helpers for the evaluatorq CLIs."""

from __future__ import annotations

import shlex
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def should_skip_confirm(yes: bool) -> bool:  # noqa: FBT001
    """Return True when the confirmation prompt must be skipped.

    Skip when the user passed --yes, or when stdin is not a TTY (CI, pipes) —
    otherwise ``typer.confirm`` blocks forever waiting on input nobody will send.
    """
    return yes or not sys.stdin.isatty()


def shell_join(args: Sequence[str]) -> str:
    """Join *args* into a command the user can paste into their own shell.

    ``shlex`` quoting is POSIX-only: cmd.exe and PowerShell do not unwrap single quotes, and backslashes
    in a Windows path make ``shlex`` quote every path. Windows gets ``subprocess.list2cmdline``.
    """
    if sys.platform == 'win32':
        return subprocess.list2cmdline(args)
    return shlex.join(args)


def shell_path(path: str | Path) -> str:
    """Quote *path* for a copy-pasteable command in the user's shell."""
    return shell_join([str(path)])
