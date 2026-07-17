"""Shared TTY helpers for the evaluatorq CLIs."""

from __future__ import annotations

import sys


def should_skip_confirm(yes: bool) -> bool:  # noqa: FBT001
    """Return True when the confirmation prompt must be skipped.

    Skip when the user passed --yes, or when stdin is not a TTY (CI, pipes) —
    otherwise ``typer.confirm`` blocks forever waiting on input nobody will send.
    """
    return yes or not sys.stdin.isatty()
