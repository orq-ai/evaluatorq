"""Shared shell command classification."""

from __future__ import annotations

import pytest

from evaluatorq.signals.shell import shell_command_family


@pytest.mark.parametrize('separator', [';', '&&'])
def test_shell_family_skips_leading_directory_change(separator: str) -> None:
    assert shell_command_family(f'cd /tmp {separator} git status --short') == 'git status'
