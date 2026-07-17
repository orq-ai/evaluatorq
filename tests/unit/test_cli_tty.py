# ruff: noqa: S101, FBT003

import sys

from evaluatorq.common.cli_tty import should_skip_confirm


def test_skip_when_non_tty(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert should_skip_confirm(False) is True


def test_no_skip_when_tty_and_no_yes(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    assert should_skip_confirm(False) is False


def test_yes_always_skips(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    assert should_skip_confirm(True) is True
