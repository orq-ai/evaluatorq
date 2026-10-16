"""Tests for evaluatorq.common.reports.console helpers."""
from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
from rich.console import Console

from evaluatorq.common.reports import ask_confirm, confirm_run_plan, write_text_report


def test_confirm_skip_renders_table_and_returns_true() -> None:
    buf = io.StringIO()
    ok = asyncio.run(
        confirm_run_plan(
            Console(file=buf, width=80, force_terminal=False),
            title="Run Plan",
            rows=[("Model", "gpt-x"), ("Datapoints", "3")],
            prompt="Go?",
            skip_confirm=True,
        )
    )
    assert ok is True
    out = buf.getvalue()
    assert "Run Plan" in out
    assert "Model" in out
    assert "gpt-x" in out


def test_confirm_skip_false_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    """When skip_confirm=False it offloads typer.confirm; monkeypatch returns True."""
    import typer

    monkeypatch.setattr(typer, "confirm", lambda *_a, **_kw: True)
    buf = io.StringIO()
    ok = asyncio.run(
        confirm_run_plan(
            Console(file=buf, width=80, force_terminal=False),
            title="Run Plan",
            rows=[("Model", "gpt-x")],
            prompt="Go?",
            skip_confirm=False,
        )
    )
    assert ok is True


def test_ask_confirm_returns_the_answer_and_keeps_the_loop_running(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    import typer

    answered = threading.Event()

    def confirm(*_a: object, **_kw: object) -> bool:
        answered.wait(timeout=5)
        return False

    monkeypatch.setattr(typer, "confirm", confirm)

    async def main() -> bool:
        pending = asyncio.ensure_future(ask_confirm("Go?"))
        await asyncio.sleep(0)
        assert not pending.done()
        answered.set()
        return await pending

    assert asyncio.run(main()) is False


def test_ask_confirm_cancellation_returns_at_once_and_drops_the_late_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    import typer

    release = threading.Event()
    finished = threading.Event()

    def confirm(*_a: object, **_kw: object) -> bool:
        release.wait(timeout=5)
        finished.set()
        return True

    monkeypatch.setattr(typer, "confirm", confirm)

    async def main() -> None:
        pending = asyncio.ensure_future(ask_confirm("Go?"))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(main())
    release.set()
    assert finished.wait(timeout=5)


def test_ask_confirm_reraises_the_prompt_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import click
    import typer

    def abort(*_a: object, **_kw: object) -> bool:
        raise click.Abort

    monkeypatch.setattr(typer, "confirm", abort)

    with pytest.raises(click.Abort):
        asyncio.run(ask_confirm("Go?"))


class _PipeStdin:
    """Stands in for ``sys.stdin``: the terminal reader only needs ``fileno`` and ``encoding``."""

    encoding = "utf-8"

    def __init__(self, fd: int) -> None:
        self._fd = fd

    def fileno(self) -> int:
        return self._fd


def _terminal_answer(monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str], *chunks: bytes) -> bool:
    import os
    import sys

    from evaluatorq.common.reports import console

    read_fd, write_fd = os.pipe()
    for chunk in chunks:
        os.write(write_fd, chunk)
    os.close(write_fd)
    monkeypatch.setattr(sys, "stdin", _PipeStdin(read_fd))
    try:
        return console._confirm_from_terminal("Go?")
    finally:
        os.close(read_fd)


@pytest.mark.parametrize(
    ("typed", "expected"),
    [(b"y\n", True), (b"YES\n", True), (b"\n", True), (b"n\n", False), (b" no \n", False)],
)
def test_terminal_confirm_answers_like_typer_confirm(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str], typed: bytes, expected: bool
) -> None:
    assert _terminal_answer(monkeypatch, capfd, typed) is expected
    assert capfd.readouterr().out == ""


def test_terminal_confirm_reprompts_after_invalid_input(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    assert _terminal_answer(monkeypatch, capfd, b"maybe\n", b"n\n") is False
    assert capfd.readouterr().out == "Error: invalid input\nGo? [Y/n]: "


def test_terminal_confirm_joins_a_line_split_across_reads(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    import os

    from evaluatorq.common.reports import console

    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"n")
    reads: list[bytes] = []
    real_read = os.read

    def read_then_finish_the_line(fd: int, n: int) -> bytes:
        chunk = real_read(fd, n)
        reads.append(chunk)
        if len(reads) == 1:
            os.write(write_fd, b"o\n")
        return chunk

    monkeypatch.setattr(console.os, "read", read_then_finish_the_line)
    try:
        assert console._read_terminal_line(read_fd) == "no\n"
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert reads == [b"n", b"o", b"\n"]


def test_terminal_confirm_aborts_on_end_of_input(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    import click

    with pytest.raises(click.Abort):
        _terminal_answer(monkeypatch, capfd)


def test_write_text_report_creates_file(tmp_path: Path) -> None:
    out_dir = tmp_path / "reports"
    path = write_text_report(out_dir, stem="my-report", fmt="md", content="# Hello")
    assert path == out_dir / "my-report.md"
    assert path.read_text(encoding="utf-8") == "# Hello"


def test_write_text_report_nested_dirs(tmp_path: Path) -> None:
    out_dir = tmp_path / "a" / "b" / "c"
    path = write_text_report(out_dir, stem="report", fmt="html", content="<html/>")
    assert path.exists()
    assert path.name == "report.html"


def test_write_text_report_oserror_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OSError on mkdir raises typer.Exit(1)."""
    import typer

    bad_dir = tmp_path / "bad"

    def _raise(*_a: object, **_kw: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "mkdir", _raise)
    with pytest.raises(typer.Exit):
        write_text_report(bad_dir, stem="r", fmt="md", content="x")
