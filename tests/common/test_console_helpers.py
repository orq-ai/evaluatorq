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
