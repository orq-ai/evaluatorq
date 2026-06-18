"""Shared launcher for the Streamlit report dashboards.

Both the red team and agent-simulation CLIs expose an ``ui`` command that boots
a Streamlit app as a subprocess, handing it a report path after the ``--``
separator. The mechanics (streamlit import check, ``streamlit run`` invocation,
exit-code propagation) are identical, so they live here rather than being copied
into each CLI.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from pathlib import Path


def _install_hint(extra: str) -> str:
    return (
        f"Streamlit is not installed. Install the {extra} extras:\n"
        f'  uv pip install "evaluatorq[{extra}]"\n'
        f'  pip install "evaluatorq[{extra}]"'
    )


def ensure_streamlit(extra: str) -> None:
    """Exit with a friendly hint if Streamlit (an optional extra) is missing.

    ``extra`` names the optional dependency group to install — each caller passes
    its own so the hint points at the right one (``redteam`` vs ``simulation``).
    """
    try:
        import streamlit  # noqa: F401
    except ImportError:
        typer.echo(_install_hint(extra), err=True)
        raise typer.Exit(code=1) from None


def launch_streamlit(
    dashboard_script: Path,
    report_path: Path,
    *,
    port: int = 8501,
    host: str = "localhost",
    extra: str,
) -> None:
    """Run ``streamlit run <dashboard_script> -- <report_path>`` and exit.

    The report path is passed after ``--`` so Streamlit forwards it to the
    dashboard's ``sys.argv`` rather than treating it as a Streamlit flag.
    """
    if not dashboard_script.exists():
        typer.echo("Error: Dashboard module not found.", err=True)
        raise typer.Exit(code=1)

    ensure_streamlit(extra)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_script),
            "--server.port",
            str(port),
            "--server.address",
            host,
            "--browser.gatherUsageStats",
            "false",
            "--",
            str(report_path),
        ],
        check=False,
    )
    raise typer.Exit(code=result.returncode)
