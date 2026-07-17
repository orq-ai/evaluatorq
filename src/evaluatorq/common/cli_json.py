"""Canonical machine-readable JSON output for the evaluatorq CLIs."""

from __future__ import annotations

import json
from typing import Any

import typer


def echo_json(obj: Any) -> None:
    """Print ``obj`` as indented JSON to stdout (nothing else on stdout)."""
    typer.echo(json.dumps(obj, indent=2, default=str))
