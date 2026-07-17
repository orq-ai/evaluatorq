"""Shared Typer context settings for the evaluatorq CLIs."""

from __future__ import annotations

# clig.dev: both -h and --help must show help. Typer only wires --help by
# default, so add -h explicitly. Defined once to keep the three Typer apps in sync.
CONTEXT_SETTINGS: dict[str, list[str]] = {'help_option_names': ['-h', '--help']}
