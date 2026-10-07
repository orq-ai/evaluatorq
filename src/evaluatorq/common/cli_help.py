"""Shared Typer context settings and help text for the evaluatorq CLIs."""

from __future__ import annotations

# clig.dev: both -h and --help must show help. Typer only wires --help by
# default, so add -h explicitly. Defined once to keep the three Typer apps in sync.
CONTEXT_SETTINGS: dict[str, list[str]] = {'help_option_names': ['-h', '--help']}

MODEL_OPTION_NOTE: str = (
    'Provider resolved from env: ORQ_API_KEY -> Orq router, else OPENAI_API_KEY '
    '(+ OPENAI_BASE_URL) -> OpenAI-compatible endpoint. The role defaults are '
    'provider-prefixed for the router; going OpenAI-direct means passing the bare id.'
)
