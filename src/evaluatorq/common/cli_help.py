"""Shared Typer context settings and help text for the evaluatorq CLIs."""

from __future__ import annotations

from evaluatorq.contracts import DEFAULT_PIPELINE_MODEL

# clig.dev: both -h and --help must show help. Typer only wires --help by
# default, so add -h explicitly. Defined once to keep the three Typer apps in sync.
CONTEXT_SETTINGS: dict[str, list[str]] = {'help_option_names': ['-h', '--help']}

# Derived, never written out: a hand-typed example goes stale the next time the
# default moves, and --help is the one place nothing checks.
BARE_DEFAULT_MODEL: str = DEFAULT_PIPELINE_MODEL.rsplit('/', 1)[-1]

MODEL_OPTION_NOTE: str = (
    'Provider resolved from env: ORQ_API_KEY -> Orq router, else OPENAI_API_KEY '
    '(+ OPENAI_BASE_URL) -> OpenAI-compatible endpoint. The default is '
    'provider-prefixed for the router; going OpenAI-direct means overriding it '
    f"with the bare id ('{BARE_DEFAULT_MODEL}')."
)
