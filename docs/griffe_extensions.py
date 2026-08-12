"""Griffe extensions applied while mkdocstrings collects the API surface."""

from __future__ import annotations

import griffe


class DropPrivateParameters(griffe.Extension):
    """Hide ``_``-prefixed parameters from rendered signatures.

    `evaluatorq()` takes five internal keyword arguments (``_send_results``,
    ``_base_url``, ``_trace_type``, ``_exit_on_failure``, ``_experiment_url_out``)
    that the CLI and the simulation/red-team runners pass to each other. They are
    not callable surface, but mkdocstrings renders the signature verbatim, so the
    library's headline function opened on a wall of internals. There is no
    per-parameter filter option — `filters` selects members, not parameters — so
    strip them at collection time instead.

    Signature-only: the underlying function is untouched, and callers that already
    pass these keywords keep working.
    """

    def on_function_instance(self, *, func: griffe.Function, **kwargs: object) -> None:  # noqa: ARG002
        keep = [p for p in func.parameters if not p.name.startswith("_")]
        if len(keep) != len(func.parameters):
            func.parameters = griffe.Parameters(*keep)
