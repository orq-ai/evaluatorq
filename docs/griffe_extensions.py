"""Griffe extensions applied while mkdocstrings collects the API surface."""

from __future__ import annotations

import griffe


# Functions whose ``_``-prefixed parameters are internal plumbing rather than callable
# surface. Named explicitly: an underscore prefix is a convention, not a guarantee, and a
# blanket rule would silently erase a parameter some future function documents on purpose
# — with nothing failing to say so.
_STRIP_PRIVATE_PARAMS = frozenset({"evaluatorq.evaluatorq.evaluatorq"})


class DropPrivateParameters(griffe.Extension):
    """Hide ``_``-prefixed parameters from the signatures listed in `_STRIP_PRIVATE_PARAMS`.

    `evaluatorq()` takes five internal keyword arguments (``_send_results``,
    ``_base_url``, ``_trace_type``, ``_exit_on_failure``, ``_experiment_url_out``)
    that the CLI and the simulation/red-team runners pass to each other. They are
    not callable surface, but mkdocstrings renders the signature verbatim, so the
    library's headline function opened on a wall of internals. There is no
    per-parameter filter option — `filters` selects members, not parameters — so
    strip them at collection time instead.

    Signature-only: the underlying function is untouched, and callers that already
    pass these keywords keep working.

    Matched on `func.path`, so renaming or moving the function drops the filter and
    the internals reappear on the page — visible, rather than a silent mismatch.
    """

    def on_function_instance(self, *, func: griffe.Function, **kwargs: object) -> None:  # noqa: ARG002
        if func.path not in _STRIP_PRIVATE_PARAMS:
            return
        keep = [p for p in func.parameters if not p.name.startswith("_")]
        if len(keep) != len(func.parameters):
            func.parameters = griffe.Parameters(*keep)
