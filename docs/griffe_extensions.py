"""Griffe extensions applied while mkdocstrings collects the API surface."""

from __future__ import annotations

import ast

import griffe


# Functions whose ``_``-prefixed parameters are internal plumbing rather than callable
# surface. Named explicitly: an underscore prefix is a convention, not a guarantee, and a
# blanket rule would silently erase a parameter some future function documents on purpose
# — with nothing failing to say so.
_STRIP_PRIVATE_PARAMS = frozenset({"evaluatorq.evaluatorq.evaluatorq"})


class DropPrivateParameters(griffe.Extension):
    """Hide ``_``-prefixed parameters from the signatures listed in `_STRIP_PRIVATE_PARAMS`.

    `evaluatorq()` takes four internal keyword arguments (``_send_results``,
    ``_base_url``, ``_trace_type``, ``_experiment_url_out``)
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


class FieldDescriptionDocstrings(griffe.Extension):
    """Render a pydantic ``Field(description=...)`` as the attribute's docstring.

    Griffe reads attribute docstrings, not ``Field`` arguments, so without this every
    ``description=`` on the result models (most of them, in ``redteam/contracts.py``)
    is missing from the API reference. An explicit attribute docstring wins.

    Only literal strings are read. A description built from a constant
    (``description=_RATE_NONE_DOC``) stays unrendered rather than importing the module.
    """

    def on_attribute_instance(self, *, attr: griffe.Attribute, **kwargs: object) -> None:  # noqa: ARG002
        value = attr.value
        if attr.docstring or not isinstance(value, griffe.ExprCall) or str(value.function) != "Field":
            return
        for arg in value.arguments:
            if isinstance(arg, griffe.ExprKeyword) and arg.name == "description":
                try:
                    text = ast.literal_eval(str(arg.value))
                except (ValueError, SyntaxError):
                    return
                if isinstance(text, str):
                    attr.docstring = griffe.Docstring(text, parent=attr)
                return
